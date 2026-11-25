//! A generation session: owns the context (tokens + KV/conv state on the GPU), prefill and decode loops.

use crate::engine::{Engine, EngineCheckpoint};
use crate::metal::CmdBuf;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// RAII guard for in-flight Metal command buffers: ensures every committed `CmdBuf`
/// finishes on the GPU even if a callback (such as `on_token`) panics and unwinds.
pub struct InflightGuard(pub VecDeque<(usize, CmdBuf)>);

impl InflightGuard {
    pub fn new() -> Self {
        Self(VecDeque::new())
    }
}

impl Default for InflightGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        while let Some((_, cb)) = self.0.pop_front() {
            let st = cb.status();
            if st != MTLCommandBufferStatus::NotEnqueued
                && st != MTLCommandBufferStatus::Completed
                && st != MTLCommandBufferStatus::Error
            {
                cb.waitUntilCompleted();
            }
        }
    }
}

pub struct Session<'a> {
    pub eng: &'a Engine,
    pub tk: &'a Tokenizer,
    /// tokens whose K/V and conv state are in the cache
    pub n_past: usize,
    /// a sampled token not yet fed through the model (it becomes the first input of the next step)
    pending: Option<u32>,
    pub stats: Stats,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub prompt_time: Duration,
    pub gen_tokens: usize,
    pub gen_time: Duration,
    /// largest activation buffer (decode arena + prefill arena) any prefill needed, bytes
    pub peak_scratch: usize,
    /// number of decode steps where a non-finite (`NaN` or `±Inf`) logit was detected
    pub non_finite_logits: usize,
}

impl<'a> Session<'a> {
    pub fn new(eng: &'a Engine, tk: &'a Tokenizer) -> Session<'a> {
        eng.reset();
        Session { eng, tk, n_past: 0, pending: None, stats: Stats::default() }
    }

    pub fn ctx_left(&self) -> usize { self.eng.max_ctx.saturating_sub(self.n_past + 1) }

    /// Fallible version of [`Session::prefill`]: validates non-empty input and context
    /// capacity before consuming any pending token or submitting GPU work.
    pub fn try_prefill(&mut self, toks: &[u32]) -> Result<u32, String> {
        let end = validate_prefill(
            self.n_past,
            self.pending.is_some(),
            toks.len(),
            self.eng.max_ctx,
        )?;
        let mut all: Vec<u32> = self.pending.take().into_iter().collect();
        all.extend_from_slice(toks);
        let t0 = Instant::now();
        self.eng.tier_leave(self.n_past);       // (--kv-disk) a prefill attends through the whole cache in memory
        self.eng.set_tokens(self.n_past, &all);
        self.eng.reserve(end + 1, true);
        self.eng.ensure_batch(all.len().min(self.eng.chunk));
        self.stats.peak_scratch = self.stats.peak_scratch.max(self.eng.mem().1);
        let mut p0 = self.n_past;
        let mut inflight = InflightGuard::new();
        while p0 < end {
            let t = (end - p0).min(self.eng.chunk);
            let cb = self.eng.gpu.cmd();
            self.eng.encode_prefill(&cb, p0, t, p0 + t == end, true);
            cb.commit();
            inflight.0.push_back((p0, cb));
            p0 += t;
        }
        if let Some((_, last)) = inflight.0.pop_back() {
            last.waitUntilCompleted();
        }
        inflight.0.clear();
        self.n_past = end;
        self.stats.prompt_tokens += all.len();
        self.stats.prompt_time += t0.elapsed();
        Ok(self.eng.token_at(end))
    }

    /// Feed `toks` (preceded by any pending sampled token) through the model in prefill chunks.
    /// Returns the greedy next token (argmax of the last position); logits stay available in the engine.
    pub fn prefill(&mut self, toks: &[u32]) -> u32 {
        self.try_prefill(toks).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Generate up to `max` tokens after a prefill. `on_token` returns false to stop early.
    /// Greedy decoding is fully pipelined on the GPU: the argmax of step p is the embedding input of step p+1, so the CPU
    /// never sits between steps. Exactly two steps are kept in flight, so the in-flight step always processes the token
    /// just emitted: when generation stops, the context (KV + conv state) is exactly "everything emitted so far".
    /// Sampling reads the logits back each step.
    pub fn generate(&mut self, first: u32, max: usize, sampler: &mut Sampler, on_token: impl FnMut(u32) -> bool) -> usize {
        let mut state = DecodeState {
            eos: self.tk.eos,
            n_past: &mut self.n_past,
            pending: &mut self.pending,
            stats: &mut self.stats,
        };
        generate_with_backend(self.eng, &mut state, first, max, sampler, on_token)
    }

    /// Return any sampled token awaiting prefill/step ingestion (for example, `eos` at turn end).
    pub fn pending(&self) -> Option<u32> {
        self.pending
    }

    /// Set or clear the pending sampled token. External decode loops should stash the
    /// terminating token here before returning control so the next `try_prefill` ingests it.
    pub fn set_pending(&mut self, tok: Option<u32>) {
        self.pending = tok;
    }

    /// Advance the model by one autoregressive token step and leave logits ready in
    /// `self.eng.logits()` / `self.eng.logits_mut()` for external logit hooks and sampling.
    pub fn try_step(&mut self, tok: u32) -> Result<(), String> {
        let mut state = DecodeState {
            eos: self.tk.eos,
            n_past: &mut self.n_past,
            pending: &mut self.pending,
            stats: &mut self.stats,
        };
        step_with_backend(self.eng, &mut state, tok)
    }

    /// Capture a complete hybrid checkpoint (`n_past`, `pending`, conv state, SWA + full KV, tokens).
    pub fn checkpoint(&self) -> Result<SessionCheckpoint, String> {
        let engine = self.eng.checkpoint(self.n_past)?;
        Ok(SessionCheckpoint {
            n_past: self.n_past,
            pending: self.pending,
            engine,
        })
    }

    /// Restore a previously captured [`SessionCheckpoint`].
    pub fn restore_checkpoint(&mut self, ckpt: &SessionCheckpoint) -> Result<(), String> {
        self.eng.restore_checkpoint(&ckpt.engine)?;
        self.n_past = ckpt.n_past;
        self.pending = ckpt.pending;
        Ok(())
    }
}

/// Complete snapshot of a [`Session`]'s logical cursor (`n_past`, `pending`) and hybrid GPU state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionCheckpoint {
    pub n_past: usize,
    pub pending: Option<u32>,
    engine: EngineCheckpoint,
}

impl SessionCheckpoint {
    /// Total heap bytes held by the underlying hybrid checkpoint.
    pub fn byte_size(&self) -> usize {
        self.engine.byte_size()
    }

    /// Resident token IDs `[0..n_past)` captured by this checkpoint.
    pub fn tokens(&self) -> &[u32] {
        self.engine.tokens()
    }
}

/// Validate that one additional autoregressive token step fits in `max_ctx`.
pub fn validate_step(n_past: usize, max_ctx: usize) -> Result<usize, String> {
    let end = n_past
        .checked_add(1)
        .ok_or_else(|| "context length overflow".to_string())?;
    if end >= max_ctx {
        return Err(format!(
            "context full ({max_ctx} tokens; cannot step at position {n_past})"
        ));
    }
    Ok(end)
}

/// Single-token autoregressive decode step shared by [`Session::try_step`] and unit tests.
pub fn step_with_backend<B: DecodeBackend>(
    eng: &B,
    state: &mut DecodeState<'_>,
    tok: u32,
) -> Result<(), String> {
    let end = validate_step(*state.n_past, eng.max_ctx())?;
    let t0 = Instant::now();
    *state.pending = None;
    eng.release_batch();
    eng.set_token(*state.n_past, tok);
    eng.gen_reserve(*state.n_past);
    let cb = eng.launch_step(*state.n_past, false);
    eng.wait_step(&cb);
    *state.n_past = end;
    state.stats.gen_tokens += 1;
    state.stats.gen_time += t0.elapsed();
    Ok(())
}

/// RAII guard for a single Metal command buffer in [`DecodeBackend`].
pub struct CmdGuard(CmdBuf);

impl Drop for CmdGuard {
    fn drop(&mut self) {
        let st = self.0.status();
        if st != MTLCommandBufferStatus::NotEnqueued
            && st != MTLCommandBufferStatus::Completed
            && st != MTLCommandBufferStatus::Error
        {
            self.0.waitUntilCompleted();
        }
    }
}

/// Minimal execution backend used by [`Session::generate`], enabling deterministic
/// unit testing of decode loop control flow without a full GPU model.
pub trait DecodeBackend {
    type Step;
    fn max_ctx(&self) -> usize;
    fn release_batch(&self);
    fn sample_logits(&self, sampler: &mut Sampler) -> u32;
    fn set_token(&self, pos: usize, tok: u32);
    fn gen_reserve(&self, pos: usize);
    fn gen_capacity(&self) -> usize;
    fn launch_step(&self, pos: usize, argmax: bool) -> Self::Step;
    fn wait_step(&self, step: &Self::Step);
    fn token_at(&self, pos: usize) -> u32;
    /// Read the logit for a single token ID without copying the vocabulary slice.
    /// Defaults to `0.0` (finite) so existing mock backends compile unchanged.
    fn logit_at(&self, _tok: usize) -> f32 {
        0.0
    }
    /// Sample a token from the current logits and report whether any non-finite
    /// logit was detected (including `Err` from all-`NaN` logits).
    fn sample_logits_checked(&self, sampler: &mut Sampler) -> (u32, bool) {
        let tok = self.sample_logits(sampler);
        let bad = !self.logit_at(tok as usize).is_finite();
        (tok, bad)
    }
}

/// Sample from `logits` and return `(token, non_finite_detected)`.
/// Flags `non_finite_detected = true` if `Sampler::try_sample` fails (e.g., empty or all-`NaN`),
/// if the winning logit is non-finite, or if any logit in `logits` is non-finite.
pub fn sample_logits_with_canary(logits: &[f32], sampler: &mut Sampler) -> (u32, bool) {
    match sampler.try_sample(logits) {
        Ok(tok) => {
            let win_bad = !logits
                .get(tok as usize)
                .copied()
                .unwrap_or(f32::NAN)
                .is_finite();
            let any_bad = win_bad || logits.iter().any(|x| !x.is_finite());
            (tok, any_bad)
        }
        Err(_) => (0, true),
    }
}

impl DecodeBackend for Engine {
    type Step = CmdGuard;
    fn max_ctx(&self) -> usize { self.max_ctx }
    fn release_batch(&self) { Engine::release_batch(self); }
    fn sample_logits(&self, sampler: &mut Sampler) -> u32 { sampler.sample(&self.logits()) }
    fn sample_logits_checked(&self, sampler: &mut Sampler) -> (u32, bool) {
        sample_logits_with_canary(&self.logits(), sampler)
    }
    fn set_token(&self, pos: usize, tok: u32) { self.set_tokens(pos, &[tok]); }
    fn gen_reserve(&self, pos: usize) { Engine::gen_reserve(self, pos); }
    fn gen_capacity(&self) -> usize { Engine::gen_capacity(self) }
    fn launch_step(&self, pos: usize, argmax: bool) -> CmdGuard {
        CmdGuard(self.step(pos, true, argmax))
    }
    fn wait_step(&self, step: &CmdGuard) { step.0.waitUntilCompleted(); }
    fn token_at(&self, pos: usize) -> u32 { Engine::token_at(self, pos) }
    fn logit_at(&self, tok: usize) -> f32 { Engine::logit_at(self, tok) }
}

/// Mutable cursor state passed into [`generate_with_backend`].
pub struct DecodeState<'a> {
    pub eos: u32,
    pub n_past: &'a mut usize,
    pub pending: &'a mut Option<u32>,
    pub stats: &'a mut Stats,
}

/// Core decode loop shared by [`Session::generate`] and unit tests.
/// Validate prefill token count and context capacity before touching the GPU.
/// Returns the new context length `end = n_past + pending + toks_len` if valid.
pub fn validate_prefill(
    n_past: usize,
    has_pending: bool,
    toks_len: usize,
    max_ctx: usize,
) -> Result<usize, String> {
    let total_new = usize::from(has_pending)
        .checked_add(toks_len)
        .ok_or_else(|| "prefill token count overflow".to_string())?;
    if total_new == 0 {
        return Err("empty prompt: cannot prefill 0 tokens".to_string());
    }
    let end = n_past
        .checked_add(total_new)
        .ok_or_else(|| "context length overflow".to_string())?;
    if end >= max_ctx {
        return Err(format!(
            "context full ({max_ctx} tokens; need {total_new} with {n_past} in context)"
        ));
    }
    Ok(end)
}

pub fn generate_with_backend<B: DecodeBackend>(
    eng: &B,
    state: &mut DecodeState<'_>,
    first: u32,
    max: usize,
    sampler: &mut Sampler,
    mut on_token: impl FnMut(u32) -> bool,
) -> usize {
    let t0 = Instant::now();
    let eos = state.eos;
    eng.release_batch();      // the prefill arena is not needed while generating
    if max == 0 {
        return 0;
    }
    let mut non_finite = 0usize;
    let mut record_non_finite = |tok: u32| {
        if non_finite == 0 {
            eprintln!(
                "[inkling] WARNING: non-finite logit detected (token {tok}); re-run with INK_SAFE_MATH=1 or run `inkling verify`"
            );
        }
        non_finite += 1;
    };
    let ctx_left = |n_past: usize| eng.max_ctx().saturating_sub(n_past + 1);
    let mut tok = if sampler.greedy() {
        if !eng.logit_at(first as usize).is_finite() {
            record_non_finite(first);
        }
        first
    } else {
        let (t, bad) = eng.sample_logits_checked(sampler);
        if bad {
            record_non_finite(t);
        }
        t
    };
    let mut n = 1usize;
    let mut stop = tok == eos || !on_token(tok) || n >= max || ctx_left(*state.n_past) < 2;
    if sampler.greedy() {
        if stop { *state.pending = Some(tok); }
        else {
            eng.set_token(*state.n_past, tok);
            let mut inflight: VecDeque<(usize, B::Step)> = VecDeque::new();
            let step = |pos: usize| (pos, eng.launch_step(pos, true));
            eng.gen_reserve(*state.n_past);
            inflight.push_back(step(*state.n_past));
            loop {
                if inflight.len() < 2 {
                    let pos = inflight.back().unwrap().0 + 1;
                    if pos >= eng.gen_capacity() {
                        // the K/V buffers must grow (or, with --kv-disk, be re-sized): nothing may be in flight meanwhile
                        for (_, cb) in inflight.iter() { eng.wait_step(cb); }
                        eng.gen_reserve(pos);
                    }
                    inflight.push_back(step(pos));
                }
                let (pos, cb) = inflight.pop_front().unwrap();
                eng.wait_step(&cb);
                *state.n_past = pos + 1;
                tok = eng.token_at(pos + 1);
                if !eng.logit_at(tok as usize).is_finite() {
                    record_non_finite(tok);
                }
                n += 1;
                stop = tok == eos || !on_token(tok) || n >= max || ctx_left(*state.n_past) < 3;
                if stop {
                    // the remaining in-flight step consumes `tok` (a real token): keep it in the context
                    match inflight.pop_front() {
                        Some((p2, cb2)) => { eng.wait_step(&cb2); *state.n_past = p2 + 1; *state.pending = None; }
                        None => *state.pending = Some(tok),
                    }
                    break;
                }
            }
        }
    } else {
        while !stop {
            eng.set_token(*state.n_past, tok);
            eng.gen_reserve(*state.n_past);
            let cb = eng.launch_step(*state.n_past, false);
            eng.wait_step(&cb);
            *state.n_past += 1;
            let (t, bad) = eng.sample_logits_checked(sampler);
            tok = t;
            if bad {
                record_non_finite(tok);
            }
            n += 1;
            stop = tok == eos || !on_token(tok) || n >= max || ctx_left(*state.n_past) < 2;
        }
        *state.pending = Some(tok);
    }
    state.stats.gen_tokens += n;
    state.stats.gen_time += t0.elapsed();
    state.stats.non_finite_logits += non_finite;
    n
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EscState {
    Normal,
    Esc,
    EscIntermediate,
    Csi,
    Osc,
    StString,
    OscEsc,
}

#[inline]
fn is_safe_normal_char(ch: char) -> bool {
    match ch {
        '\n' | '\r' | '\t' => true,
        '\x00'..='\x1F' | '\x7F' | '\u{0080}'..='\u{009F}' => false,
        _ => true,
    }
}

fn sanitize_char(ch: char, state: &mut EscState, clean: &mut String) {
    loop {
        match *state {
            EscState::Normal => {
                match ch {
                    '\x1b' => *state = EscState::Esc,
                    '\u{009B}' => *state = EscState::Csi,
                    '\u{009D}' => *state = EscState::Osc,
                    '\u{0090}' | '\u{0098}' | '\u{009E}' | '\u{009F}' => {
                        *state = EscState::StString
                    }
                    '\n' | '\r' | '\t' => clean.push(ch),
                    '\x00'..='\x1F' | '\x7F' | '\u{0080}'..='\u{009F}' => {}
                    _ => clean.push(ch),
                }
                return;
            }
            EscState::Esc => {
                match ch {
                    '[' => { *state = EscState::Csi; return; }
                    ']' => { *state = EscState::Osc; return; }
                    'P' | 'X' | '^' | '_' => { *state = EscState::StString; return; }
                    ' '..='/' => { *state = EscState::EscIntermediate; return; }
                    '0'..='~' => { *state = EscState::Normal; return; }
                    '\x1b' => return,
                    _ => { *state = EscState::Normal; }
                }
            }
            EscState::EscIntermediate => {
                match ch {
                    ' '..='/' => return,
                    '0'..='~' => { *state = EscState::Normal; return; }
                    '\x1b' => { *state = EscState::Esc; return; }
                    _ => { *state = EscState::Normal; }
                }
            }
            EscState::Csi => {
                match ch {
                    '0'..='?' | ' '..='/' => return,
                    '@'..='~' => { *state = EscState::Normal; return; }
                    '\x1b' => { *state = EscState::Esc; return; }
                    '\u{009B}' => return,
                    '\x18' | '\x1A' => { *state = EscState::Normal; return; }
                    _ => { *state = EscState::Normal; }
                }
            }
            EscState::Osc => {
                match ch {
                    '\x07' | '\u{009C}' | '\x18' | '\x1A' => { *state = EscState::Normal; return; }
                    '\x1b' => { *state = EscState::OscEsc; return; }
                    _ => return,
                }
            }
            EscState::StString => {
                match ch {
                    '\u{009C}' | '\x18' | '\x1A' => { *state = EscState::Normal; return; }
                    '\x1b' => { *state = EscState::OscEsc; return; }
                    _ => return,
                }
            }
            EscState::OscEsc => {
                if ch == '\\' {
                    *state = EscState::Normal;
                    return;
                }
                *state = EscState::Esc;
            }
        }
    }
}

fn write_text(s: &str, sanitize: bool, state: &mut EscState, out: &mut impl std::io::Write) {
    if !sanitize || (*state == EscState::Normal && s.chars().all(is_safe_normal_char)) {
        let _ = out.write_all(s.as_bytes());
        return;
    }
    let mut clean = String::with_capacity(s.len());
    for ch in s.chars() {
        sanitize_char(ch, state, &mut clean);
    }
    if !clean.is_empty() {
        let _ = out.write_all(clean.as_bytes());
    }
}

/// Streams model output: hides protocol tokens, dims the reasoning, emits valid UTF-8 only,
/// and strips raw terminal escape sequences when `sanitize_tty` is enabled.
pub struct Renderer {
    buf: Vec<u8>,
    pub thinking: bool,
    pub show_thinking: bool,
    pub raw: bool,
    pub sanitize_tty: bool,
    think_note: bool,
    esc_state: EscState,
}

impl Renderer {
    pub fn new(show_thinking: bool, raw: bool) -> Renderer {
        Renderer {
            buf: Vec::new(),
            thinking: false,
            show_thinking,
            raw,
            sanitize_tty: !raw,
            think_note: false,
            esc_state: EscState::Normal,
        }
    }

    /// Enable or disable terminal escape-sequence sanitization (defaults to `!raw`).
    pub fn with_sanitize_tty(mut self, sanitize: bool) -> Self {
        self.sanitize_tty = sanitize;
        self
    }

    fn flush_dangling_utf8(&mut self, out: &mut impl std::io::Write) {
        if !self.buf.is_empty() {
            let _ = out.write_all("\u{FFFD}".as_bytes());
            self.buf.clear();
        }
    }

    pub fn push(&mut self, tk: &Tokenizer, t: u32, out: &mut impl std::io::Write) {
        if self.raw {
            self.emit(&tk.decode_bytes(t), out);
            return;
        }
        if tk.is_special(t) {
            let s = String::from_utf8_lossy(&tk.decode_bytes(t)).to_string();
            match s.as_str() {
                "<|content_thinking|>" => {
                    self.flush_dangling_utf8(out);
                    self.esc_state = EscState::Normal;
                    self.thinking = true;
                    if self.show_thinking {
                        let _ = write!(out, "\x1b[2m");
                    } else if !self.think_note {
                        let _ = write!(out, "\x1b[2m(thinking…)\x1b[0m ");
                        self.think_note = true;
                    }
                }
                "<|content_text|>" => {
                    self.flush_dangling_utf8(out);
                    self.esc_state = EscState::Normal;
                    if self.thinking && self.show_thinking {
                        let _ = write!(out, "\x1b[0m");
                    }
                    self.thinking = false;
                }
                "<|end_message|>" => {
                    self.flush_dangling_utf8(out);
                    self.esc_state = EscState::Normal;
                    if self.thinking && self.show_thinking {
                        let _ = write!(out, "\x1b[0m\n\n");
                    }
                    self.thinking = false;
                }
                "<|message_model|>" | "<|content_model_end_sampling|>" => {}
                _ => {
                    self.flush_dangling_utf8(out);
                    write_text(&s, self.sanitize_tty && !self.raw, &mut self.esc_state, out);
                }
            }
            let _ = out.flush();
            return;
        }
        if self.thinking && !self.show_thinking {
            return;
        }
        self.emit(&tk.decode_bytes(t), out);
    }

    fn emit(&mut self, bytes: &[u8], out: &mut impl std::io::Write) {
        self.buf.extend_from_slice(bytes);
        let sanitize = self.sanitize_tty && !self.raw;
        let mut wrote = false;
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    if !s.is_empty() {
                        write_text(s, sanitize, &mut self.esc_state, out);
                        wrote = true;
                    }
                    self.buf.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        let s = std::str::from_utf8(&self.buf[..valid]).unwrap();
                        write_text(s, sanitize, &mut self.esc_state, out);
                        wrote = true;
                    }
                    match e.error_len() {
                        Some(err_len) => {
                            write_text("\u{FFFD}", sanitize, &mut self.esc_state, out);
                            wrote = true;
                            self.buf.drain(..valid + err_len);
                        }
                        None => {
                            if valid > 0 {
                                self.buf.drain(..valid);
                            }
                            break;
                        }
                    }
                }
            }
        }
        if wrote {
            let _ = out.flush();
        }
    }

    pub fn end(&mut self, out: &mut impl std::io::Write) {
        self.flush_dangling_utf8(out);
        self.esc_state = EscState::Normal;
        if self.thinking && self.show_thinking {
            let _ = write!(out, "\x1b[0m");
        }
        self.thinking = false;
        self.think_note = false;
        let _ = out.flush();
    }
}
