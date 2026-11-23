//! CLI argument parsing and prompt formatting, decoupled from execution for testing.

use crate::tokenizer::Tokenizer;
use std::fmt;

pub const DEFAULT_MODEL: &str =
    "~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf";

pub const USAGE: &str = "inkling - a hand-fused Metal inference engine for Thinking Machines Inkling 2 Mini

usage: inkling <command> [options]

commands:
  chat              interactive multi-turn chat (KV cache reused across turns)
  run -p TEXT       one prompt, streamed
  bench             greedy prefill + decode benchmark
  check --ref P     compare logits against a llama.cpp reference dump (tools/ref_logits)
  verify            audit GGUF weights/scales and verify Fast vs Safe Metal math

options:
  -m, --model PATH  GGUF file (default: LM Studio's Inkling-2-Mini Q4_K_M)
  -p, --prompt TEXT prompt text (run/bench)      -f FILE  read the prompt from a file
  -n N              max new tokens (default 2048, bench 128)
  -c, --ctx N       context length (default 16384)
  --effort E        reasoning effort: none|minimal|low|medium|high|xhigh|<0..0.99> (default high)
  --system TEXT     system prompt
  --temp T          temperature (default 1.0; 0 = greedy, fully GPU-pipelined)
  --top-p P, --top-k K, --seed S
  --greedy          same as --temp 0
  --hide-thinking   do not print the reasoning stream
  --kv-disk         keep only half of the (exact) K/V cache in memory while generating; the older half lives in a
                    temporary file and is read back for every token (slower generation)
  --kv8             8-bit K/V cache: half the cache memory, NOT exact (the default fp16 cache is)
  --raw             do not apply the chat template; print special tokens";

/// Minimum supported context length (matches `Engine::try_load_with` 64-token alignment).
pub const MIN_CTX: usize = 64;
/// Maximum context length whose full-attention KV cache (8 full slots × 2 × NKV × HD elements)
/// fits in 32-bit Metal buffer offsets (`< u32::MAX`).
pub const MAX_CTX: usize =
    ((u32::MAX as usize) / (8 * 2 * crate::engine::NKV * crate::engine::HD)) / 64 * 64;

#[derive(Debug, Clone)]
pub struct Opts {
    pub model: String,
    pub prompt: String,
    pub prompt_file: Option<String>,
    pub n: usize,
    pub ctx: usize,
    pub raw: bool,
    pub effort: String,
    pub reference: String,
    pub quiet: bool,
    pub stop: usize,
    pub system: String,
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub seed_set: bool,
    pub hide_thinking: bool,
    pub n_set: bool,
    pub kv8: bool,
    pub kv_disk: bool,
    pub json: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    MissingValue(String),
    InvalidValue { flag: String, value: String, reason: String },
    UnknownOption(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::MissingValue(flag) => write!(f, "missing value for {flag}"),
            CliError::InvalidValue { flag, value, reason } => {
                write!(f, "invalid value '{value}' for {flag}: {reason}")
            }
            CliError::UnknownOption(opt) => write!(f, "unknown option {opt}\n\n{USAGE}"),
        }
    }
}

impl std::error::Error for CliError {}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ParseOutcome {
    Help,
    Command(String, Opts),
}

fn parse_effort(e: &str) -> Option<f64> {
    match e.trim() {
        "none" => Some(0.0),
        "minimal" => Some(0.1),
        "low" => Some(0.2),
        "medium" => Some(0.7),
        "high" => Some(0.9),
        "xhigh" | "max" => Some(0.99),
        x => match x.parse::<f64>() {
            Ok(v) if v.is_finite() && (0.0..=0.99).contains(&v) => Some(v),
            _ => None,
        },
    }
}

/// Pure argument parser: parses an iterator of CLI arguments (without argv[0]).
pub fn parse_args<I, S>(iter: I, home: &str) -> Result<ParseOutcome, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = iter.into_iter().map(Into::into);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    if cmd == "-h" || cmd == "--help" {
        return Ok(ParseOutcome::Help);
    }
    let mut o = Opts {
        model: DEFAULT_MODEL.replace('~', home),
        prompt: "Write a haiku about the ocean.".into(),
        prompt_file: None,
        n: 2048,
        ctx: 16384,
        raw: false,
        effort: "high".into(),
        reference: String::new(),
        quiet: false,
        stop: 1,
        system: String::new(),
        temp: 1.0,
        top_p: 0.95,
        top_k: 64,
        seed: 0,
        seed_set: false,
        hide_thinking: false,
        n_set: false,
        kv8: false,
        kv_disk: false,
        json: false,
    };
    while let Some(a) = args.next() {
        let mut next_val = || -> Result<String, CliError> {
            args.next().ok_or_else(|| CliError::MissingValue(a.clone()))
        };
        match a.as_str() {
            "-m" | "--model" => o.model = next_val()?,
            "-p" | "--prompt" => {
                o.prompt = next_val()?;
                o.prompt_file = None;
            }
            "-f" | "--file" => o.prompt_file = Some(next_val()?),
            "-n" => {
                let raw = next_val()?;
                o.n = raw.parse().map_err(|e: std::num::ParseIntError| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw,
                    reason: e.to_string(),
                })?;
                o.n_set = true;
            }
            "-c" | "--ctx" => {
                let raw = next_val()?;
                let parsed: usize = raw.parse().map_err(|e: std::num::ParseIntError| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw.clone(),
                    reason: e.to_string(),
                })?;
                if parsed < MIN_CTX {
                    return Err(CliError::InvalidValue {
                        flag: a.clone(),
                        value: raw,
                        reason: format!("must be at least {MIN_CTX}"),
                    });
                }
                let rounded = parsed.checked_next_multiple_of(64).ok_or_else(|| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw.clone(),
                    reason: format!("must be at most {MAX_CTX}"),
                })?;
                if rounded > MAX_CTX {
                    return Err(CliError::InvalidValue {
                        flag: a.clone(),
                        value: raw,
                        reason: format!("must be at most {MAX_CTX} (32-bit KV offset limit)"),
                    });
                }
                o.ctx = rounded;
            }
            "--raw" => o.raw = true,
            "--kv8" => o.kv8 = true,
            "--kv-disk" => o.kv_disk = true,
            "--json" => o.json = true,
            "--effort" => {
                let raw = next_val()?;
                if parse_effort(&raw).is_none() {
                    return Err(CliError::InvalidValue {
                        flag: a.clone(),
                        value: raw,
                        reason: "expected none|minimal|low|medium|high|xhigh|max or finite number in 0.0..=0.99".into(),
                    });
                }
                o.effort = raw;
            }
            "--system" => o.system = next_val()?,
            "--temp" => {
                let raw = next_val()?;
                let parsed: f32 = raw.parse().map_err(|e: std::num::ParseFloatError| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw.clone(),
                    reason: e.to_string(),
                })?;
                if !parsed.is_finite() || parsed < 0.0 {
                    return Err(CliError::InvalidValue {
                        flag: a.clone(),
                        value: raw,
                        reason: "must be a finite number >= 0.0".into(),
                    });
                }
                o.temp = if parsed == 0.0 { 0.0 } else { parsed };
            }
            "--top-p" => {
                let raw = next_val()?;
                let parsed: f32 = raw.parse().map_err(|e: std::num::ParseFloatError| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw.clone(),
                    reason: e.to_string(),
                })?;
                if !parsed.is_finite() || parsed <= 0.0 || parsed > 1.0 {
                    return Err(CliError::InvalidValue {
                        flag: a.clone(),
                        value: raw,
                        reason: "must be a finite number in (0.0, 1.0]".into(),
                    });
                }
                o.top_p = parsed;
            }
            "--top-k" => {
                let raw = next_val()?;
                o.top_k = raw.parse().map_err(|e: std::num::ParseIntError| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw,
                    reason: e.to_string(),
                })?;
            }
            "--seed" => {
                let raw = next_val()?;
                o.seed = raw.parse().map_err(|e: std::num::ParseIntError| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw,
                    reason: e.to_string(),
                })?;
                o.seed_set = true;
            }
            "--greedy" => o.temp = 0.0,
            "--hide-thinking" => o.hide_thinking = true,
            "--ref" => o.reference = next_val()?,
            "-q" => o.quiet = true,
            "--stop" => {
                let raw = next_val()?;
                o.stop = raw.parse().map_err(|e: std::num::ParseIntError| CliError::InvalidValue {
                    flag: a.clone(),
                    value: raw,
                    reason: e.to_string(),
                })?;
            }
            "-h" | "--help" => return Ok(ParseOutcome::Help),
            _ => return Err(CliError::UnknownOption(a)),
        }
    }
    Ok(ParseOutcome::Command(cmd, o))
}

/// reasoning effort as rendered by the GGUF chat template
pub fn effort_str(e: &str) -> String {
    let v = match e.trim() {
        "none" => 0.0,
        "minimal" => 0.1,
        "low" => 0.2,
        "medium" => 0.7,
        "high" => 0.9,
        "xhigh" | "max" => 0.99,
        x => match x.parse::<f64>() {
            Ok(n) if n.is_finite() => n.clamp(0.0, 0.99),
            _ => 0.9,
        },
    };
    if v == 0.0 {
        "0".into()
    } else {
        format!("{v}")
    }
}

/// Format the first turn as a raw chat-template string.
///
/// NOTE: Do NOT tokenize the result with `parse_special = true` for untrusted `user` or
/// `o.system` input, as literal control tokens in the payload would be smuggled as real
/// control token IDs. Prefer [`encode_first_turn`].
pub fn first_turn(o: &Opts, user: &str) -> String {
    let mut s = String::new();
    if !o.system.is_empty() {
        s += &format!("<|message_system|><|content_text|>{}<|end_message|>", o.system);
    }
    s += &format!(
        "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>",
        effort_str(&o.effort)
    );
    s += &format!("<|message_user|><|content_text|>{user}<|end_message|>");
    s
}

fn push_special(tk: &Tokenizer, out: &mut Vec<u32>, tok: &str) {
    let id = tk
        .special_id(tok)
        .unwrap_or_else(|| panic!("tokenizer missing required control token {tok}"));
    out.push(id);
}

/// Structurally encode a user turn: control token IDs are pushed via `Tokenizer::special_id`
/// and `user` is encoded with `parse_special = false` so user text can never smuggle control tokens.
pub fn encode_user_turn(tk: &Tokenizer, user: &str) -> Vec<u32> {
    let mut out = Vec::new();
    push_special(tk, &mut out, "<|message_user|>");
    push_special(tk, &mut out, "<|content_text|>");
    out.extend_from_slice(&tk.encode(user, false));
    push_special(tk, &mut out, "<|end_message|>");
    out
}

/// Structurally encode the first turn (optional `--system` message, reasoning effort system
/// message, and user turn) so neither `--system` nor `user` can smuggle control tokens.
pub fn encode_first_turn(tk: &Tokenizer, o: &Opts, user: &str) -> Vec<u32> {
    let mut out = Vec::new();
    if !o.system.is_empty() {
        push_special(tk, &mut out, "<|message_system|>");
        push_special(tk, &mut out, "<|content_text|>");
        out.extend_from_slice(&tk.encode(&o.system, false));
        push_special(tk, &mut out, "<|end_message|>");
    }
    push_special(tk, &mut out, "<|message_system|>");
    push_special(tk, &mut out, "<|content_text|>");
    let effort_msg = format!("Thinking effort level: {}", effort_str(&o.effort));
    out.extend_from_slice(&tk.encode(&effort_msg, false));
    push_special(tk, &mut out, "<|end_message|>");
    out.extend_from_slice(&encode_user_turn(tk, user));
    out
}

/// Differential comparison between `MTLMathMode::Fast` and `MTLMathMode::Safe` logits
/// over a single step (after optional logit softcapping).
#[derive(Debug, Clone, Copy)]
pub struct LogitDiff {
    pub non_finite_fast: usize,
    pub non_finite_safe: usize,
    pub max_abs_diff: f32,
    pub kl_div: f32,
    pub argmax_fast: usize,
    pub argmax_safe: usize,
}

impl LogitDiff {
    pub fn argmax_match(&self) -> bool {
        self.argmax_fast == self.argmax_safe
    }
}

fn apply_softcap(x: f32, c: f32) -> f32 {
    if c > 0.0 && c.is_finite() {
        c * (x / c).tanh()
    } else {
        x
    }
}

/// Compare `fast` vs `safe` logit slices, counting non-finite values, max absolute
/// difference, argmax agreement, and `KL(safe || fast)` over finite entries.
pub fn diff_logits(fast: &[f32], safe: &[f32], softcap: f32) -> LogitDiff {
    let n = fast.len().min(safe.len());
    let mut non_finite_fast = 0usize;
    let mut non_finite_safe = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut argmax_fast = 0usize;
    let mut best_fast = f32::NEG_INFINITY;
    let mut argmax_safe = 0usize;
    let mut best_safe = f32::NEG_INFINITY;

    let mut f_sc = Vec::with_capacity(n);
    let mut s_sc = Vec::with_capacity(n);
    for i in 0..n {
        let fv = fast[i];
        let sv = safe[i];
        if !fv.is_finite() {
            non_finite_fast += 1;
        }
        if !sv.is_finite() {
            non_finite_safe += 1;
        }
        let fc = apply_softcap(fv, softcap);
        let sc = apply_softcap(sv, softcap);
        if fc.is_finite() && fc > best_fast {
            best_fast = fc;
            argmax_fast = i;
        }
        if sc.is_finite() && sc > best_safe {
            best_safe = sc;
            argmax_safe = i;
        }
        if fc.is_finite() && sc.is_finite() {
            max_abs_diff = max_abs_diff.max((fc - sc).abs());
        }
        f_sc.push(fc);
        s_sc.push(sc);
    }

    let kl_div = if non_finite_fast == 0 && non_finite_safe == 0 && n > 0 {
        let lse = |v: &[f32]| -> f64 {
            let m = v.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b as f64));
            let s: f64 = v.iter().map(|&x| ((x as f64) - m).exp()).sum();
            m + s.ln()
        };
        let lf = lse(&f_sc);
        let ls = lse(&s_sc);
        let kl: f64 = s_sc
            .iter()
            .zip(&f_sc)
            .map(|(&sv, &fv)| {
                let ps = ((sv as f64) - ls).exp();
                ps * (((sv as f64) - ls) - ((fv as f64) - lf))
            })
            .sum();
        kl.max(0.0) as f32
    } else {
        f32::NAN
    };

    LogitDiff {
        non_finite_fast,
        non_finite_safe,
        max_abs_diff,
        kl_div,
        argmax_fast,
        argmax_safe,
    }
}
