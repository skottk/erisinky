//! Regression and property tests for Medium-Severity #4 (Session & terminal renderer defects)
//! and Medium-Severity #6b (Raw terminal escape stripping in `Renderer`).

mod common;

use common::gguf_builder::minimal_tokenizer_gguf;
use inkling::gguf::Gguf;
use inkling::sampler::Sampler;
use inkling::session::{
    generate_with_backend, validate_prefill, DecodeBackend, DecodeState, Renderer, Session, Stats,
};
use inkling::tokenizer::Tokenizer;
use proptest::prelude::*;
use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

fn gpt2_byte_to_char(b: u8) -> char {
    let mut n = 0u32;
    for x in 0..=b as u32 {
        let keep =
            (33..=126).contains(&x) || (161..=172).contains(&x) || (174..=255).contains(&x);
        let cp = if keep {
            x
        } else {
            n += 1;
            255 + n
        };
        if x == b as u32 {
            return char::from_u32(cp).unwrap();
        }
    }
    unreachable!()
}

fn gpt2_token(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| gpt2_byte_to_char(b)).collect()
}

fn byte_tokenizer() -> &'static Tokenizer {
    static TK: OnceLock<Tokenizer> = OnceLock::new();
    TK.get_or_init(|| {
        let raw = minimal_tokenizer_gguf(&[], &[]);
        let g = Gguf::from_bytes(&raw).expect("parse minimal tokenizer gguf");
        Tokenizer::load(&g)
    })
}

/// Build a tokenizer with `chunks` registered as single multi-byte non-special tokens,
/// returning `(Tokenizer, chunk_token_ids)`.
fn tokenizer_with_byte_chunks(chunks: &[&[u8]]) -> (Tokenizer, Vec<u32>) {
    let encoded: Vec<String> = chunks.iter().map(|c| gpt2_token(c)).collect();
    let extras: Vec<(&str, i32)> = encoded.iter().map(|s| (s.as_str(), 1)).collect();
    let raw = minimal_tokenizer_gguf(&extras, &[]);
    let g = Gguf::from_bytes(&raw).expect("parse tokenizer gguf");
    let tk = Tokenizer::load(&g);
    let base_id = 256u32 + 7u32; // 256 byte tokens + 7 standard special tokens
    let ids = (0..chunks.len()).map(|i| base_id + i as u32).collect();
    (tk, ids)
}

fn render_byte_by_byte(bytes: &[u8], show_thinking: bool, raw: bool) -> Vec<u8> {
    let tk = byte_tokenizer();
    let mut r = Renderer::new(show_thinking, raw);
    let mut out = Vec::new();
    for &b in bytes {
        r.push(tk, b as u32, &mut out);
    }
    r.end(&mut out);
    out
}

fn render_chunks(chunks: &[&[u8]], show_thinking: bool, raw: bool) -> Vec<u8> {
    let (tk, ids) = tokenizer_with_byte_chunks(chunks);
    let mut r = Renderer::new(show_thinking, raw);
    let mut out = Vec::new();
    for id in ids {
        r.push(&tk, id, &mut out);
    }
    r.end(&mut out);
    out
}

fn split_by_cuts<'a>(data: &'a [u8], cuts: &[usize]) -> Vec<&'a [u8]> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut points: Vec<usize> = cuts
        .iter()
        .map(|&c| c % (data.len() + 1))
        .filter(|&c| c > 0 && c < data.len())
        .collect();
    points.sort_unstable();
    points.dedup();
    let mut out = Vec::with_capacity(points.len() + 1);
    let mut prev = 0;
    for p in points {
        out.push(&data[prev..p]);
        prev = p;
    }
    out.push(&data[prev..]);
    out
}

fn assert_no_dangerous_tty_chars(s: &str) {
    for ch in s.chars() {
        assert_ne!(ch, '\x1b', "raw ESC leaked into output: {s:?}");
        assert_ne!(ch, '\x7f', "DEL leaked into output: {s:?}");
        if ch.is_control() {
            assert!(
                matches!(ch, '\n' | '\r' | '\t'),
                "unexpected control char {ch:?} (U+{:04X}) leaked into output: {s:?}",
                ch as u32
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Renderer UTF-8 streaming & Utf8Error::error_len() regression tests
// ---------------------------------------------------------------------------

#[test]
fn renderer_utf8_invalid_byte_followed_by_ascii_preserves_all_valid_bytes() {
    let input = [0xFF, b'H', b'e', b'l', b'l', b'o'];

    let single = render_chunks(&[&input], true, false);
    assert_eq!(
        std::str::from_utf8(&single).expect("valid utf8"),
        "\u{FFFD}Hello",
        "single-token invalid byte must emit U+FFFD without swallowing following ASCII"
    );

    let streamed = render_byte_by_byte(&input, true, false);
    assert_eq!(
        std::str::from_utf8(&streamed).expect("valid utf8"),
        "\u{FFFD}Hello",
        "byte-by-byte invalid byte must emit U+FFFD without swallowing following ASCII"
    );
}

#[test]
fn renderer_utf8_invalid_multibyte_followed_by_ok_preserves_valid_bytes() {
    let input = [0xC0, 0xAF, b'o', b'k'];

    let single = render_chunks(&[&input], true, false);
    assert_eq!(
        std::str::from_utf8(&single).expect("valid utf8"),
        "\u{FFFD}\u{FFFD}ok"
    );

    let streamed = render_byte_by_byte(&input, true, false);
    assert_eq!(
        std::str::from_utf8(&streamed).expect("valid utf8"),
        "\u{FFFD}\u{FFFD}ok"
    );
}

#[test]
fn renderer_utf8_truncated_sequence_before_ascii_and_at_eof() {
    // Truncated 3-byte UTF-8 sequence followed by 'b'
    let input = [b'a', 0xE2, 0x82, b'b'];
    let streamed = render_byte_by_byte(&input, true, false);
    assert_eq!(
        std::str::from_utf8(&streamed).expect("valid utf8"),
        "a\u{FFFD}b"
    );

    // Incomplete 4-byte emoji prefix at EOF must be flushed as U+FFFD by end()
    let dangling = [b'x', 0xF0, 0x9F, 0x98];
    let streamed_eof = render_byte_by_byte(&dangling, true, false);
    assert_eq!(
        std::str::from_utf8(&streamed_eof).expect("valid utf8"),
        "x\u{FFFD}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: None,
        .. ProptestConfig::default()
    })]

    #[test]
    fn prop_renderer_valid_utf8_raw_arbitrary_chunking_roundtrips(
        s in "[\\PC\n\r\t🦀日本語é̀]{0,128}",
        cuts in prop::collection::vec(0usize..256, 0..32),
    ) {
        let bytes = s.as_bytes();
        // Raw mode preserves all valid UTF-8 regardless of chunk boundaries
        let chunks = split_by_cuts(bytes, &cuts);
        let out_chunks = render_chunks(&chunks, true, true);
        prop_assert_eq!(std::str::from_utf8(&out_chunks).unwrap(), s.as_str());

        // 1-byte steps also roundtrip identically
        let out_bytes = render_byte_by_byte(bytes, true, true);
        prop_assert_eq!(std::str::from_utf8(&out_bytes).unwrap(), s.as_str());
    }

    #[test]
    fn prop_renderer_valid_printable_utf8_sanitized_arbitrary_chunking_roundtrips(
        s in "[a-zA-Z0-9 .,!?\n\r\t🦀🎉日本語한글é\u{0301}€]{0,96}",
        cuts in prop::collection::vec(0usize..256, 0..32),
    ) {
        let bytes = s.as_bytes();
        let chunks = split_by_cuts(bytes, &cuts);
        let out_chunks = render_chunks(&chunks, true, false);
        prop_assert_eq!(std::str::from_utf8(&out_chunks).unwrap(), s.as_str());

        let out_bytes = render_byte_by_byte(bytes, true, false);
        prop_assert_eq!(std::str::from_utf8(&out_bytes).unwrap(), s.as_str());
    }

    #[test]
    fn prop_renderer_arbitrary_bytes_raw_matches_utf8_lossy(
        bytes in prop::collection::vec(any::<u8>(), 0..128),
        cuts in prop::collection::vec(0usize..256, 0..32),
    ) {
        let expected = String::from_utf8_lossy(&bytes);
        let chunks = split_by_cuts(&bytes, &cuts);
        let out_chunks = render_chunks(&chunks, true, true);
        let s_chunks = std::str::from_utf8(&out_chunks).expect("output must always be valid UTF-8");
        prop_assert_eq!(s_chunks, expected.as_ref());

        let out_bytes = render_byte_by_byte(&bytes, true, true);
        let s_bytes = std::str::from_utf8(&out_bytes).expect("output must always be valid UTF-8");
        prop_assert_eq!(s_bytes, expected.as_ref());
    }

    #[test]
    fn prop_renderer_arbitrary_bytes_sanitized_is_valid_utf8_and_safe(
        bytes in prop::collection::vec(any::<u8>(), 0..128),
        cuts in prop::collection::vec(0usize..256, 0..32),
    ) {
        let chunks = split_by_cuts(&bytes, &cuts);
        let out_chunks = render_chunks(&chunks, true, false);
        let s_chunks = std::str::from_utf8(&out_chunks).expect("sanitized output must be valid UTF-8");
        assert_no_dangerous_tty_chars(s_chunks);

        let out_bytes = render_byte_by_byte(&bytes, true, false);
        let s_bytes = std::str::from_utf8(&out_bytes).expect("sanitized output must be valid UTF-8");
        assert_no_dangerous_tty_chars(s_bytes);
    }
}

// ---------------------------------------------------------------------------
// 2. Renderer thinking-mode ANSI dim reset tests
// ---------------------------------------------------------------------------

#[test]
fn renderer_thinking_mode_resets_dim_on_content_text_without_end_message() {
    let (tk, ids) = tokenizer_with_byte_chunks(&[b"reasoning", b"answer"]);
    let t_think = tk.special_id("<|content_thinking|>").unwrap();
    let t_text = tk.special_id("<|content_text|>").unwrap();

    let mut r = Renderer::new(true, false);
    let mut out = Vec::new();
    r.push(&tk, t_think, &mut out);
    r.push(&tk, ids[0], &mut out);
    r.push(&tk, t_text, &mut out);
    let after_text_switch = String::from_utf8(out.clone()).expect("valid utf8");
    assert_eq!(
        after_text_switch, "\x1b[2mreasoning\x1b[0m",
        "<|content_text|> must emit \\x1b[0m to reset dim when exiting thinking mode"
    );

    r.push(&tk, ids[1], &mut out);
    r.end(&mut out);
    let final_out = String::from_utf8(out).expect("valid utf8");
    assert_eq!(
        final_out, "\x1b[2mreasoning\x1b[0manswer",
        "end() must not double-emit \\x1b[0m after <|content_text|> already reset thinking"
    );
}

#[test]
fn renderer_thinking_mode_resets_dim_on_end_and_does_not_double_reset() {
    let (tk, ids) = tokenizer_with_byte_chunks(&[b"reasoning"]);
    let t_think = tk.special_id("<|content_thinking|>").unwrap();

    let mut r = Renderer::new(true, false);
    let mut out = Vec::new();
    r.push(&tk, t_think, &mut out);
    r.push(&tk, ids[0], &mut out);
    r.end(&mut out);
    r.end(&mut out);

    let final_out = String::from_utf8(out).expect("valid utf8");
    assert_eq!(final_out, "\x1b[2mreasoning\x1b[0m");
}

#[test]
fn renderer_hidden_thinking_emits_note_and_clean_answer() {
    let (tk, ids) = tokenizer_with_byte_chunks(&[b"secret", b"visible"]);
    let t_think = tk.special_id("<|content_thinking|>").unwrap();
    let t_text = tk.special_id("<|content_text|>").unwrap();

    let mut r = Renderer::new(false, false);
    let mut out = Vec::new();
    r.push(&tk, t_think, &mut out);
    r.push(&tk, ids[0], &mut out);
    r.push(&tk, t_text, &mut out);
    r.push(&tk, ids[1], &mut out);
    r.end(&mut out);

    assert_eq!(
        String::from_utf8(out).unwrap(),
        "\x1b[2m(thinking…)\x1b[0m visible"
    );
}

// ---------------------------------------------------------------------------
// 3. Renderer ANSI / OSC / control-char escape sanitization tests
// ---------------------------------------------------------------------------

#[test]
fn renderer_strips_csi_osc_standalone_esc_and_c0_single_and_split() {
    let cases: &[(&[u8], &str)] = &[
        (b"\x1b[31mred\x1b[0m", "red"),
        (b"\x1b[2Jhello", "hello"),
        (b"\x1b[?25lhi\x1b[?25h", "hi"),
        (b"\x1b]0;pwned title\x07safe", "safe"),
        (b"\x1b]52;c;cHduZWQ=\x1b\\after", "after"),
        (b"hello\x1b", "hello"),
        (b"hello\x1b\nworld", "hello\nworld"),
        (b"a\x07b\x08c\x00d\x7fe", "abcde"),
        ("\u{009B}31mred\u{009B}0m".as_bytes(), "red"),
        (b"line1\r\n\tline2\n", "line1\r\n\tline2\n"),
    ];

    for &(input, expected) in cases {
        // Single token containing entire payload
        let out_single = render_chunks(&[input], true, false);
        assert_eq!(
            std::str::from_utf8(&out_single).unwrap(),
            expected,
            "single-token mismatch for {:?}",
            String::from_utf8_lossy(input)
        );

        // 1-byte streaming (splits every escape sequence across token boundaries)
        let out_bytes = render_byte_by_byte(input, true, false);
        assert_eq!(
            std::str::from_utf8(&out_bytes).unwrap(),
            expected,
            "1-byte split mismatch for {:?}",
            String::from_utf8_lossy(input)
        );

        // Every 2-way split boundary
        for split in 1..input.len() {
            let out_two = render_chunks(&[&input[..split], &input[split..]], true, false);
            assert_eq!(
                std::str::from_utf8(&out_two).unwrap(),
                expected,
                "2-way split at {split} mismatch for {:?}",
                String::from_utf8_lossy(input)
            );
        }
    }
}

#[test]
fn renderer_preserves_own_thinking_codes_while_stripping_model_escapes() {
    let (tk, ids) = tokenizer_with_byte_chunks(&[
        b"\x1b[31mthink\x1b]0;bad\x07",
        b"\x1b[2Janswer\x1b[0m",
    ]);
    let t_think = tk.special_id("<|content_thinking|>").unwrap();
    let t_text = tk.special_id("<|content_text|>").unwrap();

    let mut r = Renderer::new(true, false);
    let mut out = Vec::new();
    r.push(&tk, t_think, &mut out);
    r.push(&tk, ids[0], &mut out);
    r.push(&tk, t_text, &mut out);
    r.push(&tk, ids[1], &mut out);
    r.end(&mut out);

    assert_eq!(
        String::from_utf8(out).unwrap(),
        "\x1b[2mthink\x1b[0manswer"
    );
}

#[test]
fn renderer_unclosed_escape_in_thinking_does_not_swallow_answer_text() {
    let (tk, ids) = tokenizer_with_byte_chunks(&[b"think\x1b[31", b"answer"]);
    let t_think = tk.special_id("<|content_thinking|>").unwrap();
    let t_text = tk.special_id("<|content_text|>").unwrap();

    let mut r = Renderer::new(true, false);
    let mut out = Vec::new();
    r.push(&tk, t_think, &mut out);
    r.push(&tk, ids[0], &mut out);
    r.push(&tk, t_text, &mut out);
    r.push(&tk, ids[1], &mut out);
    r.end(&mut out);

    assert_eq!(
        String::from_utf8(out).unwrap(),
        "\x1b[2mthink\x1b[0manswer"
    );
}

#[test]
fn renderer_raw_mode_passes_escapes_through_unchanged() {
    let input = b"\x1b[31mred\x1b[0m\x1b]0;title\x07";
    let out = render_chunks(&[input], true, true);
    assert_eq!(out, input);
}

// ---------------------------------------------------------------------------
// 4. Session::generate (`max == 0` and decode loop control-flow) tests
// ---------------------------------------------------------------------------

struct MockBackend {
    max_ctx: usize,
    capacity: Cell<usize>,
    released_batch: Cell<usize>,
    steps_launched: Cell<usize>,
    logits_sampled: Cell<usize>,
    tokens: RefCell<Vec<u32>>,
    next_toks: Vec<u32>,
}

impl MockBackend {
    fn new(max_ctx: usize, next_toks: Vec<u32>) -> Self {
        Self {
            max_ctx,
            capacity: Cell::new(max_ctx),
            released_batch: Cell::new(0),
            steps_launched: Cell::new(0),
            logits_sampled: Cell::new(0),
            tokens: RefCell::new(vec![0u32; max_ctx + 8]),
            next_toks,
        }
    }
}

impl DecodeBackend for MockBackend {
    type Step = usize;

    fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn release_batch(&self) {
        self.released_batch.set(self.released_batch.get() + 1);
    }

    fn sample_logits(&self, _sampler: &mut Sampler) -> u32 {
        let idx = self.logits_sampled.get();
        self.logits_sampled.set(idx + 1);
        *self.next_toks.get(idx).unwrap_or(&42)
    }

    fn set_token(&self, pos: usize, tok: u32) {
        self.tokens.borrow_mut()[pos] = tok;
    }

    fn gen_reserve(&self, pos: usize) {
        if pos >= self.capacity.get() {
            self.capacity.set(pos + 64);
        }
    }

    fn gen_capacity(&self) -> usize {
        self.capacity.get()
    }

    fn launch_step(&self, pos: usize, _argmax: bool) -> usize {
        let idx = self.steps_launched.get();
        self.steps_launched.set(idx + 1);
        let next = *self.next_toks.get(idx).unwrap_or(&99);
        self.tokens.borrow_mut()[pos + 1] = next;
        pos
    }

    fn wait_step(&self, _step: &usize) {}

    fn token_at(&self, pos: usize) -> u32 {
        self.tokens.borrow()[pos]
    }
}

#[test]
fn session_generate_max_zero_emits_no_tokens_greedy_or_sampled() {
    for (temp, top_k) in [(0.0f32, 1usize), (0.8f32, 40usize)] {
        let eng = MockBackend::new(64, vec![10, 11, 12]);
        let mut sampler = Sampler::new(temp, top_k, 0.95, 0.0, 123);
        let mut n_past = 4usize;
        let mut pending: Option<u32> = None;
        let mut stats = Stats::default();
        let mut emitted = Vec::new();

        let count = {
            let mut state = DecodeState {
                eos: 256,
                n_past: &mut n_past,
                pending: &mut pending,
                stats: &mut stats,
            };
            generate_with_backend(&eng, &mut state, 7, 0, &mut sampler, |t| {
                emitted.push(t);
                true
            })
        };

        assert_eq!(count, 0, "max == 0 must return 0 (temp={temp})");
        assert!(emitted.is_empty(), "on_token must never be called when max == 0");
        assert_eq!(stats.gen_tokens, 0, "gen_tokens must stay 0 when max == 0");
        assert_eq!(pending, None, "pending must stay None when max == 0");
        assert_eq!(n_past, 4, "n_past must not advance when max == 0");
        assert_eq!(eng.steps_launched.get(), 0, "no decode steps may launch when max == 0");
        assert_eq!(eng.logits_sampled.get(), 0, "logits must not be sampled when max == 0");
        assert_eq!(eng.released_batch.get(), 1, "prefill batch arena should still be released");
    }
}

#[test]
fn session_generate_max_one_and_multi_step_greedy_and_sampled() {
    // Greedy max == 1
    {
        let eng = MockBackend::new(64, vec![10, 11, 12]);
        let mut sampler = Sampler::new(0.0, 1, 1.0, 0.0, 1);
        let mut n_past = 4usize;
        let mut pending = None;
        let mut stats = Stats::default();
        let mut emitted = Vec::new();
        let count = {
            let mut state = DecodeState {
                eos: 256,
                n_past: &mut n_past,
                pending: &mut pending,
                stats: &mut stats,
            };
            generate_with_backend(&eng, &mut state, 7, 1, &mut sampler, |t| {
                emitted.push(t);
                true
            })
        };
        assert_eq!(count, 1);
        assert_eq!(emitted, vec![7]);
        assert_eq!(pending, Some(7));
        assert_eq!(stats.gen_tokens, 1);
        assert_eq!(eng.steps_launched.get(), 0);
    }

    // Greedy max == 3 (pipelined 2-in-flight)
    {
        let eng = MockBackend::new(64, vec![10, 11, 12]);
        let mut sampler = Sampler::new(0.0, 1, 1.0, 0.0, 1);
        let mut n_past = 4usize;
        let mut pending = None;
        let mut stats = Stats::default();
        let mut emitted = Vec::new();
        let count = {
            let mut state = DecodeState {
                eos: 256,
                n_past: &mut n_past,
                pending: &mut pending,
                stats: &mut stats,
            };
            generate_with_backend(&eng, &mut state, 7, 3, &mut sampler, |t| {
                emitted.push(t);
                true
            })
        };
        assert_eq!(count, 3);
        assert_eq!(emitted, vec![7, 10, 11]);
        assert_eq!(stats.gen_tokens, 3);
        assert_eq!(n_past, 7);
        assert_eq!(pending, None);
    }

    // Sampled max == 3
    {
        let eng = MockBackend::new(64, vec![20, 21, 22, 23]);
        let mut sampler = Sampler::new(0.7, 40, 0.95, 0.0, 1);
        let mut n_past = 4usize;
        let mut pending = None;
        let mut stats = Stats::default();
        let mut emitted = Vec::new();
        let count = {
            let mut state = DecodeState {
                eos: 256,
                n_past: &mut n_past,
                pending: &mut pending,
                stats: &mut stats,
            };
            generate_with_backend(&eng, &mut state, 7, 3, &mut sampler, |t| {
                emitted.push(t);
                true
            })
        };
        assert_eq!(count, 3);
        assert_eq!(emitted, vec![20, 21, 22]);
        assert_eq!(stats.gen_tokens, 3);
        assert_eq!(pending, Some(22));
    }
}

#[test]
fn renderer_with_sanitize_tty_false_passes_escapes_through_in_non_raw_mode() {
    let input = b"\x1b[31mred\x1b[0m\x1b]0;title\x07\x08";
    let (tk, ids) = tokenizer_with_byte_chunks(&[input]);
    let mut r = Renderer::new(true, false).with_sanitize_tty(false);
    let mut out = Vec::new();
    r.push(&tk, ids[0], &mut out);
    r.end(&mut out);
    assert_eq!(out, input);
}

#[test]
fn renderer_strips_dcs_apc_charset_ris_and_aborted_osc() {
    let cases: &[(&[u8], &str)] = &[
        (b"\x1bP+q544e\x1b\\ok", "ok"),
        (b"\x1b_Ga=T,f=32;AAAA\x1b\\ok", "ok"),
        (b"\x1b(0abc\x1b(B", "abc"),
        (b"\x1bchello", "hello"),
        (b"\x1b]0;foo\x1b[31mhi", "hi"),
    ];
    for &(input, expected) in cases {
        let out_single = render_chunks(&[input], true, false);
        assert_eq!(std::str::from_utf8(&out_single).unwrap(), expected);

        let out_bytes = render_byte_by_byte(input, true, false);
        assert_eq!(std::str::from_utf8(&out_bytes).unwrap(), expected);
    }
}

#[test]
fn session_validate_prefill_rejects_empty_overflow_and_full_context() {
    // Empty prompt with no pending token -> Err
    assert!(validate_prefill(0, false, 0, 64).is_err());
    assert!(validate_prefill(10, false, 0, 64).is_err());

    // Pending token alone counts as 1 token -> Ok
    assert_eq!(validate_prefill(10, true, 0, 64), Ok(11));

    // Exact boundary: end == max_ctx - 1 is Ok; end == max_ctx is Err
    assert_eq!(validate_prefill(0, false, 63, 64), Ok(63));
    assert!(validate_prefill(0, false, 64, 64).is_err());
    assert!(validate_prefill(60, false, 4, 64).is_err());
    assert!(validate_prefill(60, true, 3, 64).is_err());
    assert_eq!(validate_prefill(60, true, 2, 64), Ok(63));

    // Arithmetic overflow -> Err (never panics or wraps)
    assert!(validate_prefill(usize::MAX, false, 1, usize::MAX).is_err());
    assert!(validate_prefill(0, true, usize::MAX, usize::MAX).is_err());
}

/// Compile-time verification that `Session::try_prefill` and `Session::prefill` keep exact signatures.
#[allow(dead_code)]
fn _assert_session_prefill_signatures(s: &mut Session<'_>, toks: &[u32]) {
    let _: Result<u32, String> = s.try_prefill(toks);
    let _: u32 = s.prefill(toks);
}

#[test]
fn session_generate_stops_on_eos_and_callback_false() {
    // EOS on first token: on_token is not called, returns 1, pending = Some(eos)
    {
        let eng = MockBackend::new(64, vec![10, 11]);
        let mut sampler = Sampler::new(0.0, 1, 1.0, 0.0, 1);
        let mut n_past = 4usize;
        let mut pending = None;
        let mut stats = Stats::default();
        let mut emitted = Vec::new();
        let count = {
            let mut state = DecodeState {
                eos: 256,
                n_past: &mut n_past,
                pending: &mut pending,
                stats: &mut stats,
            };
            generate_with_backend(&eng, &mut state, 256, 10, &mut sampler, |t| {
                emitted.push(t);
                true
            })
        };
        assert_eq!(count, 1);
        assert!(emitted.is_empty());
        assert_eq!(pending, Some(256));
    }

    // on_token returns false on second token
    {
        let eng = MockBackend::new(64, vec![10, 11, 12]);
        let mut sampler = Sampler::new(0.0, 1, 1.0, 0.0, 1);
        let mut n_past = 4usize;
        let mut pending = None;
        let mut stats = Stats::default();
        let mut emitted = Vec::new();
        let count = {
            let mut state = DecodeState {
                eos: 256,
                n_past: &mut n_past,
                pending: &mut pending,
                stats: &mut stats,
            };
            generate_with_backend(&eng, &mut state, 7, 10, &mut sampler, |t| {
                emitted.push(t);
                emitted.len() < 2
            })
        };
        assert_eq!(count, 2);
        assert_eq!(emitted, vec![7, 10]);
        assert_eq!(stats.gen_tokens, 2);
    }
}
