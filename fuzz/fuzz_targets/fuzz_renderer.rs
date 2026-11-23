#![no_main]

use arbitrary::Arbitrary;
use inkling::session::Renderer;
use inkling_fuzz::{fuzz_tokenizer, SPECIAL_TOKENS};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Op {
    /// Push a single-byte token (`0..=255`).
    Byte(u8),
    /// Push a special/control token (`256..256 + SPECIAL_TOKENS.len()`).
    Special(u8),
    /// Push an arbitrary token ID (including multi-byte tokens and out-of-range IDs).
    Token(u32),
    /// Push a multi-byte sequence via single-byte tokens.
    Bytes(Vec<u8>),
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    show_thinking: bool,
    raw: bool,
    sanitize_tty: bool,
    ops: Vec<Op>,
}

fuzz_target!(|input: FuzzInput| {
    let tk = fuzz_tokenizer();
    let mut r = Renderer::new(input.show_thinking, input.raw).with_sanitize_tty(input.sanitize_tty);
    let mut out = Vec::new();

    for op in input.ops.into_iter().take(512) {
        match op {
            Op::Byte(b) => r.push(tk, b as u32, &mut out),
            Op::Special(idx) => {
                let id = 256u32 + (idx as u32 % SPECIAL_TOKENS.len() as u32);
                r.push(tk, id, &mut out);
            }
            Op::Token(id) => r.push(tk, id, &mut out),
            Op::Bytes(bs) => {
                for b in bs.into_iter().take(64) {
                    r.push(tk, b as u32, &mut out);
                }
            }
        }
    }
    r.end(&mut out);

    // Invariant 1: Renderer output must ALWAYS be valid UTF-8.
    let s = std::str::from_utf8(&out).expect("Renderer emitted invalid UTF-8");

    // Invariant 2: Thinking mode must never remain active after `end()`.
    assert!(!r.thinking, "Renderer::end() left thinking == true");

    // Invariant 3: When TTY sanitization is active, no raw ESC/DEL/C0/C1 controls may leak
    // outside the renderer's own `\x1b[2m` / `\x1b[0m` dim sequences.
    if input.sanitize_tty && !input.raw {
        let stripped = s.replace("\x1b[2m", "").replace("\x1b[0m", "");
        for ch in stripped.chars() {
            assert_ne!(ch, '\x1b', "raw ESC leaked into sanitized output: {s:?}");
            assert_ne!(ch, '\x7f', "DEL leaked into sanitized output: {s:?}");
            if ch.is_control() {
                assert!(
                    matches!(ch, '\n' | '\r' | '\t'),
                    "unsafe control char U+{:04X} leaked into sanitized output: {s:?}",
                    ch as u32
                );
            }
        }
    }
});
