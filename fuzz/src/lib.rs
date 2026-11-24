//! Shared synthetic tokenizer builder for `fuzz_tokenizer` and `fuzz_renderer`.

use inkling::gguf::Gguf;
use inkling::tokenizer::Tokenizer;
use std::sync::OnceLock;

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn gpt2_byte_char(b: u8) -> char {
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

fn gpt2_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| gpt2_byte_char(b)).collect()
}

/// Standard Inkling control tokens + one unknown control token for fuzzing fallback paths.
pub const SPECIAL_TOKENS: &[&str] = &[
    "<|end_message|>",
    "<|message_system|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_model_end_sampling|>",
    "<|unknown_ctrl|>",
];

/// Build a self-contained synthetic GGUF byte stream containing:
/// - All 256 GPT-2 byte tokens (`0..256`)
/// - All 8 special/control tokens (`256..264`, `tok_type == 3`)
/// - Multi-byte BPE tokens & chained merges (including spaces, repeated letters, ANSI escape bytes,
///   and multi-byte UTF-8 fragments) so both the <=32B stack fast path and >32B O(L log L) heap path
///   are thoroughly exercised.
pub fn build_fuzz_tokenizer_gguf() -> Vec<u8> {
    let mut tokens: Vec<String> = Vec::new();
    let mut types: Vec<i32> = Vec::new();

    for b in 0..=255u8 {
        tokens.push(gpt2_byte_char(b).to_string());
        types.push(1);
    }
    for &ctrl in SPECIAL_TOKENS {
        tokens.push(ctrl.to_string());
        types.push(3);
    }

    let extra_merges: &[(&[u8], &[u8])] = &[
        (b"a", b"a"),
        (b"aa", b"aa"),
        (b"aaaa", b"aaaa"),
        (b"t", b"h"),
        (b"th", b"e"),
        (b" ", b"the"),
        (b"i", b"n"),
        (b"in", b"g"),
        (b"h", b"e"),
        (b"l", b"l"),
        (b"he", b"ll"),
        (b"hell", b"o"),
        (b"<", b"|"),
        (b"|", b">"),
        (b"\x1b", b"["),
        (b"\x1b[", b"31m"),
        (b"\x1b", b"]"),
        (b"\xF0\x9F", b"\x98\x80"),
    ];

    let mut merges: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = tokens.iter().cloned().collect();
    for &(a, b) in extra_merges {
        let sa = gpt2_encode(a);
        let sb = gpt2_encode(b);
        let sab = format!("{sa}{sb}");
        for s in [&sa, &sb, &sab] {
            if seen.insert(s.clone()) {
                tokens.push(s.clone());
                types.push(1);
            }
        }
        merges.push(format!("{sa} {sb}"));
    }

    // Also add raw partial/invalid UTF-8 multi-byte tokens for `fuzz_renderer`.
    for raw_seq in [
        &[0xC3u8, 0xA9][..],       // valid é
        &[0xE2, 0x82][..],         // truncated €
        &[0xFF, 0xFE][..],         // invalid UTF-8
        &[0x1B, b'[', b'2', b'J'], // CSI clear screen
        &[0x1B, b']', b'0', b';', b'X', 0x07], // OSC title
    ] {
        let s = gpt2_encode(raw_seq);
        if seen.insert(s.clone()) {
            tokens.push(s);
            types.push(1);
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
    out.extend_from_slice(&4u64.to_le_bytes()); // n_kv

    // tokenizer.ggml.tokens
    write_str(&mut out, "tokenizer.ggml.tokens");
    out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
    out.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    out.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for t in &tokens {
        write_str(&mut out, t);
    }

    // tokenizer.ggml.token_type
    write_str(&mut out, "tokenizer.ggml.token_type");
    out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
    out.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
    out.extend_from_slice(&(types.len() as u64).to_le_bytes());
    for &ty in &types {
        out.extend_from_slice(&ty.to_le_bytes());
    }

    // tokenizer.ggml.merges
    write_str(&mut out, "tokenizer.ggml.merges");
    out.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
    out.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    out.extend_from_slice(&(merges.len() as u64).to_le_bytes());
    for m in &merges {
        write_str(&mut out, m);
    }

    // tokenizer.ggml.eos_token_id
    write_str(&mut out, "tokenizer.ggml.eos_token_id");
    out.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    out.extend_from_slice(&256u32.to_le_bytes());

    while out.len() % 32 != 0 {
        out.push(0);
    }
    out
}

pub fn fuzz_tokenizer() -> &'static Tokenizer {
    static TK: OnceLock<Tokenizer> = OnceLock::new();
    TK.get_or_init(|| {
        let raw = build_fuzz_tokenizer_gguf();
        let g = Gguf::from_bytes(&raw).expect("synthetic fuzz tokenizer gguf must parse");
        Tokenizer::load(&g)
    })
}
