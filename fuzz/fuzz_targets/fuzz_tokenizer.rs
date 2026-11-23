#![no_main]

use arbitrary::Arbitrary;
use inkling_fuzz::fuzz_tokenizer;
use libfuzzer_sys::fuzz_target;
use std::time::{Duration, Instant};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    text: String,
    parse_special: bool,
    decode_ids: Vec<u32>,
}

/// Cap per-iteration text length so libFuzzer stays responsive while still exercising
/// both the <=32B stack fast path and >32B O(L log L) heap BPE path.
const MAX_TEXT_BYTES: usize = 8192;

fuzz_target!(|input: FuzzInput| {
    let tk = fuzz_tokenizer();

    let mut text = input.text.as_str();
    if text.len() > MAX_TEXT_BYTES {
        let mut end = MAX_TEXT_BYTES;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text = &text[..end];
    }

    // 1. Complexity & correctness invariants on Tokenizer::encode.
    let t0 = Instant::now();
    let ids = tk.encode(text, input.parse_special);
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "super-linear BPE regression: encoding {} bytes took {:?}",
        text.len(),
        elapsed
    );

    if !input.parse_special {
        // Security invariant: encode(..., false) must NEVER emit a special/control token ID.
        for &id in &ids {
            assert!(
                !tk.is_special(id),
                "control-token smuggling: encode(..., false) emitted special token {id}"
            );
        }
        // Round-trip invariant: byte-level BPE must be lossless on arbitrary Unicode.
        let decoded: Vec<u8> = ids.iter().flat_map(|&id| tk.decode_bytes(id)).collect();
        assert_eq!(
            decoded.as_slice(),
            text.as_bytes(),
            "encode/decode round-trip mismatch"
        );
    } else {
        // Decoding every emitted ID when parse_special=true must also never panic.
        for &id in &ids {
            let _ = tk.decode_bytes(id);
        }
    }

    // 2. Arbitrary token ID decoding (including OOB IDs, u32::MAX, special tokens).
    for id in input.decode_ids.iter().take(256).copied() {
        let _ = tk.is_special(id);
        let _ = tk.decode_bytes(id);
    }
});
