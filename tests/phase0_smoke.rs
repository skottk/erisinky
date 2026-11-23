mod common;

use common::gguf_builder::{minimal_tokenizer_gguf, GgufBuilder, KvVal};
use inkling::cli::{parse_args, ParseOutcome};
use inkling::engine::tier_math;
use inkling::gguf::{Gguf, T_F32};
use inkling::tokenizer::Tokenizer;

#[test]
fn phase0_gguf_from_bytes_and_open_roundtrip() {
    let payload: Vec<u8> = [1.0f32, -2.5, 3.25, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let builder = GgufBuilder::new()
        .kv("general.architecture", KvVal::Str("test".into()))
        .kv("test.count", KvVal::U32(42))
        .tensor("w", vec![4], T_F32, payload.clone());

    let raw = builder.build();
    let g_mem = Gguf::from_bytes(&raw).expect("parse from_bytes");
    assert_eq!(g_mem.u("test.count"), 42);
    assert_eq!(g_mem.bytes(g_mem.tensor("w")), payload.as_slice());

    let tmp = builder.write_temp();
    let g_file = Gguf::open(tmp.path().to_str().unwrap()).expect("open mmap");
    assert_eq!(g_file.u("test.count"), 42);
    assert_eq!(g_file.bytes(g_file.tensor("w")), payload.as_slice());
}

#[test]
fn phase0_tokenizer_from_synthetic_gguf() {
    let raw = minimal_tokenizer_gguf(&[], &[]);
    let g = Gguf::from_bytes(&raw).expect("parse tokenizer gguf");
    let tk = Tokenizer::load(&g);
    let ids = tk.encode("Hello, world!", false);
    let bytes: Vec<u8> = ids.iter().flat_map(|&id| tk.decode_bytes(id)).collect();
    assert_eq!(String::from_utf8(bytes).unwrap(), "Hello, world!");
}

#[test]
fn phase0_cli_parse_args_smoke() {
    let res = parse_args(["run", "-p", "hi", "-c", "200", "--greedy"], "/tmp").expect("parse");
    match res {
        ParseOutcome::Command(cmd, o) => {
            assert_eq!(cmd, "run");
            assert_eq!(o.prompt, "hi");
            assert_eq!(o.ctx, 256);
            assert_eq!(o.temp, 0.0);
        }
        ParseOutcome::Help => panic!("unexpected help"),
    }
    assert!(matches!(parse_args(["run", "-h"], "/tmp"), Ok(ParseOutcome::Help)));
}

#[test]
fn phase0_tier_math_rows_rec_roundtrip() {
    let cap = 8;
    let mut buf = vec![0u8; 2 * 2 * 8 * cap * 128 * 2];
    let mut rec = vec![0u8; 2 * 8 * 128 * 2];
    for (i, b) in rec.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(7);
    }
    tier_math::rec_to_rows(&rec, &mut buf, 1, cap, 5);
    let mut out = vec![0u8; rec.len()];
    tier_math::rows_to_rec(&buf, 1, cap, 5, &mut out);
    assert_eq!(out, rec);
}
