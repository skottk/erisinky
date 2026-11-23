//! Red -> Green tests for `inkling verify` (CLI + differential Fast-vs-Safe logit comparison).

mod common;

use common::gguf_builder::{GgufBuilder, KvVal};
use inkling::cli::{diff_logits, parse_args, ParseOutcome, USAGE};
use inkling::gguf::T_F32;
use std::process::Command;

fn inkling_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_inkling"))
}

#[test]
fn usage_and_cli_parser_support_verify_subcommand() {
    assert!(
        USAGE.contains("verify"),
        "USAGE must document the `verify` subcommand"
    );
    let parsed = parse_args(["verify", "-m", "model.gguf", "--json"], "/tmp").expect("parse verify");
    match parsed {
        ParseOutcome::Command(cmd, o) => {
            assert_eq!(cmd, "verify");
            assert_eq!(o.model, "model.gguf");
            assert!(o.json);
        }
        ParseOutcome::Help => panic!("unexpected help"),
    }
}

#[test]
fn diff_logits_computes_exact_parity_kl_and_non_finite_counts() {
    let safe = [1.0f32, 3.0, 2.0, -1.0];
    let fast_same = [1.0f32, 3.0, 2.0, -1.0];
    let d0 = diff_logits(&fast_same, &safe, 30.0);
    assert_eq!(d0.non_finite_fast, 0);
    assert_eq!(d0.non_finite_safe, 0);
    assert!(d0.max_abs_diff < 1e-6);
    assert!(d0.kl_div.abs() < 1e-6);
    assert_eq!(d0.argmax_fast, 1);
    assert_eq!(d0.argmax_safe, 1);
    assert!(d0.argmax_match());

    // Slight perturbation preserves argmax and produces small positive KL
    let fast_close = [1.01f32, 2.99, 2.02, -0.98];
    let d1 = diff_logits(&fast_close, &safe, 30.0);
    assert_eq!(d1.non_finite_fast, 0);
    assert!(d1.max_abs_diff > 0.0 && d1.max_abs_diff < 0.03);
    assert!(d1.kl_div >= 0.0 && d1.kl_div < 1e-2);
    assert!(d1.argmax_match());

    // Non-finite logit in fast path must be counted cleanly without NaN panic
    let fast_nan = [1.0f32, f32::NAN, f32::INFINITY, -1.0];
    let d2 = diff_logits(&fast_nan, &safe, 30.0);
    assert_eq!(d2.non_finite_fast, 2);
    assert_eq!(d2.non_finite_safe, 0);
}

#[test]
fn verify_missing_model_exits_nonzero_without_panic() {
    let out = inkling_bin()
        .args(["verify", "-m", "/nonexistent/inkling-verify-missing.gguf"])
        .output()
        .expect("run inkling verify");
    assert!(!out.status.success(), "expected non-zero exit on missing model");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "verify must not panic on missing model; stderr:\n{stderr}"
    );
    assert!(!stderr.trim().is_empty(), "expected diagnostic on stderr");
}

#[test]
fn verify_rejects_malformed_or_nan_gguf_cleanly_without_panic() {
    let payload: Vec<u8> = [1.0f32, f32::NAN, 2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let tmp = GgufBuilder::new()
        .kv("general.architecture", KvVal::Str("inkling2_loop".into()))
        .tensor("bad_f32", vec![4], T_F32, payload)
        .write_temp();

    let out = inkling_bin()
        .args(["verify", "-m", tmp.path().to_str().unwrap()])
        .output()
        .expect("run inkling verify on malformed gguf");
    assert!(!out.status.success(), "expected non-zero exit on malformed/NaN gguf");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "verify must not panic on malformed gguf; stderr:\n{stderr}"
    );
    assert!(!stderr.trim().is_empty(), "expected diagnostic on stderr");
}
