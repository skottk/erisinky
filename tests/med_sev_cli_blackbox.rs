//! Black-box CLI regression tests for Medium-Severity #1 (unhandled CLI panics),
//! #2 (`main.rs` bounds/ref guards), #3 (`--seed 0` wiring), and #4 (empty prompt handling).

mod common;

use common::gguf_builder::{minimal_tokenizer_gguf, GgufBuilder};
use inkling::cli::{parse_args, ParseOutcome};
use std::io::Write;
use std::process::{Command, Output};

fn run_inkling(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_inkling"))
        .args(args)
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("spawn inkling binary")
}

fn assert_clean_failure(args: &[&str], label: &str) -> String {
    let out = run_inkling(args);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "[{label}] expected non-zero exit for args {args:?}, got success (stdout={:?}, stderr={stderr:?})",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        !stderr.trim().is_empty(),
        "[{label}] expected non-empty diagnostic on stderr for args {args:?}"
    );
    assert!(
        !stderr.contains("panicked at"),
        "[{label}] binary panicked on args {args:?}:\n{stderr}"
    );
    stderr
}

#[test]
fn missing_model_file_exits_cleanly_without_panic() {
    let missing = "/nonexistent/inkling-does-not-exist.gguf";
    let cases: &[(&[&str], &str)] = &[
        (&["run", "-m", missing, "-p", "hi"], "run missing -m"),
        (&["chat", "-m", missing], "chat missing -m"),
        (&["bench", "-m", missing, "-p", "hi"], "bench missing -m"),
        (&["debug", "-m", missing, "-p", "hi"], "debug missing -m"),
        (&["prof", "-m", missing], "prof missing -m"),
        (&["prof_pf", "-m", missing], "prof_pf missing -m"),
        (&["vcost", "-m", missing], "vcost missing -m"),
        (&["kmoepf", "-m", missing, "-p", "hi"], "kmoepf missing -m"),
        (&["kvprune", "-m", missing, "-p", "hi"], "kvprune missing -m"),
        (&["kvscreen", "-m", missing, "-p", "hi"], "kvscreen missing -m"),
        (&["kmoe", "-m", missing], "kmoe missing -m"),
        (&["kgemm", "-m", missing], "kgemm missing -m"),
        (&["ksg", "-m", missing], "ksg missing -m"),
        (&["qclip", "-m", missing], "qclip missing -m"),
        (&["kpeak", "-m", missing], "kpeak missing -m"),
        (&["kmode", "-m", missing], "kmode missing -m"),
        (&["bw", "-m", missing], "bw missing -m"),
        (&["kbench", "-m", missing], "kbench missing -m"),
    ];
    for &(args, label) in cases {
        let stderr = assert_clean_failure(args, label);
        assert!(
            stderr.contains("error:") || stderr.contains("load model"),
            "[{label}] expected clean error diagnostic, got: {stderr:?}"
        );
    }
}

#[test]
fn missing_prompt_file_exits_cleanly_without_panic() {
    let stderr = assert_clean_failure(
        &["run", "-f", "/nonexistent/prompt.txt"],
        "missing -f prompt file",
    );
    assert!(
        stderr.contains("prompt file") || stderr.contains("error:"),
        "expected prompt file error message, got: {stderr:?}"
    );
}

#[test]
fn non_utf8_prompt_file_exits_cleanly_without_panic() {
    let mut tmp = tempfile::NamedTempFile::new().expect("create temp prompt file");
    tmp.write_all(&[0xFF, 0xFE, 0x00]).expect("write invalid utf8");
    tmp.flush().expect("flush temp prompt file");
    let path = tmp.path().to_str().expect("utf8 temp path");

    let stderr = assert_clean_failure(&["run", "-f", path], "non-UTF-8 -f prompt file");
    assert!(
        stderr.contains("prompt file") || stderr.contains("error:"),
        "expected prompt file UTF-8 error message, got: {stderr:?}"
    );
}

#[test]
fn help_flag_after_nonexistent_prompt_file_exits_zero() {
    for help_flag in ["-h", "--help"] {
        let out = run_inkling(&["run", "-f", "/nonexistent/prompt.txt", help_flag]);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "expected exit 0 for `-f /nonexistent/prompt.txt {help_flag}`, got status {:?}, stderr={stderr:?}",
            out.status.code()
        );
        assert!(
            stdout.contains("usage: inkling"),
            "expected usage text on stdout for `{help_flag}`, got: {stdout:?}"
        );
        assert!(
            !stderr.contains("panicked at"),
            "unexpected panic on stderr: {stderr}"
        );
    }
}

#[test]
fn malformed_gguf_exits_cleanly_without_panic() {
    // 1. Bad magic
    let mut bad_magic = GgufBuilder::new();
    bad_magic.magic = *b"BAD!";
    let f_magic = bad_magic.write_temp();

    // 2. Truncated header
    let mut truncated = GgufBuilder::new();
    truncated.truncate_to = Some(12);
    let f_trunc = truncated.write_temp();

    // 3. Valid GGUF container missing Inkling architecture / tensors
    let mut f_incomplete = tempfile::NamedTempFile::new().expect("create incomplete gguf");
    f_incomplete
        .write_all(&minimal_tokenizer_gguf(&[], &[]))
        .expect("write incomplete gguf");
    f_incomplete.flush().expect("flush incomplete gguf");

    // 4. Empty 0-byte file
    let f_empty = tempfile::NamedTempFile::new().expect("create empty gguf");

    for (label, file) in [
        ("bad magic GGUF", &f_magic),
        ("truncated GGUF", &f_trunc),
        ("incomplete GGUF missing tensors", &f_incomplete),
        ("empty 0-byte file", &f_empty),
    ] {
        let path = file.path().to_str().expect("utf8 path");
        let stderr = assert_clean_failure(&["run", "-m", path, "-p", "hi"], label);
        assert!(
            stderr.contains("error:") || stderr.contains("load model"),
            "[{label}] expected clean diagnostic, got: {stderr:?}"
        );
    }
}

#[test]
fn invalid_flags_exit_cleanly_without_panic() {
    let cases: &[(&[&str], &str)] = &[
        (&["run", "-c", "0"], "-c 0"),
        (&["run", "-c", "63"], "-c 63"),
        (&["run", "-c", "999999999999999"], "-c huge"),
        (&["run", "-n", "notanint"], "-n notanint"),
        (&["run", "--temp", "NaN"], "--temp NaN"),
        (&["run", "--temp", "-1"], "--temp -1"),
        (&["run", "--top-p", "0"], "--top-p 0"),
        (&["run", "--top-p", "1.5"], "--top-p 1.5"),
        (&["run", "--effort", "NaN"], "--effort NaN"),
        (&["run", "--effort", "bogus"], "--effort bogus"),
        (&["run", "--top-k", "nope"], "--top-k nope"),
        (&["run", "--seed", "nope"], "--seed nope"),
        (&["run", "--stop", "nope"], "--stop nope"),
        (&["run", "-p"], "missing value for -p"),
        (&["run", "-f"], "missing value for -f"),
        (&["run", "-m"], "missing value for -m"),
        (&["run", "-c"], "missing value for -c"),
        (&["run", "--unknown-flag"], "unknown option"),
    ];
    for &(args, label) in cases {
        assert_clean_failure(args, label);
    }
}

#[test]
fn check_routes_kvdump_missing_or_invalid_ref_exit_cleanly_without_panic() {
    let mut bad_gguf = GgufBuilder::new();
    bad_gguf.magic = *b"BAD!";
    let f_bad = bad_gguf.write_temp();
    let bad_path = f_bad.path().to_str().expect("utf8 path");

    // check with nonexistent --ref and malformed / missing model
    assert_clean_failure(
        &["check", "-m", bad_path, "--ref", "/nonexistent/ref"],
        "check malformed -m + nonexistent --ref",
    );
    assert_clean_failure(
        &[
            "check",
            "-m",
            "/nonexistent/model.gguf",
            "--ref",
            "/nonexistent/ref",
        ],
        "check missing -m + nonexistent --ref",
    );

    // missing --ref in check, routes, kvdump
    let stderr_check = assert_clean_failure(&["check", "-m", bad_path], "check missing --ref");
    assert!(
        stderr_check.contains("--ref"),
        "expected --ref diagnostic for check, got: {stderr_check:?}"
    );
    let stderr_routes =
        assert_clean_failure(&["routes", "-m", bad_path, "-p", "hi"], "routes missing --ref");
    assert!(
        stderr_routes.contains("--ref"),
        "expected --ref diagnostic for routes, got: {stderr_routes:?}"
    );
    let stderr_kvdump =
        assert_clean_failure(&["kvdump", "-m", bad_path, "-p", "hi"], "kvdump missing --ref");
    assert!(
        stderr_kvdump.contains("--ref"),
        "expected --ref diagnostic for kvdump, got: {stderr_kvdump:?}"
    );

    // check with syntactically invalid .tokens and misaligned .logits
    let dir = tempfile::tempdir().expect("create temp ref dir");
    let prefix = dir.path().join("badref");
    let prefix_str = prefix.to_str().expect("utf8 prefix");
    std::fs::write(format!("{prefix_str}.tokens"), "not_a_token\n").expect("write bad tokens");
    std::fs::write(format!("{prefix_str}.logits"), [0u8; 4]).expect("write logits");
    let stderr_tok = assert_clean_failure(
        &["check", "-m", bad_path, "--ref", prefix_str],
        "check invalid token id",
    );
    assert!(
        stderr_tok.contains("invalid token id"),
        "expected invalid token id error, got: {stderr_tok:?}"
    );

    std::fs::write(format!("{prefix_str}.tokens"), "   \n\n").expect("write empty tokens");
    let stderr_empty = assert_clean_failure(
        &["check", "-m", bad_path, "--ref", prefix_str],
        "check empty tokens file",
    );
    assert!(
        stderr_empty.contains("empty"),
        "expected empty tokens error, got: {stderr_empty:?}"
    );

    std::fs::write(format!("{prefix_str}.tokens"), "1\n2\n").expect("write valid tokens");
    std::fs::write(format!("{prefix_str}.logits"), [0u8; 3]).expect("write misaligned logits");
    let stderr_align = assert_clean_failure(
        &["check", "-m", bad_path, "--ref", prefix_str],
        "check misaligned logits",
    );
    assert!(
        stderr_align.contains("multiple of 4"),
        "expected multiple-of-4 error, got: {stderr_align:?}"
    );
}

#[test]
fn empty_raw_prompt_fails_fast_cleanly_without_panic() {
    let stderr_run = assert_clean_failure(
        &["run", "--raw", "-p", "", "-m", "/nonexistent/model.gguf"],
        "run --raw empty prompt",
    );
    assert!(
        stderr_run.contains("empty prompt"),
        "expected fast empty-prompt error for run, got: {stderr_run:?}"
    );

    let stderr_bench = assert_clean_failure(
        &["bench", "--raw", "-p", "", "-m", "/nonexistent/model.gguf"],
        "bench --raw empty prompt",
    );
    assert!(
        stderr_bench.contains("empty prompt"),
        "expected fast empty-prompt error for bench, got: {stderr_bench:?}"
    );
}

#[test]
fn prof_pf_and_vcost_reject_insufficient_ctx_cleanly_without_panic() {
    let stderr_pf = assert_clean_failure(
        &["prof_pf", "-c", "4096", "-m", "/nonexistent/model.gguf"],
        "prof_pf with -c 4096",
    );
    assert!(
        stderr_pf.contains("8192"),
        "expected ctx >= 8192 diagnostic for prof_pf, got: {stderr_pf:?}"
    );

    let stderr_vc = assert_clean_failure(
        &["vcost", "-c", "2048", "-m", "/nonexistent/model.gguf"],
        "vcost with -c 2048",
    );
    assert!(
        stderr_vc.contains("4096"),
        "expected ctx >= 4096 diagnostic for vcost, got: {stderr_vc:?}"
    );
}

#[test]
fn seed_zero_is_preserved_in_cli_opts_and_accepted_by_binary() {
    match parse_args(["run", "--seed", "0"], "/tmp").expect("parse --seed 0") {
        ParseOutcome::Command(_, o) => {
            assert_eq!(o.seed, 0);
            assert!(o.seed_set, "explicit --seed 0 must set seed_set = true");
        }
        ParseOutcome::Help => panic!("unexpected Help"),
    }

    match parse_args(["run"], "/tmp").expect("parse default without --seed") {
        ParseOutcome::Command(_, o) => {
            assert_eq!(o.seed, 0);
            assert!(!o.seed_set, "omitting --seed must leave seed_set = false");
        }
        ParseOutcome::Help => panic!("unexpected Help"),
    }

    // Binary accepts --seed 0 cleanly and fails later on missing model without panicking
    let stderr = assert_clean_failure(
        &["run", "--seed", "0", "-m", "/nonexistent/model.gguf", "-p", "hi"],
        "run --seed 0 missing model",
    );
    assert!(
        stderr.contains("load model") || stderr.contains("error:"),
        "expected model load error after valid --seed 0, got: {stderr:?}"
    );
}
