use inkling::cli::{effort_str, parse_args, CliError, ParseOutcome};
use inkling::sampler::Sampler;
use proptest::prelude::*;
use std::collections::HashSet;

fn edge_f32_strategy() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(0.0f32),
        Just(-0.0f32),
        Just(1.0f32),
        Just(-1.0f32),
        Just(f32::from_bits(1)),        // smallest positive subnormal
        Just(-f32::from_bits(1)),       // largest negative subnormal
        Just(f32::MIN_POSITIVE),
        Just(-f32::MIN_POSITIVE),
        Just(1e-38f32),
        Just(1e38f32),
        Just(-1e38f32),
        Just(f32::MAX),
        Just(f32::MIN),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        Just(f32::NAN),
        any::<f32>(),
    ]
}

fn logits_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop_oneof![
        Just(Vec::new()),
        edge_f32_strategy().prop_map(|v| vec![v]),
        (edge_f32_strategy(), 1usize..=16).prop_map(|(v, n)| vec![v; n]),
        prop::collection::vec(edge_f32_strategy(), 0..=48),
    ]
}

fn top_k_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(0usize),
        Just(1usize),
        Just(2usize),
        Just(8usize),
        Just(64usize),
        Just(usize::MAX),
        0usize..=128,
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn prop_sampler_try_sample_and_sample_never_panic(
        logits in logits_strategy(),
        temp in edge_f32_strategy(),
        top_p in edge_f32_strategy(),
        top_k in top_k_strategy(),
        softcap in prop_oneof![Just(0.0f32), Just(30.0f32), edge_f32_strategy()],
        seed in any::<u64>(),
    ) {
        let mut s_try = Sampler::new(temp, top_k, top_p, softcap, seed);
        let res = s_try.try_sample(&logits);
        let all_nan_or_empty = logits.is_empty() || logits.iter().all(|x| x.is_nan());
        if all_nan_or_empty {
            prop_assert!(res.is_err(), "expected Err on empty or all-NaN logits, got {res:?}");
        } else {
            let id = res.expect("expected Ok when at least one non-NaN logit exists");
            prop_assert!((id as usize) < logits.len());
            prop_assert!(!logits[id as usize].is_nan());
        }

        if !logits.is_empty() {
            let mut s = Sampler::new(temp, top_k, top_p, softcap, seed);
            let id = s.sample(&logits);
            if logits.iter().all(|x| x.is_nan()) {
                prop_assert_eq!(id, 0);
            } else {
                prop_assert!((id as usize) < logits.len());
                prop_assert!(!logits[id as usize].is_nan());
            }
        }
    }

    #[test]
    fn prop_parse_args_never_panics(
        args in prop::collection::vec(
            prop_oneof![
                Just("run".to_string()),
                Just("chat".to_string()),
                Just("bench".to_string()),
                Just("-c".to_string()),
                Just("--ctx".to_string()),
                Just("-n".to_string()),
                Just("--temp".to_string()),
                Just("--top-p".to_string()),
                Just("--top-k".to_string()),
                Just("--seed".to_string()),
                Just("--effort".to_string()),
                Just("-f".to_string()),
                Just("--file".to_string()),
                Just("-m".to_string()),
                Just("--model".to_string()),
                Just("--stop".to_string()),
                Just("-p".to_string()),
                Just("--prompt".to_string()),
                Just("--system".to_string()),
                Just("--ref".to_string()),
                Just("--greedy".to_string()),
                Just("--raw".to_string()),
                Just("--kv8".to_string()),
                Just("--kv-disk".to_string()),
                Just("--json".to_string()),
                Just("--hide-thinking".to_string()),
                Just("-q".to_string()),
                Just("-h".to_string()),
                Just("--help".to_string()),
                Just("0".to_string()),
                Just("1".to_string()),
                Just("63".to_string()),
                Just("64".to_string()),
                Just("18446744073709551615".to_string()),
                Just("999999999999999".to_string()),
                Just("NaN".to_string()),
                Just("inf".to_string()),
                Just("-inf".to_string()),
                Just("-1.0".to_string()),
                Just("1.5".to_string()),
                Just("high".to_string()),
                "\\PC{0,16}",
            ],
            0..=20,
        )
    ) {
        let _ = parse_args(args, "/tmp");
    }
}

#[test]
fn sampler_empty_and_all_nan_and_single_element() {
    let mut s = Sampler::new(1.0, 0, 0.95, 30.0, 42);
    assert!(s.try_sample(&[]).is_err());
    assert_eq!(s.sample(&[]), 0);

    let all_nan = [f32::NAN; 5];
    assert!(s.try_sample(&all_nan).is_err());
    assert_eq!(s.sample(&all_nan), 0);

    for v in [0.0f32, -42.5, f32::NEG_INFINITY, f32::INFINITY, 1e38] {
        assert_eq!(s.try_sample(&[v]), Ok(0));
        assert_eq!(s.sample(&[v]), 0);
    }
}

#[test]
fn sampler_identical_seeds_produce_identical_sequences() {
    let logits = [0.2f32, 1.5, 1.4, 0.9, -0.3, 1.45, 0.8, 1.1];
    for seed in [0u64, 1, 42, 0xDEAD_BEEF_CAFE_BABE] {
        let mut s1 = Sampler::new(1.0, 0, 0.95, 30.0, seed);
        let mut s2 = Sampler::new(1.0, 0, 0.95, 30.0, seed);
        let seq1: Vec<u32> = (0..32).map(|_| s1.sample(&logits)).collect();
        let seq2: Vec<u32> = (0..32).map(|_| s2.sample(&logits)).collect();
        assert_eq!(seq1, seq2);
    }
}

#[test]
fn sampler_greedy_and_top_k_one_equal_argmax_ignoring_nan() {
    let logits = [f32::NAN, -5.0, 2.5, f32::NAN, 7.25, 3.0, f32::NEG_INFINITY];
    for temp in [0.0f32, -0.5, -100.0, f32::NEG_INFINITY] {
        let mut s = Sampler::new(temp, 0, 0.95, 30.0, 123);
        assert_eq!(s.try_sample(&logits), Ok(4));
        assert_eq!(s.sample(&logits), 4);
    }
    let mut s_k1 = Sampler::new(1.0, 1, 0.95, 30.0, 123);
    assert_eq!(s_k1.try_sample(&logits), Ok(4));
    assert_eq!(s_k1.sample(&logits), 4);
}

#[test]
fn sampler_top_k_zero_disables_top_k_full_vocab() {
    let logits = [1.0f32; 10];
    let mut seen = HashSet::new();
    for seed in 0..64u64 {
        let mut s = Sampler::new(1.0, 0, 1.0, 0.0, seed);
        let id = s.try_sample(&logits).expect("valid sample");
        assert!((id as usize) < logits.len());
        seen.insert(id);
    }
    assert!(
        seen.len() > 1,
        "top_k == 0 should sample across full vocab, but only saw {seen:?}"
    );
}

#[test]
fn sampler_tiny_positive_temp_does_not_overflow_to_nan() {
    let logits = [1.0f32, 5.0, 2.0, -3.0];
    for top_k in [0usize, 4] {
        for temp in [1e-38f32, f32::MIN_POSITIVE, f32::from_bits(1)] {
            for softcap in [0.0f32, 30.0] {
                let mut s = Sampler::new(temp, top_k, 0.95, softcap, 7);
                assert_eq!(
                    s.try_sample(&logits),
                    Ok(1),
                    "top_k={top_k}, temp={temp:e}, softcap={softcap}"
                );
            }
        }
    }

    let extreme = [1e38f32, -1e38, 0.0];
    for top_k in [0usize, 3] {
        let mut s = Sampler::new(f32::MIN_POSITIVE, top_k, 1.0, 0.0, 7);
        assert_eq!(s.try_sample(&extreme), Ok(0));
    }
}

#[test]
fn sampler_nan_temp_does_not_panic_or_produce_nan() {
    let logits = [f32::NAN, 1.0, 3.5, 2.0];
    for top_k in [0usize, 4] {
        let mut s = Sampler::new(f32::NAN, top_k, 0.95, 30.0, 99);
        assert_eq!(s.try_sample(&logits), Ok(2));
        assert_eq!(s.sample(&logits), 2);
    }
}

#[test]
fn sampler_pos_inf_logits_samples_only_inf_candidates() {
    let logits = [1.0f32, f32::INFINITY, 2.0, f32::INFINITY, f32::NEG_INFINITY];
    let mut seen = HashSet::new();
    for seed in 0..64u64 {
        let mut s = Sampler::new(1.0, 0, 1.0, 0.0, seed);
        let id = s.try_sample(&logits).expect("valid sample");
        assert!(id == 1 || id == 3, "unexpected non-inf token {id}");
        seen.insert(id);
    }
    assert_eq!(seen.len(), 2);
}

#[test]
fn cli_rejects_ctx_usize_max_overflow() {
    let res = parse_args(["run", "-c", "18446744073709551615"], "/tmp");
    assert!(
        matches!(res, Err(CliError::InvalidValue { .. })),
        "expected InvalidValue for -c usize::MAX, got {res:?}"
    );
}

#[test]
fn cli_rejects_invalid_ctx_values() {
    let bad = [
        "0",
        "1",
        "63",
        "18446744073709551615",
        "999999999999999",
        "262144",
        "-64",
        "not_a_num",
    ];
    for val in bad {
        let res = parse_args(["run", "-c", val], "/tmp");
        assert!(
            matches!(res, Err(CliError::InvalidValue { .. })),
            "expected InvalidValue for -c {val}, got {res:?}"
        );
    }

    // Valid boundaries
    for (raw, expected) in [("64", 64usize), ("65", 128), ("200", 256), ("262080", 262080)] {
        match parse_args(["run", "-c", raw], "/tmp") {
            Ok(ParseOutcome::Command(_, o)) => assert_eq!(o.ctx, expected),
            other => panic!("expected Ok for -c {raw}, got {other:?}"),
        }
    }
}

#[test]
fn cli_rejects_invalid_temp_top_p_and_effort() {
    for val in ["NaN", "inf", "-inf", "-1.0", "-0.001"] {
        let res = parse_args(["run", "--temp", val], "/tmp");
        assert!(
            matches!(res, Err(CliError::InvalidValue { .. })),
            "expected InvalidValue for --temp {val}, got {res:?}"
        );
    }
    for val in ["0", "0.0", "0.7", "1.0", "2.5"] {
        assert!(
            parse_args(["run", "--temp", val], "/tmp").is_ok(),
            "expected Ok for --temp {val}"
        );
    }

    for val in ["NaN", "inf", "-inf", "0.0", "-0.5", "1.5"] {
        let res = parse_args(["run", "--top-p", val], "/tmp");
        assert!(
            matches!(res, Err(CliError::InvalidValue { .. })),
            "expected InvalidValue for --top-p {val}, got {res:?}"
        );
    }
    for val in ["0.01", "0.95", "1.0"] {
        assert!(
            parse_args(["run", "--top-p", val], "/tmp").is_ok(),
            "expected Ok for --top-p {val}"
        );
    }

    for val in ["NaN", "inf", "-inf", "-0.1", "1.5", "1.0", "bogus", ""] {
        let res = parse_args(["run", "--effort", val], "/tmp");
        assert!(
            matches!(res, Err(CliError::InvalidValue { .. })),
            "expected InvalidValue for --effort {val}, got {res:?}"
        );
    }
    for val in [
        "none", "minimal", "low", "medium", "high", "xhigh", "max", "0", "0.0", "0.5", "0.99",
    ] {
        assert!(
            parse_args(["run", "--effort", val], "/tmp").is_ok(),
            "expected Ok for --effort {val}"
        );
    }
}

#[test]
fn cli_rejects_missing_value_for_every_value_flag() {
    let flags = [
        "-m", "--model", "-p", "--prompt", "-f", "--file", "-n", "-c", "--ctx", "--effort",
        "--system", "--temp", "--top-p", "--top-k", "--seed", "--ref", "--stop",
    ];
    for flag in flags {
        let res = parse_args(["run", flag], "/tmp");
        assert!(
            matches!(res, Err(CliError::MissingValue(ref f)) if f == flag),
            "expected MissingValue({flag}), got {res:?}"
        );
    }
}

#[test]
fn cli_seed_zero_sets_seed_set_flag() {
    match parse_args(["run", "--seed", "0"], "/tmp").expect("parse --seed 0") {
        ParseOutcome::Command(_, o) => {
            assert_eq!(o.seed, 0);
            assert!(o.seed_set, "--seed 0 must set seed_set = true");
        }
        ParseOutcome::Help => panic!("unexpected help"),
    }

    match parse_args(["run", "--seed", "42"], "/tmp").expect("parse --seed 42") {
        ParseOutcome::Command(_, o) => {
            assert_eq!(o.seed, 42);
            assert!(o.seed_set);
        }
        ParseOutcome::Help => panic!("unexpected help"),
    }

    match parse_args(["run"], "/tmp").expect("parse default") {
        ParseOutcome::Command(_, o) => {
            assert_eq!(o.seed, 0);
            assert!(!o.seed_set, "omitting --seed must leave seed_set = false");
        }
        ParseOutcome::Help => panic!("unexpected help"),
    }
}

#[test]
fn effort_str_rejects_non_finite_and_falls_back_safely() {
    for bad in ["NaN", "nan", "inf", "-inf", "infinity", "-infinity", "bogus"] {
        let out = effort_str(bad);
        assert_eq!(out, "0.9", "effort_str({bad:?}) returned {out:?}");
        assert!(!out.contains("NaN") && !out.contains("inf"));
    }
    assert_eq!(effort_str("none"), "0");
    assert_eq!(effort_str("minimal"), "0.1");
    assert_eq!(effort_str("low"), "0.2");
    assert_eq!(effort_str("medium"), "0.7");
    assert_eq!(effort_str("high"), "0.9");
    assert_eq!(effort_str("xhigh"), "0.99");
    assert_eq!(effort_str("max"), "0.99");
    assert_eq!(effort_str("0.5"), "0.5");
}
