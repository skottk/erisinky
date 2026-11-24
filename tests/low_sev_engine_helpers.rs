//! Regression tests for Low-Severity #2 (inverted min/max in kbench/kbench_down)
//! and Low-Severity #4a/b/c (kv_screen underflow, kv_quant_sim div-by-zero/shift overflow,
//! INK_HEADROOM=0,0 div-by-zero).

use inkling::engine::dev_helpers::{
    best_bandwidth_gbps, parse_headroom, quantize_row_sim, kv_screen_range, reserve_capacity,
};

#[test]
fn kbench_best_bandwidth_picks_fastest_not_slowest() {
    let samples = [210.5, 285.2, 240.0];
    let best = best_bandwidth_gbps(samples);
    assert!(
        (best - 285.2).abs() < 1e-9,
        "expected fastest bandwidth 285.2 GB/s, got {best}"
    );
    assert_eq!(best_bandwidth_gbps(std::iter::empty::<f64>()), 0.0);
    // Ignore non-finite samples (e.g., t == 0.0 -> inf or NaN).
    assert!((best_bandwidth_gbps([f64::NAN, f64::INFINITY, 150.0, 200.0]) - 200.0).abs() < 1e-9);
}

#[test]
fn reserve_capacity_handles_ink_headroom_zero_zero_without_div_by_zero() {
    assert_eq!(parse_headroom(Some("0,0")), (0, 0));
    assert_eq!(parse_headroom(Some("bad,input")), (16, 0));
    assert_eq!(parse_headroom(None), (16, 0));

    // INK_HEADROOM=0,0 must not divide by zero when headroom is true.
    let cap = reserve_capacity(100, 4096, true, 0, 0);
    assert_eq!(cap, 128);

    // Normal headroom (16, 0): 100 + 100/16 = 106 -> rounded up to multiple of 64 = 128.
    assert_eq!(reserve_capacity(100, 4096, true, 16, 0), 128);
    // Respects limit cap.
    assert_eq!(reserve_capacity(4090, 4096, true, 1, 512), 4096);
}

#[test]
fn kv_screen_range_does_not_underflow_for_short_prompts() {
    // Prompt with < 48 tokens (nq = 48): must saturate at 0..n instead of underflowing.
    assert_eq!(kv_screen_range(10, 48), 0..10);
    assert_eq!(kv_screen_range(0, 48), 0..0);
    assert_eq!(kv_screen_range(48, 48), 0..48);
    assert_eq!(kv_screen_range(100, 48), 52..100);
    assert_eq!(kv_screen_range(100, 0), 100..100);
}

#[test]
fn kv_quant_sim_handles_bits_1_and_33_to_99_and_float_formats_without_nan_or_overflow() {
    let input = [-1.5f32, -0.25, 0.0, 0.5, 2.0];

    // bits == 0: untouched.
    let mut r0 = input.to_vec();
    quantize_row_sim(&mut r0, 0);
    assert_eq!(r0, input);

    // bits == 1: must not divide by 0.0 or produce NaN.
    let mut r1 = input.to_vec();
    quantize_row_sim(&mut r1, 1);
    assert!(r1.iter().all(|v| v.is_finite()), "bits==1 produced non-finite: {r1:?}");
    assert_eq!(r1[0], -2.0);
    assert_eq!(r1[4], 2.0);

    // bits in 33..100: must not overflow shift (1u32 << (bits - 1)) or produce NaN.
    for bits in [32u32, 33, 64, 99] {
        let mut r = input.to_vec();
        quantize_row_sim(&mut r, bits);
        assert!(
            r.iter().all(|v| v.is_finite()),
            "bits=={bits} produced non-finite: {r:?}"
        );
        for (&a, &b) in input.iter().zip(&r) {
            assert!((a - b).abs() < 1e-3, "bits=={bits} distorted {a} -> {b}");
        }
    }

    // Float formats (100+m, 200+m): must not shift-overflow before branching.
    for bits in [104u32, 204] {
        let mut r = input.to_vec();
        quantize_row_sim(&mut r, bits);
        assert!(
            r.iter().all(|v| v.is_finite()),
            "bits=={bits} produced non-finite: {r:?}"
        );
    }

    // Standard 4-bit linear quantization stays accurate and finite.
    let mut r4 = input.to_vec();
    quantize_row_sim(&mut r4, 4);
    assert!(r4.iter().all(|v| v.is_finite()));
    assert!((r4[4] - 2.0).abs() < 1e-4);
}
