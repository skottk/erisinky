mod common;

use common::gguf_builder::{GgufBuilder, KvVal};
use inkling::engine::{bounds::validate_f32_bytes_finite, Engine};
use inkling::sampler::Sampler;
use inkling::session::{
    generate_with_backend, sample_logits_with_canary, DecodeBackend, DecodeState, Stats,
};
use objc2_metal::MTLMathMode;
use proptest::prelude::*;
use std::cell::{Cell, RefCell};
use std::io::ErrorKind;

fn floats_to_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn pb_finiteness_check_rejects_nan_and_inf_and_accepts_finite() {
    // Finite values including -0.0, subnormals, MIN_POSITIVE, MIN, MAX, and empty slice
    let finite_vals = [
        0.0f32,
        -0.0f32,
        1.0f32,
        -1.0f32,
        f32::from_bits(1),          // smallest positive subnormal
        f32::from_bits(0x8000_0001), // smallest negative subnormal
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
    ];
    let finite_bytes = floats_to_le_bytes(&finite_vals);
    assert!(validate_f32_bytes_finite("blk.0.attn_norm.weight", &finite_bytes).is_ok());
    assert!(validate_f32_bytes_finite("blk.0.attn_norm.weight", &[]).is_ok());

    // Quiet NaN, signaling NaN, +Inf, -Inf (both alone and embedded after valid floats)
    let non_finite_cases: &[(&str, f32)] = &[
        ("quiet_nan", f32::NAN),
        ("signaling_nan", f32::from_bits(0x7f80_0001)),
        ("neg_nan", f32::from_bits(0xffc0_0000)),
        ("pos_inf", f32::INFINITY),
        ("neg_inf", f32::NEG_INFINITY),
    ];
    for &(label, bad) in non_finite_cases {
        let bytes = floats_to_le_bytes(&[1.0, -2.5, bad, 3.0]);
        let err = validate_f32_bytes_finite("blk.0.attn_norm.weight", &bytes)
            .expect_err(&format!("{label} must be rejected"));
        assert_eq!(err.kind(), ErrorKind::InvalidData, "{label}: {err}");
        assert!(
            err.to_string().contains("blk.0.attn_norm.weight"),
            "error should mention tensor name for {label}: {err}"
        );
    }

    // Non-multiple-of-4 byte slices
    for bad_len in [1usize, 2, 3, 5, 6, 7] {
        let bytes = vec![0u8; bad_len];
        let err = validate_f32_bytes_finite("blk.0.attn_norm.weight", &bytes)
            .expect_err(&format!("len {bad_len} must be rejected"));
        assert_eq!(err.kind(), ErrorKind::InvalidData, "len={bad_len}: {err}");
    }
}

/// Configurable mock backend for testing greedy and sampled logit finiteness canaries.
struct CanaryMockBackend {
    max_ctx: usize,
    tokens: RefCell<Vec<u32>>,
    /// Sequence of tokens emitted on greedy decode steps (after `first`).
    greedy_next_toks: Vec<u32>,
    /// Per-step logit tables (indexed by decode step counter: 0 = initial/first, 1 = after step 0, ...).
    step_logits: Vec<Vec<f32>>,
    step_idx: Cell<usize>,
    wait_idx: Cell<usize>,
    sample_idx: Cell<usize>,
    capacity: Cell<usize>,
}

impl CanaryMockBackend {
    fn new_greedy(max_ctx: usize, greedy_next_toks: Vec<u32>, step_logits: Vec<Vec<f32>>) -> Self {
        Self {
            max_ctx,
            tokens: RefCell::new(vec![0; max_ctx + 8]),
            greedy_next_toks,
            step_logits,
            step_idx: Cell::new(0),
            wait_idx: Cell::new(0),
            sample_idx: Cell::new(0),
            capacity: Cell::new(max_ctx),
        }
    }

    fn new_sampled(max_ctx: usize, step_logits: Vec<Vec<f32>>) -> Self {
        Self {
            max_ctx,
            tokens: RefCell::new(vec![0; max_ctx + 8]),
            greedy_next_toks: Vec::new(),
            step_logits,
            step_idx: Cell::new(0),
            wait_idx: Cell::new(0),
            sample_idx: Cell::new(0),
            capacity: Cell::new(max_ctx),
        }
    }

    fn current_logits(&self) -> &[f32] {
        let idx = self.wait_idx.get().min(self.step_logits.len().saturating_sub(1));
        &self.step_logits[idx]
    }
}

impl DecodeBackend for CanaryMockBackend {
    type Step = usize;

    fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn release_batch(&self) {}

    fn sample_logits(&self, sampler: &mut Sampler) -> u32 {
        let idx = self.sample_idx.get();
        self.sample_idx.set(idx + 1);
        let logits = &self.step_logits[idx.min(self.step_logits.len() - 1)];
        sampler.sample(logits)
    }

    fn sample_logits_checked(&self, sampler: &mut Sampler) -> (u32, bool) {
        let idx = self.sample_idx.get();
        self.sample_idx.set(idx + 1);
        let logits = &self.step_logits[idx.min(self.step_logits.len() - 1)];
        sample_logits_with_canary(logits, sampler)
    }

    fn logit_at(&self, tok: usize) -> f32 {
        self.current_logits()
            .get(tok)
            .copied()
            .unwrap_or(f32::NAN)
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
        let idx = self.step_idx.get();
        self.step_idx.set(idx + 1);
        let next = *self.greedy_next_toks.get(idx).unwrap_or(&0);
        self.tokens.borrow_mut()[pos + 1] = next;
        idx + 1
    }

    fn wait_step(&self, step: &usize) {
        self.wait_idx.set(*step);
    }

    fn token_at(&self, pos: usize) -> u32 {
        self.tokens.borrow()[pos]
    }
}

#[test]
fn session_greedy_canary_detects_non_finite_winning_logit() {
    // Case 1: NaN on a subsequent greedy decode step (step 1 produces tok=2 with NaN logit)
    let eng = CanaryMockBackend::new_greedy(
        64,
        vec![2, 3, 1],
        vec![
            vec![1.0, 2.0, 3.0, 4.0],        // step 0 (`first` = 1 -> finite 2.0)
            vec![1.0, 2.0, f32::NAN, 4.0],   // step 1 (tok = 2 -> NaN!)
            vec![1.0, 2.0, 3.0, 4.0],        // step 2 (tok = 3 -> finite 4.0)
            vec![1.0, f32::INFINITY, 3.0, 4.0], // step 3 (tok = 1 -> +Inf!)
        ],
    );
    let mut n_past = 4usize;
    let mut pending = None;
    let mut stats = Stats::default();
    let mut sampler = Sampler::new(0.0, 0, 1.0, 0.0, 0);
    let mut emitted = Vec::new();
    let mut state = DecodeState {
        eos: 99,
        n_past: &mut n_past,
        pending: &mut pending,
        stats: &mut stats,
    };
    let n = generate_with_backend(&eng, &mut state, 1, 4, &mut sampler, |tok| {
        emitted.push(tok);
        true
    });
    assert_eq!(n, 4);
    assert_eq!(emitted, vec![1, 2, 3, 1]);
    assert_eq!(
        stats.non_finite_logits, 2,
        "expected 2 non-finite greedy winning logits (1 NaN + 1 +Inf)"
    );

    // Case 2: Non-finite winning logit on the very first token (`first`)
    let eng_first = CanaryMockBackend::new_greedy(
        64,
        vec![],
        vec![vec![1.0, f32::NEG_INFINITY, 2.0]],
    );
    let mut n_past2 = 4usize;
    let mut pending2 = None;
    let mut stats2 = Stats::default();
    let mut state2 = DecodeState {
        eos: 99,
        n_past: &mut n_past2,
        pending: &mut pending2,
        stats: &mut stats2,
    };
    let n2 = generate_with_backend(&eng_first, &mut state2, 1, 1, &mut sampler, |_| true);
    assert_eq!(n2, 1);
    assert!(
        stats2.non_finite_logits >= 1,
        "expected non-finite detection on `first` token"
    );
}

#[test]
fn session_sampled_canary_detects_non_finite_logits() {
    // Case 1: +Infinity in top candidate wins sampling
    let eng_inf = CanaryMockBackend::new_sampled(
        64,
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, f32::INFINITY, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ],
    );
    let mut n_past = 4usize;
    let mut pending = None;
    let mut stats = Stats::default();
    let mut sampler = Sampler::new(0.8, 4, 0.95, 0.0, 123);
    let mut state = DecodeState {
        eos: 99,
        n_past: &mut n_past,
        pending: &mut pending,
        stats: &mut stats,
    };
    let n = generate_with_backend(&eng_inf, &mut state, 0, 3, &mut sampler, |_| true);
    assert_eq!(n, 3);
    assert!(
        stats.non_finite_logits >= 1,
        "expected +Inf winning logit in sampled mode to be detected"
    );

    // Case 2: All NaN logits (`try_sample` returns Err, falls back to 0 without panicking)
    let eng_all_nan = CanaryMockBackend::new_sampled(
        64,
        vec![vec![f32::NAN; 4], vec![1.0, 2.0, 3.0, 4.0]],
    );
    let mut n_past2 = 4usize;
    let mut pending2 = None;
    let mut stats2 = Stats::default();
    let mut state2 = DecodeState {
        eos: 99,
        n_past: &mut n_past2,
        pending: &mut pending2,
        stats: &mut stats2,
    };
    let n2 = generate_with_backend(&eng_all_nan, &mut state2, 0, 2, &mut sampler, |_| true);
    assert_eq!(n2, 2);
    assert_eq!(
        stats2.non_finite_logits, 1,
        "expected all-NaN logits in sampled mode to be detected"
    );

    // Case 3: Partial NaN alongside finite logits
    let eng_partial_nan = CanaryMockBackend::new_sampled(
        64,
        vec![vec![f32::NAN, 10.0, 5.0, 1.0]],
    );
    let mut n_past3 = 4usize;
    let mut pending3 = None;
    let mut stats3 = Stats::default();
    let mut state3 = DecodeState {
        eos: 99,
        n_past: &mut n_past3,
        pending: &mut pending3,
        stats: &mut stats3,
    };
    let n3 = generate_with_backend(&eng_partial_nan, &mut state3, 0, 1, &mut sampler, |_| true);
    assert_eq!(n3, 1);
    assert_eq!(
        stats3.non_finite_logits, 1,
        "expected partial NaN in sampled logits to be detected"
    );

    // Case 4: Empty logits returns (0, true) without panicking
    let (tok_empty, bad_empty) = sample_logits_with_canary(&[], &mut sampler);
    assert_eq!(tok_empty, 0);
    assert!(bad_empty);
}

#[test]
fn session_canary_stays_zero_on_clean_finite_generation() {
    // Greedy clean generation
    let eng_greedy = CanaryMockBackend::new_greedy(
        64,
        vec![2, 3, 0],
        vec![
            vec![1.0, 5.0, 2.0, 3.0],
            vec![1.0, 2.0, 6.0, 3.0],
            vec![1.0, 2.0, 3.0, 7.0],
            vec![8.0, 2.0, 3.0, 1.0],
        ],
    );
    let mut n_past = 4usize;
    let mut pending = None;
    let mut stats_g = Stats::default();
    let mut greedy_sampler = Sampler::new(0.0, 0, 1.0, 0.0, 0);
    let mut state_g = DecodeState {
        eos: 99,
        n_past: &mut n_past,
        pending: &mut pending,
        stats: &mut stats_g,
    };
    let ng = generate_with_backend(&eng_greedy, &mut state_g, 1, 4, &mut greedy_sampler, |_| {
        true
    });
    assert_eq!(ng, 4);
    assert_eq!(stats_g.non_finite_logits, 0);

    // Sampled clean generation
    let eng_sampled = CanaryMockBackend::new_sampled(
        64,
        vec![
            vec![-1.5, 2.0, 3.5, 0.0],
            vec![0.5, -0.0, 4.2, 1.1],
            vec![2.2, 3.3, -4.4, 5.5],
        ],
    );
    let mut n_past_s = 4usize;
    let mut pending_s = None;
    let mut stats_s = Stats::default();
    let mut stoch_sampler = Sampler::new(0.8, 4, 0.95, 30.0, 7);
    let mut state_s = DecodeState {
        eos: 99,
        n_past: &mut n_past_s,
        pending: &mut pending_s,
        stats: &mut stats_s,
    };
    let ns = generate_with_backend(&eng_sampled, &mut state_s, 0, 3, &mut stoch_sampler, |_| {
        true
    });
    assert_eq!(ns, 3);
    assert_eq!(stats_s.non_finite_logits, 0);
}

#[test]
fn engine_try_load_with_math_mode_exists_and_rejects_invalid_gguf() {
    let f = GgufBuilder::new()
        .kv("general.architecture", KvVal::Str("llama".into()))
        .write_temp();
    let path = f.path().to_str().unwrap();

    let err_safe = Engine::try_load_with_math_mode(path, 64, false, MTLMathMode::Safe)
        .err()
        .expect("wrong architecture must fail in Safe math mode");
    assert_eq!(err_safe.kind(), ErrorKind::InvalidData);

    let err_fast = Engine::try_load_with_math_mode(path, 64, false, MTLMathMode::Fast)
        .err()
        .expect("wrong architecture must fail in Fast math mode");
    assert_eq!(err_fast.kind(), ErrorKind::InvalidData);
}

proptest! {
    #[test]
    fn prop_validate_f32_bytes_finite_matches_elementwise_finiteness(
        vals in proptest::collection::vec(prop::num::f32::ANY, 0..64)
    ) {
        let bytes = floats_to_le_bytes(&vals);
        let all_finite = vals.iter().all(|v| v.is_finite());
        let res = validate_f32_bytes_finite("prop.weight", &bytes);
        prop_assert_eq!(res.is_ok(), all_finite);
        if let Err(err) = res {
            prop_assert_eq!(err.kind(), ErrorKind::InvalidData);
        }
    }
}
