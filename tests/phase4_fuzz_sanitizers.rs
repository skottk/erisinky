//! Phase 4: Miri, ASan, TSan, unwind-safety, and fuzz-target invariant tests.

mod common;

use common::gguf_builder::{minimal_tokenizer_gguf, GgufBuilder, KvVal};
use inkling::engine::{tier_math, HD, NKV};
use inkling::gguf::{Gguf, T_F16, T_F32, T_Q4_K};
use inkling::sampler::Sampler;
use inkling::session::{generate_with_backend, DecodeBackend, DecodeState, Renderer, Stats};
use inkling::tokenizer::Tokenizer;
use std::cell::{Cell, RefCell};
use std::os::unix::fs::FileExt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const REC: usize = 2 * NKV * HD * 2;

fn test_tokenizer() -> Tokenizer {
    let raw = minimal_tokenizer_gguf(
        &[("he", 1), ("ll", 1), ("hell", 1), ("hello", 1)],
        &["h e", "l l", "he ll", "hell o"],
    );
    let g = Gguf::from_bytes(&raw).expect("parse minimal tokenizer gguf");
    Tokenizer::load(&g)
}

#[test]
fn tokenizer_decode_bytes_and_renderer_handle_oob_token_ids() {
    let tk = test_tokenizer();
    // Out-of-bounds token IDs (e.g., padded vocab slot or corrupted/fuzzed ID) must never panic.
    assert!(tk.decode_bytes(u32::MAX).is_empty());
    assert!(tk.decode_bytes(10_000).is_empty());
    assert!(!tk.is_special(u32::MAX));

    for (show_thinking, raw) in [(false, false), (true, false), (false, true)] {
        let mut r = Renderer::new(show_thinking, raw);
        let mut out = Vec::new();
        r.push(&tk, b'A' as u32, &mut out);
        r.push(&tk, u32::MAX, &mut out);
        r.push(&tk, 999_999, &mut out);
        r.push(&tk, b'Z' as u32, &mut out);
        r.end(&mut out);
        assert_eq!(std::str::from_utf8(&out).unwrap(), "AZ");
    }
}

#[test]
fn renderer_end_resets_thinking_even_when_show_thinking_is_false() {
    let tk = test_tokenizer();
    let think_id = tk.special_id("<|content_thinking|>").expect("thinking special");
    let mut r = Renderer::new(false, false);
    let mut out = Vec::new();
    r.push(&tk, think_id, &mut out);
    assert!(r.thinking);
    r.end(&mut out);
    assert!(!r.thinking, "Renderer::end() must reset thinking when show_thinking=false");
}

/// Tracked in-flight step handle used to verify that `generate_with_backend` drops
/// (drains) every launched step when `on_token` panics and unwinds.
struct TrackedStep {
    id: usize,
    dropped: Arc<AtomicUsize>,
}

impl Drop for TrackedStep {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct UnwindBackend {
    max_ctx: usize,
    launched: Cell<usize>,
    dropped: Arc<AtomicUsize>,
    tokens: RefCell<Vec<u32>>,
    logits: Vec<f32>,
}

impl UnwindBackend {
    fn new(max_ctx: usize) -> Self {
        Self {
            max_ctx,
            launched: Cell::new(0),
            dropped: Arc::new(AtomicUsize::new(0)),
            tokens: RefCell::new(vec![0; max_ctx + 8]),
            logits: vec![0.1, 5.0, 0.2, 0.3],
        }
    }
}

impl DecodeBackend for UnwindBackend {
    type Step = TrackedStep;
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn release_batch(&self) {}
    fn sample_logits(&self, sampler: &mut Sampler) -> u32 {
        sampler.sample(&self.logits)
    }
    fn set_token(&self, pos: usize, tok: u32) {
        self.tokens.borrow_mut()[pos] = tok;
    }
    fn gen_reserve(&self, _pos: usize) {}
    fn gen_capacity(&self) -> usize {
        self.max_ctx
    }
    fn launch_step(&self, pos: usize, argmax: bool) -> TrackedStep {
        let id = self.launched.get();
        self.launched.set(id + 1);
        if argmax {
            self.tokens.borrow_mut()[pos + 1] = 1;
        }
        TrackedStep {
            id,
            dropped: self.dropped.clone(),
        }
    }
    fn wait_step(&self, step: &TrackedStep) {
        let _ = step.id;
    }
    fn token_at(&self, pos: usize) -> u32 {
        self.tokens.borrow()[pos]
    }
}

#[test]
fn generate_with_backend_unwind_drops_all_inflight_steps() {
    // Greedy pipelined path (temp == 0.0) keeps 2 steps in flight; panicking inside on_token
    // must unwind cleanly and drop every launched step handle (which calls waitUntilCompleted on GPU).
    for temp in [0.0f32, 0.8f32] {
        let eng = UnwindBackend::new(64);
        let mut n_past = 4usize;
        let mut pending = None;
        let mut stats = Stats::default();
        let mut sampler = Sampler::new(temp, 4, 0.9, 0.0, 42);

        let res = catch_unwind(AssertUnwindSafe(|| {
            let mut state = DecodeState {
                eos: 99,
                n_past: &mut n_past,
                pending: &mut pending,
                stats: &mut stats,
            };
            let mut seen = 0usize;
            generate_with_backend(&eng, &mut state, 1, 16, &mut sampler, |_tok| {
                seen += 1;
                if seen == 3 {
                    panic!("simulated on_token unwind at step 3 (temp={temp})");
                }
                true
            });
        }));
        assert!(res.is_err(), "expected unwind for temp={temp}");
        assert!(eng.launched.get() > 0);
        assert_eq!(
            eng.dropped.load(Ordering::SeqCst),
            eng.launched.get(),
            "every launched step must be dropped on unwind (temp={temp})"
        );
    }
}

/// Pure-Rust (Miri-compatible) round-trip of `rows_to_rec` / `rec_to_rows` across multiple slots.
#[test]
fn miri_tier_math_rows_rec_roundtrip_and_plan_read_parts() {
    let n_slots = 2usize;
    let cap = 4usize;
    let total_bytes = n_slots * 2 * NKV * cap * HD * 2;
    let mut src = vec![0u8; total_bytes];
    for (idx, b) in src.iter_mut().enumerate() {
        *b = ((idx.wrapping_mul(37) ^ (idx >> 5)) & 0xFF) as u8;
    }

    let mut dst = vec![0u8; total_bytes];
    let mut rec = vec![0u8; REC];
    for slot in 0..n_slots {
        for pos in 0..cap {
            tier_math::rows_to_rec(&src, slot, cap, pos, &mut rec);
            tier_math::rec_to_rows(&rec, &mut dst, slot, cap, pos);
        }
    }
    assert_eq!(src, dst);

    let jobs = [(0u64, REC * 3, 0x1000usize), (REC as u64 * 8, REC * 2, 0x9000usize)];
    let parts = tier_math::plan_read_parts(&jobs, REC);
    assert_eq!(parts.len(), 5);
    assert_eq!(parts.iter().map(|p| p.1).sum::<usize>(), REC * 5);
}

/// Pure-Rust (Miri + TSan compatible) multi-threaded GGUF parse & weight finiteness audit.
#[test]
fn concurrent_gguf_parse_and_weight_audit_clean() {
    let f32_bytes: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32) * 0.25).to_le_bytes())
        .collect();
    let f16_bytes: Vec<u8> = (0..64)
        .flat_map(|_| 0x3C00u16.to_le_bytes())
        .collect();
    let mut q4k_bytes = vec![0u8; 144 * 2];
    q4k_bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    q4k_bytes[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
    q4k_bytes[144..146].copy_from_slice(&0x3C00u16.to_le_bytes());
    q4k_bytes[146..148].copy_from_slice(&0x3800u16.to_le_bytes());

    let mut b = GgufBuilder::new().kv("general.architecture", KvVal::Str("inkling2_loop".into()));
    for i in 0..6 {
        b = b
            .tensor(format!("t_f32_{i}"), vec![64], T_F32, f32_bytes.clone())
            .tensor(format!("t_f16_{i}"), vec![64], T_F16, f16_bytes.clone())
            .tensor(format!("t_q4k_{i}"), vec![512], T_Q4_K, q4k_bytes.clone());
    }
    let raw = b.build();
    let g = Gguf::from_bytes(&raw).expect("parse valid gguf");
    let report = g.audit_weights_finiteness();
    assert!(report.is_clean());
    assert_eq!(report.tensors_checked, 18);
    assert_eq!(report.blocks_checked, 6 * (64 + 64 + 2));
}

/// TSan-targeted stress test mirroring `Engine::tier_read`: multiple worker threads
/// concurrently `read_exact_at` disjoint slices of a backing file into disjoint parts
/// of a shared staging buffer via `tier_math::plan_read_parts`.
#[test]
fn tsan_parallel_tier_read_disjoint_slices() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("kv_tier_tsan.bin");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create temp tier file");

    let n_recs = 32usize;
    let total = n_recs * REC;
    let payload: Vec<u8> = (0..total)
        .map(|i| ((i.wrapping_mul(131) ^ (i >> 7)) & 0xFF) as u8)
        .collect();
    file.write_all_at(&payload, 0).expect("write payload");

    let mut staging = vec![0u8; total];
    let base_addr = staging.as_mut_ptr() as usize;
    let jobs = [
        (0u64, 10 * REC, base_addr),
        ((10 * REC) as u64, 12 * REC, base_addr + 10 * REC),
        ((22 * REC) as u64, 10 * REC, base_addr + 22 * REC),
    ];
    let parts = tier_math::plan_read_parts(&jobs, REC * 2);
    assert_eq!(parts.iter().map(|p| p.1).sum::<usize>(), total);

    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..4 {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= parts.len() {
                    break;
                }
                let (off, len, dst) = parts[k];
                // SAFETY: `plan_read_parts` partitions `jobs` into disjoint sub-slices within
                // `staging[..total]`, and `staging` outlives this `thread::scope`.
                let sl = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
                file.read_exact_at(sl, off).expect("pread piece");
            });
        }
    });
    assert_eq!(staging, payload);
}
