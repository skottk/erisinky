mod common;

use common::gguf_builder::minimal_tokenizer_gguf;
use inkling::engine::checkpoint::{kv_pack_stride, kv_unpack_stride, validate_restore_meta};
use inkling::engine::LogitsMut;
use inkling::gguf::Gguf;
use inkling::metal::Gpu;
use inkling::sampler::Sampler;
use inkling::session::{step_with_backend, validate_step, DecodeBackend, DecodeState, Stats};
use inkling::tokenizer::Tokenizer;
use objc2_metal::MTLBuffer;
use proptest::prelude::*;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// 1. LogitsMut zero-copy mutation + borrow safety on a real Metal shared buffer
// ---------------------------------------------------------------------------

#[test]
fn logits_mut_modifies_metal_buffer_in_place_and_prevents_aliasing() {
    let gpu = Gpu::new("kernel void k_noop() {}", &["k_noop"]);
    let n_f32 = 512usize;
    let act = RefCell::new(gpu.buffer(n_f32 * std::mem::size_of::<f32>()));

    let raw_ptr = act.borrow().contents().as_ptr() as *mut f32;
    unsafe {
        for i in 0..n_f32 {
            *raw_ptr.add(i) = i as f32;
        }
    }

    let off = 64;
    let len = 128;
    {
        let mut guard = LogitsMut::new(act.borrow_mut(), off, len);
        assert_eq!(guard.len(), len);
        assert_eq!(guard.as_ptr(), unsafe { (raw_ptr as *const f32).add(off) });
        assert_eq!(guard[0], 64.0);
        assert_eq!(guard[len - 1], 191.0);

        // Exclusive borrow must reject both shared and mutable re-borrows.
        assert!(act.try_borrow().is_err());
        assert!(act.try_borrow_mut().is_err());

        // Simulate an in-place grammar mask setting disallowed tokens to -inf.
        for (i, v) in guard.iter_mut().enumerate() {
            if i % 2 == 1 {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Mutations must persist directly in the underlying Metal shared buffer.
    let buf = act.borrow();
    let ptr = buf.contents().as_ptr() as *const f32;
    unsafe {
        assert_eq!(*ptr.add(off), 64.0);
        assert_eq!(*ptr.add(off + 1), f32::NEG_INFINITY);
        assert_eq!(*ptr.add(off + 2), 66.0);
        assert_eq!(*ptr.add(off + len - 1), f32::NEG_INFINITY);
        // Neighboring elements outside [off..off+len) must remain untouched.
        assert_eq!(*ptr.add(off - 1), 63.0);
        assert_eq!(*ptr.add(off + len), 192.0);
    }

    // Out-of-bounds construction must panic cleanly.
    let oob = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = LogitsMut::new(act.borrow_mut(), n_f32 - 8, 16);
    }));
    assert!(oob.is_err(), "out-of-bounds LogitsMut::new must panic");
}

// ---------------------------------------------------------------------------
// 2. Hybrid KV checkpoint pack / unpack + branching & error handling
// ---------------------------------------------------------------------------

fn make_strided_kv(n_blocks: usize, cap: usize, row_bytes: usize, active_len: usize, seed: u8) -> Vec<u8> {
    let mut buf = vec![0xAAu8; n_blocks * cap * row_bytes];
    for blk in 0..n_blocks {
        for pos in 0..active_len {
            let base = (blk * cap + pos) * row_bytes;
            for b in 0..row_bytes {
                buf[base + b] = seed
                    .wrapping_add(blk as u8)
                    .wrapping_mul(31)
                    .wrapping_add(pos as u8)
                    .wrapping_add(b as u8);
            }
        }
    }
    buf
}

#[test]
fn kv_pack_unpack_empty_active_len_is_noop() {
    let src = vec![0x55u8; 4 * 64 * 16];
    let packed = kv_pack_stride(0, 64, 4, 16, &src).expect("pack empty");
    assert!(packed.is_empty());

    let mut dst = vec![0xCCu8; 4 * 128 * 16];
    kv_unpack_stride(0, 128, 4, 16, &packed, &mut dst).expect("unpack empty");
    assert!(dst.iter().all(|&b| b == 0xCC));
}

#[test]
fn kv_pack_unpack_survives_capacity_growth_and_branching() {
    let n_blocks = 8;
    let row_bytes = 32;

    // Checkpoint A at n_past = 10, cap = 16.
    let buf_a = make_strided_kv(n_blocks, 16, row_bytes, 10, 1);
    let ckpt_a = kv_pack_stride(10, 16, n_blocks, row_bytes, &buf_a).expect("pack A");

    // Advance on Branch 1 with a grown capacity (cap = 32, n_past = 20) -> Checkpoint B.
    let mut live = vec![0x00u8; n_blocks * 32 * row_bytes];
    kv_unpack_stride(10, 32, n_blocks, row_bytes, &ckpt_a, &mut live).expect("unpack A into cap 32");
    for blk in 0..n_blocks {
        for pos in 10..20 {
            let base = (blk * 32 + pos) * row_bytes;
            live[base..base + row_bytes].fill(0x11u8.wrapping_add(blk as u8).wrapping_add(pos as u8));
        }
    }
    let ckpt_b = kv_pack_stride(20, 32, n_blocks, row_bytes, &live).expect("pack B");

    // Roll back to A and overwrite positions 10..20 with Branch 2 data.
    kv_unpack_stride(10, 32, n_blocks, row_bytes, &ckpt_a, &mut live).expect("restore A");
    for blk in 0..n_blocks {
        for pos in 0..10 {
            let a_base = (blk * 16 + pos) * row_bytes;
            let l_base = (blk * 32 + pos) * row_bytes;
            assert_eq!(
                &live[l_base..l_base + row_bytes],
                &buf_a[a_base..a_base + row_bytes],
                "branch A prefix mismatch at blk={blk} pos={pos}"
            );
        }
        for pos in 10..20 {
            let base = (blk * 32 + pos) * row_bytes;
            live[base..base + row_bytes].fill(0xEE);
        }
    }

    // Restore Branch B into an even larger capacity (cap = 64) and verify bit-exact recovery.
    let mut grown = vec![0x00u8; n_blocks * 64 * row_bytes];
    kv_unpack_stride(20, 64, n_blocks, row_bytes, &ckpt_b, &mut grown).expect("restore B into cap 64");
    for blk in 0..n_blocks {
        for pos in 0..10 {
            let a_base = (blk * 16 + pos) * row_bytes;
            let g_base = (blk * 64 + pos) * row_bytes;
            assert_eq!(&grown[g_base..g_base + row_bytes], &buf_a[a_base..a_base + row_bytes]);
        }
        for pos in 10..20 {
            let g_base = (blk * 64 + pos) * row_bytes;
            let expected = 0x11u8.wrapping_add(blk as u8).wrapping_add(pos as u8);
            assert!(grown[g_base..g_base + row_bytes].iter().all(|&b| b == expected));
        }
    }
}

#[test]
fn kv_pack_unpack_and_meta_reject_invalid_inputs() {
    assert!(kv_pack_stride(17, 16, 2, 8, &vec![0u8; 2 * 16 * 8]).is_err());
    assert!(kv_pack_stride(4, 16, 2, 8, &vec![0u8; 16]).is_err());
    assert!(kv_pack_stride(1, usize::MAX, 2, 8, &[]).is_err());

    let mut dst = vec![0u8; 2 * 16 * 8];
    assert!(kv_unpack_stride(17, 16, 2, 8, &[], &mut dst).is_err());
    assert!(kv_unpack_stride(4, 16, 2, 8, &vec![0u8; 3], &mut dst).is_err());
    assert!(kv_unpack_stride(4, 16, 2, 8, &vec![0u8; 2 * 4 * 8], &mut [0u8; 10]).is_err());

    assert!(validate_restore_meta(10, 64, false, false, 128, 128, 10).is_ok());
    assert!(validate_restore_meta(10, 64, true, false, 128, 128, 10).is_err());
    assert!(validate_restore_meta(64, 64, false, false, 128, 128, 64).is_err());
    assert!(validate_restore_meta(10, 64, false, false, 120, 128, 10).is_err());
    assert!(validate_restore_meta(10, 64, false, false, 128, 128, 9).is_err());
}

proptest! {
    #[test]
    fn proptest_kv_pack_unpack_arbitrary_strides(
        n_blocks in 1usize..=16,
        row_bytes in prop_oneof![Just(8usize), Just(16usize), Just(32usize)],
        active_len in 0usize..=24,
        src_slack in 0usize..=16,
        dst_slack in 0usize..=24,
        seed in any::<u8>(),
    ) {
        let cap_src = active_len + src_slack;
        let cap_dst = active_len + dst_slack;
        let src = make_strided_kv(n_blocks, cap_src, row_bytes, active_len, seed);
        let packed = kv_pack_stride(active_len, cap_src, n_blocks, row_bytes, &src).unwrap();
        prop_assert_eq!(packed.len(), n_blocks * active_len * row_bytes);

        let mut dst = vec![0x77u8; n_blocks * cap_dst * row_bytes];
        kv_unpack_stride(active_len, cap_dst, n_blocks, row_bytes, &packed, &mut dst).unwrap();

        for blk in 0..n_blocks {
            for pos in 0..active_len {
                let s0 = (blk * cap_src + pos) * row_bytes;
                let d0 = (blk * cap_dst + pos) * row_bytes;
                prop_assert_eq!(&dst[d0..d0 + row_bytes], &src[s0..s0 + row_bytes]);
            }
            for pos in active_len..cap_dst {
                let d0 = (blk * cap_dst + pos) * row_bytes;
                prop_assert!(dst[d0..d0 + row_bytes].iter().all(|&b| b == 0x77));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Single-step decode backend validation & state transitions
// ---------------------------------------------------------------------------

struct MockStepBackend {
    max_ctx: usize,
    log: RefCell<Vec<String>>,
}

impl MockStepBackend {
    fn new(max_ctx: usize) -> Self {
        Self {
            max_ctx,
            log: RefCell::new(Vec::new()),
        }
    }
}

impl DecodeBackend for MockStepBackend {
    type Step = usize;
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn release_batch(&self) {
        self.log.borrow_mut().push("release_batch".into());
    }
    fn sample_logits(&self, _sampler: &mut Sampler) -> u32 {
        0
    }
    fn set_token(&self, pos: usize, tok: u32) {
        self.log.borrow_mut().push(format!("set_token({pos},{tok})"));
    }
    fn gen_reserve(&self, pos: usize) {
        self.log.borrow_mut().push(format!("gen_reserve({pos})"));
    }
    fn gen_capacity(&self) -> usize {
        self.max_ctx
    }
    fn launch_step(&self, pos: usize, argmax: bool) -> usize {
        self.log.borrow_mut().push(format!("launch_step({pos},{argmax})"));
        pos
    }
    fn wait_step(&self, step: &usize) {
        self.log.borrow_mut().push(format!("wait_step({step})"));
    }
    fn token_at(&self, _pos: usize) -> u32 {
        0
    }
}

#[test]
fn step_with_backend_advances_cursor_clears_pending_and_respects_max_ctx() {
    let eng = MockStepBackend::new(4);
    let mut n_past = 2usize;
    let mut pending = Some(200006u32);
    let mut stats = Stats::default();
    let mut state = DecodeState {
        eos: 200006,
        n_past: &mut n_past,
        pending: &mut pending,
        stats: &mut stats,
    };

    step_with_backend(&eng, &mut state, 42).expect("step at pos 2 fits in max_ctx=4");
    assert_eq!(*state.n_past, 3);
    assert_eq!(*state.pending, None);
    assert_eq!(state.stats.gen_tokens, 1);
    assert_eq!(
        *eng.log.borrow(),
        vec![
            "release_batch",
            "set_token(2,42)",
            "gen_reserve(2)",
            "launch_step(2,false)",
            "wait_step(2)",
        ]
    );

    // Next step would reach end == max_ctx (4), which must fail without touching GPU or cursor.
    eng.log.borrow_mut().clear();
    *state.pending = Some(99);
    let err = step_with_backend(&eng, &mut state, 43).unwrap_err();
    assert!(err.contains("context full"));
    assert_eq!(*state.n_past, 3);
    assert_eq!(*state.pending, Some(99));
    assert!(eng.log.borrow().is_empty());

    assert!(validate_step(usize::MAX, 16).is_err());
}

// ---------------------------------------------------------------------------
// 4. Tokenizer metadata accessors (vocab_size, specials)
// ---------------------------------------------------------------------------

#[test]
fn tokenizer_exposes_vocab_size_and_sorted_specials() {
    let raw = minimal_tokenizer_gguf(
        &[
            ("<|content_tool_error|>", 3),
            ("<|content_invoke_tool_json|>", 3),
            ("hello", 1),
        ],
        &["h e", "he l", "hel l", "hell o"],
    );
    let g = Gguf::from_bytes(&raw).expect("parse synthetic gguf");
    let tk = Tokenizer::load(&g);

    assert!(tk.vocab_size() >= 256 + 3);
    let specials = tk.specials();
    assert!(specials.iter().any(|(s, _)| s == "<|content_tool_error|>"));
    assert!(specials.iter().any(|(s, _)| s == "<|content_invoke_tool_json|>"));
    // Sorted longest-first for greedy prefix matching.
    assert!(specials.windows(2).all(|w| w[0].0.len() >= w[1].0.len()));
    for (s, id) in specials {
        assert_eq!(tk.special_id(s), Some(*id));
        assert!(tk.is_special(*id));
    }
}
