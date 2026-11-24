use inkling::engine::{bounds, tier_math, E, EXT, HD, KPS_CHUNK, NH, NKV, N_SLOT};
use proptest::prelude::*;

const REC: usize = 2 * NKV * HD * 2;

// =============================================================================
// Medium-Severity #5: Disk tier (`--kv-disk`) edge cases
// =============================================================================

#[test]
fn tier_piece_zero_clamped_and_plan_read_parts_no_infinite_loop() {
    let clamped = tier_math::clamp_tier_piece_bytes(0);
    assert!(clamped >= (1 << 20), "INK_TIER_PIECE=0 must clamp to >= 1 MiB, got {clamped}");
    assert_eq!(tier_math::clamp_tier_piece_bytes(4), 4 << 20);
    // Overflow protection: huge MiB must not wrap to 0.
    assert!(tier_math::clamp_tier_piece_bytes(usize::MAX) >= (1 << 20));

    // Direct zero piece_bytes passed to plan_read_parts must also clamp >= 1 and terminate.
    let jobs = [(128u64, 16usize, 0x1000usize), (512u64, 0usize, 0x2000usize)];
    let parts = tier_math::plan_read_parts(&jobs, 0);
    assert_eq!(parts.len(), 16);
    assert_eq!(parts.iter().map(|p| p.1).sum::<usize>(), 16);
    assert_eq!(parts[0], (128, 1, 0x1000));
    assert_eq!(parts[15], (143, 1, 0x100f));
}

#[test]
fn tier_threads_zero_clamped_to_at_least_one() {
    assert_eq!(tier_math::clamp_tier_threads(0), 1);
    assert_eq!(tier_math::clamp_tier_threads(1), 1);
    assert_eq!(tier_math::clamp_tier_threads(4), 4);
}

#[test]
fn tier_min_zero_and_n_zero_and_split_target_zero_no_div_by_zero() {
    assert!(tier_math::clamp_tier_min(0) >= 1);
    assert_eq!(tier_math::clamp_tier_min(512), 512);

    // split_kps must never divide by zero or return 0.
    assert_eq!(tier_math::split_kps(0, 0), KPS_CHUNK);
    assert_eq!(tier_math::split_kps(0, 8), KPS_CHUNK);
    assert!(tier_math::split_kps(1024, 0) >= KPS_CHUNK);
    assert!(tier_math::split_kps(1024, usize::MAX) >= KPS_CHUNK);

    // tier_targets(0, ...) must not divide by zero when c == 0.
    let (hs, hf, s) = tier_math::tier_targets(0, 32, 8, None);
    assert!(hs >= 64 && hf >= 64);
    assert!(s / 2 / REC >= tier_math::split_kps(64, 8));

    let (hs0, hf0, s0) = tier_math::tier_targets(0, 0, 0, None);
    assert!(hs0 >= 64 && hf0 >= 64);
    assert!(s0 / 2 / REC >= tier_math::split_kps(64, 0));
}

#[test]
fn tier_targets_above_32k_staging_covers_64_token_checkpoint() {
    // Above 32K context, 32 MiB staging (half_recs = 4096) was previously exceeded by
    // split_kps(pos) between 64-token gen_reserve checkpoints.
    for n in [32_768usize, 32_769, 49_152, 65_536, 131_072] {
        for split_target in [1usize, 4, 8, 16] {
            let (hs, hf, s) = tier_math::tier_targets(n, 32, split_target, None);
            assert!(hs >= 64 && hf >= 64);
            let half_recs = s / 2 / REC;
            let kps_next = tier_math::split_kps(n + 64, split_target);
            assert!(
                half_recs >= kps_next,
                "n={n}, split_target={split_target}: half_recs={half_recs} < split_kps(n+64)={kps_next}"
            );
            for delta in 0..=64 {
                let kps_pos = tier_math::split_kps(n + delta, split_target);
                assert!(
                    half_recs >= kps_pos,
                    "n={n}, delta={delta}: half_recs={half_recs} < split_kps={kps_pos}"
                );
            }
        }
    }
}

#[test]
fn tier_hot_override_clamped_and_gpu_drain_check() {
    for h in [0usize, 1, 2, 16, 63] {
        let (hs, hf, s) = tier_math::tier_targets(4096, 32, 8, Some(h));
        assert!(
            hs >= 64 && hf >= 64,
            "hot_override=Some({h}) must clamp hot ring >= 64, got ({hs}, {hf})"
        );
        assert!(s / 2 / REC >= tier_math::split_kps(4096 + 64, 8));
    }

    let (hs, hf, _) = tier_math::tier_targets(4096, 32, 8, Some(128));
    assert_eq!((hs, hf), (128, 128));

    // GPU drain guard whenever upto > done (prevents CPU tier_write racing with in-flight GPU KV writes).
    assert!(tier_math::tier_needs_gpu_drain(1, 0));
    assert!(tier_math::tier_needs_gpu_drain(100, 68));
    assert!(!tier_math::tier_needs_gpu_drain(0, 0));
    assert!(!tier_math::tier_needs_gpu_drain(64, 64));
    assert!(!tier_math::tier_needs_gpu_drain(32, 64));
}

proptest! {
    #[test]
    fn prop_tier_math_targets_and_clamps_invariants(
        n in 0usize..=131_072,
        n_swa in 0usize..=N_SLOT,
        split_target in 0usize..=32,
        hot_override in prop::option::of(0usize..=2048),
        piece_mib in 0usize..=16,
        threads in 0usize..=16,
        min_n in 0usize..=1024,
    ) {
        let piece_bytes = tier_math::clamp_tier_piece_bytes(piece_mib);
        prop_assert!(piece_bytes >= (1 << 20));
        prop_assert!(tier_math::clamp_tier_threads(threads) >= 1);
        prop_assert!(tier_math::clamp_tier_min(min_n) >= 1);

        let kps = tier_math::split_kps(n, split_target);
        prop_assert!(kps >= KPS_CHUNK);
        prop_assert_eq!(kps % KPS_CHUNK, 0);

        let (hs, hf, s) = tier_math::tier_targets(n, n_swa, split_target, hot_override);
        prop_assert!(hs >= 64);
        prop_assert!(hf >= 64);
        let half_recs = s / 2 / REC;
        prop_assert!(half_recs >= tier_math::split_kps(n + 64, split_target));
    }

    #[test]
    fn prop_tier_pieces_exact_coverage_and_bounds(
        kind in 0usize..=1,
        n_swa in 1usize..N_SLOT,
        max_ctx in 64usize..=16_384,
        slot_seed in 0usize..N_SLOT,
        raw_a in 0usize..=16_384,
        raw_len in 0usize..=2048,
    ) {
        let n_kind = if kind == 0 { n_swa } else { N_SLOT - n_swa };
        let i = slot_seed % n_kind;
        let a = raw_a % max_ctx;
        let b = if kind == 0 {
            // Sliding-window writes never span more than EXT positions at once.
            (a + raw_len.min(EXT)).min(max_ctx)
        } else {
            (a + raw_len).min(max_ctx)
        };

        let pieces = tier_math::tier_pieces(kind, i, a, b, n_swa, max_ctx);
        let total: usize = pieces.iter().map(|&(_, cnt)| cnt).sum();
        prop_assert_eq!(total, b.saturating_sub(a));

        let (lo_bound, hi_bound) = if kind == 0 {
            ((i * EXT * REC) as u64, ((i + 1) * EXT * REC) as u64)
        } else {
            (((n_swa * EXT + i * max_ctx) * REC) as u64, ((n_swa * EXT + (i + 1) * max_ctx) * REC) as u64)
        };

        let mut covered = 0usize;
        for &(off, cnt) in &pieces {
            prop_assert!(cnt > 0);
            let expected_off = tier_math::tier_foff(kind, i, a + covered, n_swa, max_ctx);
            prop_assert_eq!(off, expected_off);
            let end_off = off + (cnt * REC) as u64;
            prop_assert!(off >= lo_bound && end_off <= hi_bound);
            covered += cnt;
        }
        prop_assert_eq!(covered, b.saturating_sub(a));
    }

    #[test]
    fn prop_plan_read_parts_exact_byte_coverage(
        jobs in prop::collection::vec((0u64..1_000_000, 0usize..4096, 0usize..1_000_000), 0..16),
        piece_bytes in 0usize..=2048,
    ) {
        let parts = tier_math::plan_read_parts(&jobs, piece_bytes);
        let eff_piece = piece_bytes.max(1);
        let expected_total: usize = jobs.iter().map(|j| j.1).sum();
        let actual_total: usize = parts.iter().map(|p| p.1).sum();
        prop_assert_eq!(actual_total, expected_total);

        // Verify each non-empty job is covered sequentially with no gaps or overlaps.
        let mut cursor = 0usize;
        for &(off, bytes, dst) in &jobs {
            let mut seen = 0usize;
            while seen < bytes {
                let (poff, plen, pdst) = parts[cursor];
                prop_assert!(plen >= 1 && plen <= eff_piece);
                prop_assert_eq!(poff, off + seen as u64);
                prop_assert_eq!(pdst, dst + seen);
                seen += plen;
                cursor += 1;
            }
            prop_assert_eq!(seen, bytes);
        }
        prop_assert_eq!(cursor, parts.len());
    }

    #[test]
    fn prop_rows_rec_roundtrip(
        nslots in 1usize..=4,
        slot_seed in 0usize..4,
        cap in 1usize..=16,
        idx_seed in 0usize..16,
        seed in any::<u8>(),
    ) {
        let slot = slot_seed % nslots;
        let idx = idx_seed % cap;
        let mut buf = vec![0u8; nslots * 2 * NKV * cap * HD * 2];
        let mut rec = vec![0u8; REC];
        for (k, b) in rec.iter_mut().enumerate() {
            *b = seed.wrapping_add((k as u8).wrapping_mul(37));
        }
        tier_math::rec_to_rows(&rec, &mut buf, slot, cap, idx);
        let mut out = vec![0u8; REC];
        tier_math::rows_to_rec(&buf, slot, cap, idx, &mut out);
        prop_assert_eq!(out, rec);
    }
}

// =============================================================================
// Medium-Severity #2: Buffer bounds validators for Engine methods & kbench
// =============================================================================

#[test]
fn bounds_validate_set_tokens_checks_range_overflow_and_vocab() {
    let max_ctx = 512usize;
    let vocab_rows = 32_000u32;

    // Valid writes up to max_ctx + 8
    assert!(bounds::validate_set_tokens(0, &[0, 1, 31_999], max_ctx, vocab_rows).is_ok());
    assert!(bounds::validate_set_tokens(max_ctx + 7, &[42], max_ctx, vocab_rows).is_ok());
    assert!(bounds::validate_set_tokens(max_ctx + 8, &[], max_ctx, vocab_rows).is_ok());

    // Out-of-bounds range (e.g. prof_pf writing 8192 tokens with -c 512)
    let big = vec![1u32; 8192];
    assert!(bounds::validate_set_tokens(0, &big, max_ctx, vocab_rows).is_err());
    assert!(bounds::validate_set_tokens(max_ctx + 8, &[0], max_ctx, vocab_rows).is_err());
    assert!(bounds::validate_set_tokens(max_ctx + 9, &[], max_ctx, vocab_rows).is_err());

    // Integer overflow
    assert!(bounds::validate_set_tokens(usize::MAX, &[0], max_ctx, vocab_rows).is_err());
    assert!(bounds::validate_set_tokens(0, &[0], usize::MAX, vocab_rows).is_err());

    // Invalid token IDs (>= vocab_rows or > i32::MAX)
    assert!(bounds::validate_set_tokens(0, &[vocab_rows], max_ctx, vocab_rows).is_err());
    assert!(bounds::validate_set_tokens(0, &[0, 1, u32::MAX], max_ctx, vocab_rows).is_err());
    assert!(bounds::validate_set_tokens(0, &[0], max_ctx, 0).is_err());
}

#[test]
fn bounds_validate_token_at_checks_capacity_and_overflow() {
    assert!(bounds::validate_token_at(0, 512).is_ok());
    assert!(bounds::validate_token_at(519, 512).is_ok());
    assert!(bounds::validate_token_at(520, 512).is_err());
    assert!(bounds::validate_token_at(usize::MAX, 512).is_err());
    assert!(bounds::validate_token_at(0, usize::MAX).is_err());
}

#[test]
fn bounds_validate_arena_range_checks_capacity_and_overflow() {
    assert!(bounds::validate_arena_range(0, 1024, 1024).is_ok());
    assert!(bounds::validate_arena_range(1024, 0, 1024).is_ok());
    assert!(bounds::validate_arena_range(1024, 1, 1024).is_err());
    assert!(bounds::validate_arena_range(usize::MAX, 1, 1024).is_err());
    assert!(bounds::validate_arena_range(1, usize::MAX, 1024).is_err());
}

#[test]
fn bounds_validate_restore_checks_cs_and_kv_sizes() {
    assert!(bounds::validate_restore_kv_len(4096, 4096).is_ok());
    assert!(bounds::validate_restore_kv_len(4097, 4096).is_err());
    assert!(bounds::validate_restore_kv_len(2048, 4096).is_err());

    assert!(bounds::validate_restore_range(1024, 1024, 4096, 4096).is_ok());
    assert!(bounds::validate_restore_range(1025, 1024, 4096, 4096).is_err());
    assert!(bounds::validate_restore_range(1024, 1024, 4097, 4096).is_err());
    assert!(bounds::validate_restore_range(512, 1024, 4096, 4096).is_err());
}

#[test]
fn bounds_validate_kbench_t_and_output_scratch_sizing() {
    // Reject t == 0 and t > chunk
    assert!(bounds::validate_kbench_t(0, 4096).is_err());
    assert!(bounds::validate_kbench_t(4097, 4096).is_err());
    assert!(bounds::validate_kbench_t(1, 4096).is_ok());
    assert!(bounds::validate_kbench_t(4096, 4096).is_ok());
    assert!(bounds::validate_kbench_t(1, 0).is_err());

    // Old bug: b.gain_o[0] has cap * E (2304) floats, but kbench writes t * (NH * HD = 3072) floats!
    let cap = 1024usize;
    let out_rows = NH * HD; // 3072
    assert!(
        bounds::validate_kbench_output_buf(cap, out_rows, cap * E).is_err(),
        "b.gain_o[0] sized for cap * 2304 must be rejected for t * 3072 output"
    );
    assert!(bounds::validate_kbench_output_buf(cap, out_rows, cap * out_rows).is_ok());
    assert!(bounds::validate_kbench_output_buf(usize::MAX, out_rows, cap * out_rows).is_err());

    // Scratch layout calculator allocates enough for both input (fp16 or f32) and output(s)
    for half_in in [false, true] {
        let (out_off, out_stride, total_floats) =
            bounds::kbench_scratch_layout(cap, E, out_rows, half_in, 1).expect("valid scratch");
        assert_eq!(out_off % 64, 0);
        assert_eq!(out_stride % 64, 0);
        let min_in_floats = if half_in { (cap * E).div_ceil(2) } else { cap * E };
        assert!(out_off as usize >= min_in_floats);
        assert!(
            bounds::validate_kbench_output_buf(cap, out_rows, total_floats - out_off as usize).is_ok()
        );
    }

    // Two outputs (for kbench_mode: ref + test)
    let (out0, out_stride, total_floats) =
        bounds::kbench_scratch_layout(cap, E, out_rows, true, 2).expect("valid 2-output scratch");
    let out1 = out0 as usize + out_stride;
    assert!(out1 + cap * out_rows <= total_floats);

    // Overflow rejected
    assert!(bounds::kbench_scratch_layout(usize::MAX, E, out_rows, false, 1).is_err());
    assert!(bounds::kbench_scratch_layout(0, E, out_rows, false, 1).is_err());
    assert!(bounds::kbench_scratch_layout(cap, E, out_rows, true, 0).is_err());
}

proptest! {
    #[test]
    fn prop_bounds_validators_never_panic(
        start in any::<usize>(),
        len in 0usize..=64,
        max_ctx in any::<usize>(),
        vocab_rows in any::<u32>(),
        tok_val in any::<u32>(),
        off in any::<usize>(),
        n in any::<usize>(),
        cap in any::<usize>(),
        t in any::<usize>(),
        chunk in any::<usize>(),
    ) {
        let toks = vec![tok_val; len];
        let set_res = bounds::validate_set_tokens(start, &toks, max_ctx, vocab_rows);
        if set_res.is_ok() {
            prop_assert!(max_ctx.checked_add(8).is_some());
            prop_assert!(start.checked_add(len).unwrap() <= max_ctx + 8);
            if len > 0 {
                prop_assert!(tok_val < vocab_rows);
                prop_assert!(tok_val <= i32::MAX as u32);
            }
        }

        let at_res = bounds::validate_token_at(start, max_ctx);
        if at_res.is_ok() {
            prop_assert!(start < max_ctx + 8);
        }

        let ar_res = bounds::validate_arena_range(off, n, cap);
        if ar_res.is_ok() {
            prop_assert!(off.checked_add(n).unwrap() <= cap);
        }

        let kb_res = bounds::validate_kbench_t(t, chunk);
        if kb_res.is_ok() {
            prop_assert!(t >= 1 && t <= chunk);
        }
    }
}
