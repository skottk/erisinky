//! GPU-vs-CPU-reference tests for High-Severity #1:
//! Silent prefill corruption on 1-token batches (`t == 1`) and `INK_OLD_MIXB=1`
//! in `MixJob` / `dispatch_mix_jobs` / `k_mix` / `k_mix_b_*`.

use inkling::engine::{dispatch_mix_jobs, MixJob, E};
use inkling::metal::{Enc, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use std::sync::{Mutex, OnceLock};

const SENTINEL_F32_BITS: u32 = 0x7FC0_DEAD; // quiet NaN
const SENTINEL_F16_BITS: u16 = 0x7EAD; // quiet NaN in IEEE-754 binary16

fn with_gpu<R>(f: impl FnOnce(&Gpu) -> R) -> R {
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let guard = GPU
        .get_or_init(|| {
            let src = format!(
                "{}\n{}",
                include_str!("../src/kernels.metal"),
                include_str!("../src/prefill.metal")
            );
            Mutex::new(Gpu::new(&src, &[]))
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    f(&guard)
}

/// Deterministic xorshift64* PRNG producing values in `[-1.5, 1.5]` with varied scale.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * 1.5
    }
}

/// Exact IEEE-754 binary16 -> f32 conversion.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let val = (mant as f32) * 2.0f32.powi(-24);
            if sign != 0 { -val } else { val }
        }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits(sign | 0x7F80_0000)
        } else {
            f32::from_bits(sign | 0x7FC0_0000 | (mant << 13))
        }
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Exact IEEE-754 f32 -> binary16 (round-to-nearest-even).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        return if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00 | ((mant >> 13) as u16)
        };
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow to inf
    }
    if unbiased < -24 {
        return sign; // underflow to zero
    }
    if unbiased < -14 {
        // subnormal binary16
        let full_mant = mant | 0x0080_0000;
        let shift = (-14 - unbiased + 13) as u32;
        let half_bit = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let rem = full_mant & mask;
        let mut hmant = full_mant >> shift;
        if rem > half_bit || (rem == half_bit && (hmant & 1) != 0) {
            hmant += 1;
        }
        return sign | (hmant as u16);
    }

    let hexp = (unbiased + 15) as u32;
    let rem = mant & 0x1FFF;
    let mut hmant = mant >> 13;
    if rem > 0x1000 || (rem == 0x1000 && (hmant & 1) != 0) {
        hmant += 1;
        if hmant == 0x400 {
            let new_exp = hexp + 1;
            if new_exp >= 31 {
                return sign | 0x7C00;
            }
            return sign | ((new_exp as u16) << 10);
        }
    }
    sign | ((hexp as u16) << 10) | (hmant as u16)
}

#[test]
fn f16_codec_self_check() {
    for &x in &[0.0f32, -0.0, 1.0, -1.0, 0.5, -2.25, 65504.0, 6.1035156e-5, 5.9604645e-8] {
        let h = f32_to_f16(x);
        let back = f16_to_f32(h);
        assert_eq!(back.to_bits(), x.to_bits(), "roundtrip failed for {x}");
    }
    assert!(f16_to_f32(SENTINEL_F16_BITS).is_nan());
    assert!(f32::from_bits(SENTINEL_F32_BITS).is_nan());
}

#[test]
fn mixjob_abi_layout() {
    assert_eq!(std::mem::size_of::<MixJob>(), 200);
    assert_eq!(std::mem::align_of::<MixJob>(), 4);
}

fn empty_job() -> MixJob {
    MixJob {
        q_off: -1,
        norm_off: [-1; 4],
        eps: 1e-6,
        ncopy: [[-1; 3]; 4],
        hcopy: [[-1; 3]; 4],
        hout: [-1; 4],
        hstride: (2 * E) as u32,
        ..Default::default()
    }
}

/// CPU f64 reference for a slice of `MixJob`s operating on `pb` (bytes) and `arena` (f32 words).
/// Mutates `arena` in-place identically to the GPU kernels (`k_mix` / `k_mix_b_*`).
fn cpu_ref_mix_jobs(pb: &[u8], arena: &mut [u32], jobs: &[MixJob], t: usize) {
    let read_pb_f32 = |byte_off: usize, idx: usize| -> f64 {
        let off = byte_off + idx * 4;
        f32::from_le_bytes(pb[off..off + 4].try_into().unwrap()) as f64
    };

    // Snapshot input arena because multiple independent jobs read before barrier.
    let src_arena = arena.to_vec();

    for job in jobs {
        let n_s = job.n_s as usize;
        let n_out = if job.q_off < 0 { 1 } else { job.n_q as usize };

        for tok in 0..t {
            let toff = tok * job.tstride as usize;
            let mut s = vec![vec![0.0f64; E]; n_s];
            for (i, s_i) in s.iter_mut().enumerate() {
                let base = job.s_off[i] as usize + toff;
                for (e, dst) in s_i.iter_mut().enumerate() {
                    *dst = f32::from_bits(src_arena[base + e]) as f64;
                }
            }

            let mut o = vec![vec![0.0f64; E]; n_out];
            if job.q_off < 0 {
                o[0].copy_from_slice(&s[0]);
            } else {
                let q_base = job.q_off as usize;
                let mut rr = vec![0.0f64; n_s];
                for (i, s_i) in s.iter().enumerate() {
                    let ss: f64 = s_i.iter().map(|&v| v * v).sum();
                    rr[i] = 1.0 / (ss / (E as f64) + 1e-6).sqrt();
                }
                for (q, o_q) in o.iter_mut().enumerate() {
                    let mut sc = vec![0.0f64; n_s];
                    let mut mx = f64::NEG_INFINITY;
                    for (i, s_i) in s.iter().enumerate() {
                        let mut dot = 0.0f64;
                        for (e, &sv) in s_i.iter().enumerate() {
                            let qv = read_pb_f32(q_base, q * E + e);
                            dot += sv * qv;
                        }
                        sc[i] = rr[i] * dot / 48.0;
                        if sc[i] > mx {
                            mx = sc[i];
                        }
                    }
                    let mut den = 0.0f64;
                    for sc_i in &mut sc {
                        *sc_i = (*sc_i - mx).exp();
                        den += *sc_i;
                    }
                    for (i, s_i) in s.iter().enumerate() {
                        let w = sc[i] / den * rr[i];
                        for (dst, &sv) in o_q.iter_mut().zip(s_i.iter()) {
                            *dst += w * sv;
                        }
                    }
                }
            }

            for (q, o_q) in o.iter().enumerate() {
                let ss_o: f64 = o_q.iter().map(|&v| v * v).sum();
                let rn = 1.0 / (ss_o / (E as f64) + 1e-6).sqrt();

                for c in 0..3 {
                    if job.ncopy[q][c] >= 0 {
                        let base = job.ncopy[q][c] as usize + toff;
                        for (e, &ov) in o_q.iter().enumerate() {
                            arena[base + e] = ((ov * rn) as f32).to_bits();
                        }
                    }
                    if job.hcopy[q][c] >= 0 {
                        let hbase_halves = job.hcopy[q][c] as usize * 2 + tok * job.hstride as usize;
                        for (e, &ov) in o_q.iter().enumerate() {
                            let h = f32_to_f16((ov * rn) as f32);
                            write_half_in_u32_slice(arena, hbase_halves + e, h);
                        }
                    }
                }

                let mut out_v = vec![0.0f64; E];
                if job.norm_off[q] >= 0 {
                    let rw = 1.0 / (ss_o / (E as f64) + job.eps as f64).sqrt();
                    let w_base = job.norm_off[q] as usize;
                    for (e, (dst, &ov)) in out_v.iter_mut().zip(o_q.iter()).enumerate() {
                        *dst = ov * rw * read_pb_f32(w_base, e);
                    }
                } else {
                    out_v.copy_from_slice(o_q);
                }

                if job.hout[q] >= 0 {
                    let hbase_halves = job.hout[q] as usize * 2 + tok * E;
                    for (e, &v) in out_v.iter().enumerate() {
                        let h = f32_to_f16(v as f32);
                        write_half_in_u32_slice(arena, hbase_halves + e, h);
                    }
                }

                if ((job.nof32 >> q) & 1) == 0 {
                    let base = job.out_off[q] as usize + toff;
                    for (e, &v) in out_v.iter().enumerate() {
                        arena[base + e] = (v as f32).to_bits();
                    }
                }
            }
        }
    }
}

fn write_half_in_u32_slice(words: &mut [u32], half_idx: usize, val: u16) {
    let w = &mut words[half_idx / 2];
    if half_idx.is_multiple_of(2) {
        *w = (*w & 0xFFFF_0000) | (val as u32);
    } else {
        *w = (*w & 0x0000_FFFF) | ((val as u32) << 16);
    }
}

fn read_half_in_u32_slice(words: &[u32], half_idx: usize) -> u16 {
    let w = words[half_idx / 2];
    if half_idx.is_multiple_of(2) {
        (w & 0xFFFF) as u16
    } else {
        (w >> 16) as u16
    }
}

/// Simple bump allocator for synthetic PB (256-byte aligned) and Arena (64-float aligned).
struct Fixture {
    pb: Vec<u8>,
    arena: Vec<u32>,
}

impl Fixture {
    fn new() -> Self {
        Self { pb: Vec::new(), arena: Vec::new() }
    }

    fn push_pb_f32(&mut self, data: &[f32]) -> i32 {
        while !self.pb.len().is_multiple_of(256) {
            self.pb.push(0);
        }
        let off = self.pb.len() as i32;
        for &v in data {
            self.pb.extend_from_slice(&v.to_le_bytes());
        }
        off
    }

    fn alloc_f32_sentinel(&mut self, n_floats: usize) -> u32 {
        while !self.arena.len().is_multiple_of(64) {
            self.arena.push(SENTINEL_F32_BITS);
        }
        let off = self.arena.len() as u32;
        self.arena.resize(self.arena.len() + n_floats, SENTINEL_F32_BITS);
        off
    }

    fn alloc_f16_sentinel(&mut self, n_halves: usize) -> u32 {
        assert_eq!(n_halves % 2, 0);
        while !self.arena.len().is_multiple_of(64) {
            self.arena.push(SENTINEL_F32_BITS);
        }
        let off = self.arena.len() as u32;
        let packed = (SENTINEL_F16_BITS as u32) | ((SENTINEL_F16_BITS as u32) << 16);
        self.arena.resize(self.arena.len() + n_halves / 2, packed);
        off
    }

    fn push_f32_slice(&mut self, data: &[f32]) -> u32 {
        let off = self.alloc_f32_sentinel(data.len());
        for (i, &v) in data.iter().enumerate() {
            self.arena[off as usize + i] = v.to_bits();
        }
        off
    }

    /// Execute `dispatch_mix_jobs` on the GPU and return the resulting arena words.
    fn run_gpu(&self, jobs: &[MixJob], t: usize, old_mixb: bool) -> Vec<u32> {
        with_gpu(|g| {
            let pb_bytes = if self.pb.is_empty() { vec![0u8; 256] } else { self.pb.clone() };
            let pb_buf = g.buffer_with(&pb_bytes);
            let arena_bytes: Vec<u8> = self.arena.iter().flat_map(|w| w.to_le_bytes()).collect();
            let act_buf = g.buffer_with(&arena_bytes);

            let cb = g.cmd();
            let enc = Enc::new(&cb);
            enc.buf(&pb_buf, 0, 1);
            enc.buf(&act_buf, 0, 2);
            dispatch_mix_jobs(g, &enc, jobs, t, old_mixb);
            enc.end();
            cb.commit();
            cb.waitUntilCompleted();

            assert_eq!(
                cb.status(),
                objc2_metal::MTLCommandBufferStatus::Completed,
                "Metal command buffer did not complete cleanly: {:?}",
                cb.error()
            );

            let mut out = vec![0u32; self.arena.len()];
            // SAFETY: `act_buf` is StorageModeShared, was allocated with `arena_bytes.len() == self.arena.len() * 4` bytes,
            // and `cb.waitUntilCompleted()` guarantees the GPU has finished all writes before CPU reads.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    act_buf.contents().as_ptr() as *const u32,
                    out.as_mut_ptr(),
                    out.len(),
                );
            }
            out
        })
    }
}

fn assert_f32_slice_matches(label: &str, gpu: &[u32], cpu: &[u32], off: u32, len: usize) {
    let base = off as usize;
    for i in 0..len {
        let gb = gpu[base + i];
        let cb = cpu[base + i];
        assert_ne!(
            gb, SENTINEL_F32_BITS,
            "{label}: element {i} (offset {}) was NOT written (still sentinel NaN 0x7FC0DEAD)",
            base + i
        );
        let gv = f32::from_bits(gb);
        let cv = f32::from_bits(cb);
        assert!(gv.is_finite(), "{label}: element {i} is non-finite ({gv})");
        let tol = 2e-4 + 2e-4 * cv.abs();
        assert!(
            (gv - cv).abs() <= tol,
            "{label}: f32 mismatch at {i}: gpu={gv} cpu={cv} diff={} tol={tol}",
            (gv - cv).abs()
        );
    }
}

fn assert_f32_slice_untouched(label: &str, gpu: &[u32], off: u32, len: usize) {
    let base = off as usize;
    for i in 0..len {
        assert_eq!(
            gpu[base + i],
            SENTINEL_F32_BITS,
            "{label}: suppressed f32 output (nof32) was clobbered at {i}: got 0x{:08x}",
            gpu[base + i]
        );
    }
}

fn assert_f16_region_matches(
    label: &str,
    gpu: &[u32],
    cpu: &[u32],
    off_floats: u32,
    t: usize,
    stride_halves: usize,
) {
    for tok in 0..t {
        let base_h = off_floats as usize * 2 + tok * stride_halves;
        for e in 0..E {
            let gh = read_half_in_u32_slice(gpu, base_h + e);
            let ch = read_half_in_u32_slice(cpu, base_h + e);
            assert_ne!(
                gh, SENTINEL_F16_BITS,
                "{label}: tok={tok} e={e} was NOT written (still sentinel f16 NaN 0x7EAD)"
            );
            let gv = f16_to_f32(gh);
            let cv = f16_to_f32(ch);
            assert!(gv.is_finite(), "{label}: tok={tok} e={e} non-finite ({gv}, bits=0x{gh:04x})");
            let tol = 2e-3 + 2e-3 * cv.abs();
            assert!(
                (gv - cv).abs() <= tol,
                "{label}: f16 mismatch at tok={tok} e={e}: gpu={gv} (0x{gh:04x}) cpu={cv} (0x{ch:04x}) diff={} tol={tol}",
                (gv - cv).abs()
            );
        }
    }
}

fn assert_f16_region_untouched(
    label: &str,
    gpu: &[u32],
    off_floats: u32,
    t: usize,
    stride_halves: usize,
) {
    for tok in 0..t {
        let base_h = off_floats as usize * 2 + tok * stride_halves;
        for e in 0..E {
            let gh = read_half_in_u32_slice(gpu, base_h + e);
            assert_eq!(
                gh, SENTINEL_F16_BITS,
                "{label}: guard fp16 region clobbered at tok={tok} e={e}: got 0x{gh:04x}"
            );
        }
    }
}

fn verify_job_outputs(label: &str, job: &MixJob, t: usize, gpu: &[u32], cpu: &[u32]) {
    let n_out = if job.q_off < 0 { 1 } else { job.n_q as usize };
    for q in 0..n_out {
        for c in 0..3 {
            if job.ncopy[q][c] >= 0 {
                for tok in 0..t {
                    assert_f32_slice_matches(
                        &format!("{label} ncopy[{q}][{c}] tok={tok}"),
                        gpu,
                        cpu,
                        job.ncopy[q][c] as u32 + (tok * job.tstride as usize) as u32,
                        E,
                    );
                }
            }
            if job.hcopy[q][c] >= 0 {
                assert_f16_region_matches(
                    &format!("{label} hcopy[{q}][{c}]"),
                    gpu,
                    cpu,
                    job.hcopy[q][c] as u32,
                    t,
                    job.hstride as usize,
                );
            }
        }
        if job.hout[q] >= 0 {
            assert_f16_region_matches(
                &format!("{label} hout[{q}]"),
                gpu,
                cpu,
                job.hout[q] as u32,
                t,
                E,
            );
        }
        if ((job.nof32 >> q) & 1) == 0 {
            for tok in 0..t {
                assert_f32_slice_matches(
                    &format!("{label} out_off[{q}] tok={tok}"),
                    gpu,
                    cpu,
                    job.out_off[q] + (tok * job.tstride as usize) as u32,
                    E,
                );
            }
        } else {
            for tok in 0..t {
                assert_f32_slice_untouched(
                    &format!("{label} suppressed out_off[{q}] tok={tok}"),
                    gpu,
                    job.out_off[q] + (tok * job.tstride as usize) as u32,
                    E,
                );
            }
        }
    }
}

fn rand_vec(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32()).collect()
}

/// 1) T = 1 uncontrolled hgemm prefill: `hout` + `nof32 = 0xF` (exact 1-token corruption bug!).
#[test]
fn t1_hout_nof32_uncontrolled_hgemm_prefill() {
    let mut rng = Rng::new(0x1111_2222_3333_4444);
    for &(n_s, n_q, with_norm) in &[(1, 1, true), (3, 1, true), (4, 3, true), (5, 1, false)] {
        let t = 1usize;
        let mut fx = Fixture::new();
        let mut job = empty_job();
        job.n_s = n_s;
        job.n_q = n_q;
        job.tstride = E as u32;
        job.eps = 1e-5;
        job.nof32 = 0xF;

        for i in 0..n_s as usize {
            let v = rand_vec(&mut rng, t * E);
            job.s_off[i] = fx.push_f32_slice(&v);
        }
        let qv = rand_vec(&mut rng, n_q as usize * E);
        job.q_off = fx.push_pb_f32(&qv);
        for q in 0..n_q as usize {
            job.out_off[q] = fx.alloc_f32_sentinel(t * E);
            job.hout[q] = fx.alloc_f16_sentinel(t * E) as i32;
            if with_norm {
                let nw = rand_vec(&mut rng, E);
                job.norm_off[q] = fx.push_pb_f32(&nw);
            }
        }

        let mut cpu = fx.arena.clone();
        cpu_ref_mix_jobs(&fx.pb, &mut cpu, &[job], t);
        let gpu_out = fx.run_gpu(&[job], t, false);
        verify_job_outputs(
            &format!("t1_hout_nof32(n_s={n_s},n_q={n_q},norm={with_norm})"),
            &job,
            t,
            &gpu_out,
            &cpu,
        );
    }
}

/// T = 1 controlled hgemm prefill: `hcopy` into `[u^ | m^]` with `hstride = 2 * E`
/// (exact 1-token loop-controller bug!).
#[test]
fn t1_hcopy_controlled_hgemm_prefill() {
    let mut rng = Rng::new(0xAAAA_BBBB_CCCC_DDDD);
    for &swa in &[true, false] {
        let t = 1usize;
        let mut fx = Fixture::new();

        // Mirror layer_b controlled attention mix: ja (n_s=3, n_q=1 or 3) + jm (n_s=2, n_q=4)
        // writing normalized copies into cath[0..4] (each [t][2*E] halves).
        let cath: [u32; 4] = std::array::from_fn(|_| fx.alloc_f16_sentinel(t * 2 * E));

        let n_q_a = if swa { 1 } else { 3 };
        let mut ja = empty_job();
        ja.n_s = 3;
        ja.n_q = n_q_a;
        ja.tstride = E as u32;
        for i in 0..3 {
            let v = rand_vec(&mut rng, t * E);
            ja.s_off[i] = fx.push_f32_slice(&v);
        }
        let qa = rand_vec(&mut rng, n_q_a as usize * E);
        ja.q_off = fx.push_pb_f32(&qa);
        for q in 0..n_q_a as usize {
            ja.out_off[q] = fx.alloc_f32_sentinel(t * E);
        }
        if swa {
            ja.hcopy[0] = [cath[0] as i32, cath[1] as i32, cath[2] as i32];
        } else {
            for (j, &c) in cath.iter().enumerate().take(3) {
                ja.hcopy[j][0] = c as i32;
            }
        }

        let mut jm = empty_job();
        jm.n_s = 2;
        jm.n_q = 4;
        jm.tstride = E as u32;
        for i in 0..2 {
            let v = rand_vec(&mut rng, t * E);
            jm.s_off[i] = fx.push_f32_slice(&v);
        }
        let qm = rand_vec(&mut rng, 4 * E);
        jm.q_off = fx.push_pb_f32(&qm);
        for (j, &c) in cath.iter().enumerate() {
            jm.out_off[j] = fx.alloc_f32_sentinel(t * E);
            jm.hcopy[j][0] = (c + (E / 2) as u32) as i32;
        }

        let jobs = [ja, jm];
        let mut cpu = fx.arena.clone();
        cpu_ref_mix_jobs(&fx.pb, &mut cpu, &jobs, t);
        let gpu_out = fx.run_gpu(&jobs, t, false);
        verify_job_outputs(&format!("t1_hcopy_ja(swa={swa})"), &ja, t, &gpu_out, &cpu);
        verify_job_outputs(&format!("t1_hcopy_jm(swa={swa})"), &jm, t, &gpu_out, &cpu);
        // cath[3][0..E] was NOT written by ja (only cath[3][E..2E] written by jm), so guard must remain untouched!
        assert_f16_region_untouched(&format!("t1_hcopy_cath3_left(swa={swa})"), &gpu_out, cath[3], t, 2 * E);
    }
}

/// `old_mixb = true` (`INK_OLD_MIXB=1`) across `T = 1, 2, 4, 17` with `hout`, `hcopy`, `ncopy`, and `nof32`.
#[test]
fn old_mixb_all_batch_sizes_hout_hcopy_nof32() {
    let mut rng = Rng::new(0x5555_6666_7777_8888);
    for &t in &[1usize, 2, 4, 17] {
        let mut fx = Fixture::new();
        let cath: [u32; 3] = std::array::from_fn(|_| fx.alloc_f16_sentinel(t * 2 * E));

        let mut job = empty_job();
        job.n_s = 4;
        job.n_q = 3;
        job.tstride = E as u32;
        job.eps = 1e-6;
        // Suppress f32 output for q=0 and q=2, keep f32 output for q=1!
        job.nof32 = 0b0101;

        for i in 0..4 {
            let v = rand_vec(&mut rng, t * E);
            job.s_off[i] = fx.push_f32_slice(&v);
        }
        let qv = rand_vec(&mut rng, 3 * E);
        job.q_off = fx.push_pb_f32(&qv);

        for (q, &c) in cath.iter().enumerate() {
            job.out_off[q] = fx.alloc_f32_sentinel(t * E);
            job.hout[q] = fx.alloc_f16_sentinel(t * E) as i32;
            job.ncopy[q][0] = fx.alloc_f32_sentinel(t * E) as i32;
            job.hcopy[q][0] = c as i32;
            // Norm q=0 and q=2, leave q=1 unnormed to test both branches in k_mix.
            if q != 1 {
                let nw = rand_vec(&mut rng, E);
                job.norm_off[q] = fx.push_pb_f32(&nw);
            }
        }

        let mut cpu = fx.arena.clone();
        cpu_ref_mix_jobs(&fx.pb, &mut cpu, &[job], t);
        let gpu_out = fx.run_gpu(&[job], t, true);
        verify_job_outputs(&format!("old_mixb(t={t})"), &job, t, &gpu_out, &cpu);
        // Right half of cath[q] (offset + E/2 floats) was not written -> must remain sentinel!
        for (q, &c) in cath.iter().enumerate() {
            assert_f16_region_untouched(
                &format!("old_mixb(t={t}) cath[{q}]_right"),
                &gpu_out,
                c + (E / 2) as u32,
                t,
                2 * E,
            );
        }
    }
}

/// Metamorphic consistency: token 0 in `T = 1` vs `T = 4` vs `T = 17` (and `old_mixb = false` vs `true`).
#[test]
fn metamorphic_t1_vs_t4_vs_t17_consistency() {
    let mut rng = Rng::new(0x9999_AAAA_BBBB_CCCC);
    let n_s = 3usize;
    let n_q = 3usize;
    let max_t = 17usize;

    // Shared input data so token 0 is identical across T = 1, 4, 17.
    let states: Vec<Vec<f32>> = (0..n_s).map(|_| rand_vec(&mut rng, max_t * E)).collect();
    let q_weights = rand_vec(&mut rng, n_q * E);
    let norm_weights: Vec<Vec<f32>> = (0..n_q).map(|_| rand_vec(&mut rng, E)).collect();

    let run_case = |t: usize, old_mixb: bool| -> (Vec<u32>, MixJob) {
        let mut fx = Fixture::new();
        let mut job = empty_job();
        job.n_s = n_s as u32;
        job.n_q = n_q as u32;
        job.tstride = E as u32;
        job.eps = 1e-6;
        job.nof32 = 0b0010; // keep f32 for q=0,2; suppress q=1

        for (i, st) in states.iter().enumerate() {
            job.s_off[i] = fx.push_f32_slice(&st[..t * E]);
        }
        job.q_off = fx.push_pb_f32(&q_weights);
        for (q, nw) in norm_weights.iter().enumerate() {
            job.norm_off[q] = fx.push_pb_f32(nw);
            job.out_off[q] = fx.alloc_f32_sentinel(t * E);
            job.hout[q] = fx.alloc_f16_sentinel(t * E) as i32;
            job.ncopy[q][0] = fx.alloc_f32_sentinel(t * E) as i32;
            job.hcopy[q][0] = fx.alloc_f16_sentinel(t * 2 * E) as i32;
        }
        let mut cpu = fx.arena.clone();
        cpu_ref_mix_jobs(&fx.pb, &mut cpu, &[job], t);
        let gpu_out = fx.run_gpu(&[job], t, old_mixb);
        verify_job_outputs(&format!("meta(t={t},old={old_mixb})"), &job, t, &gpu_out, &cpu);
        (gpu_out, job)
    };

    let (out_t1_new, j1_new) = run_case(1, false);
    let (out_t4_new, j4_new) = run_case(4, false);
    let (out_t17_new, j17_new) = run_case(17, false);
    let (out_t1_old, j1_old) = run_case(1, true);
    let (out_t17_old, j17_old) = run_case(17, true);

    // Token 0 across T=1, T=4, T=17 under the same dispatch path (old_mixb=false) must match bit-for-bit!
    for q in 0..n_q {
        for e in 0..E {
            let h1 = read_half_in_u32_slice(&out_t1_new, j1_new.hout[q] as usize * 2 + e);
            let h4 = read_half_in_u32_slice(&out_t4_new, j4_new.hout[q] as usize * 2 + e);
            let h17 = read_half_in_u32_slice(&out_t17_new, j17_new.hout[q] as usize * 2 + e);
            assert_eq!(h1, h4, "hout[{q}] tok=0 T=1 vs T=4 mismatch at e={e}");
            assert_eq!(h1, h17, "hout[{q}] tok=0 T=1 vs T=17 mismatch at e={e}");

            let hc1 = read_half_in_u32_slice(&out_t1_new, j1_new.hcopy[q][0] as usize * 2 + e);
            let hc4 = read_half_in_u32_slice(&out_t4_new, j4_new.hcopy[q][0] as usize * 2 + e);
            let hc17 = read_half_in_u32_slice(&out_t17_new, j17_new.hcopy[q][0] as usize * 2 + e);
            assert_eq!(hc1, hc4, "hcopy[{q}] tok=0 T=1 vs T=4 mismatch at e={e}");
            assert_eq!(hc1, hc17, "hcopy[{q}] tok=0 T=1 vs T=17 mismatch at e={e}");

            // Cross-kernel check: old_mixb=false vs old_mixb=true on token 0
            let h1_old = read_half_in_u32_slice(&out_t1_old, j1_old.hout[q] as usize * 2 + e);
            let h17_old = read_half_in_u32_slice(&out_t17_old, j17_old.hout[q] as usize * 2 + e);
            assert_eq!(h1_old, h17_old, "k_mix hout[{q}] tok=0 T=1 vs T=17 mismatch at e={e}");
            let v_new = f16_to_f32(h1);
            let v_old = f16_to_f32(h1_old);
            let tol = 2e-3 + 2e-3 * v_new.abs();
            assert!(
                (v_new - v_old).abs() <= tol,
                "k_mix_b vs k_mix hout[{q}] tok=0 mismatch at e={e}: new={v_new} old={v_old}"
            );
        }
        if ((j1_new.nof32 >> q) & 1) == 0 {
            for e in 0..E {
                let f1 = out_t1_new[j1_new.out_off[q] as usize + e];
                let f4 = out_t4_new[j4_new.out_off[q] as usize + e];
                let f17 = out_t17_new[j17_new.out_off[q] as usize + e];
                assert_eq!(f1, f4, "out_off[{q}] tok=0 T=1 vs T=4 mismatch at e={e}");
                assert_eq!(f1, f17, "out_off[{q}] tok=0 T=1 vs T=17 mismatch at e={e}");
            }
        }
    }
}

/// 5) Standard decode-style T = 1 (`hout=-1, hcopy=-1, nof32=0, tstride=0`) across all model `(n_s, n_q)`
///    AND `q_off < 0` (`k_mix_b_id`) paths (with and without `hout`/`hcopy`/`nof32`).
#[test]
fn decode_t1_standard_and_identity_paths() {
    let mut rng = Rng::new(0xCAFE_BABE_1234_5678);

    // Every (n_s, n_q) combination used across decode and prefill:
    for &(n_s, n_q) in &[
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
        (2, 4), (6, 4),
    ] {
        for &old_mixb in &[false, true] {
            let mut fx = Fixture::new();
            let mut job = empty_job();
            job.n_s = n_s;
            job.n_q = n_q;
            job.tstride = 0; // decode style
            job.eps = 1e-6;
            for i in 0..n_s as usize {
                let v = rand_vec(&mut rng, E);
                job.s_off[i] = fx.push_f32_slice(&v);
            }
            let qv = rand_vec(&mut rng, n_q as usize * E);
            job.q_off = fx.push_pb_f32(&qv);
            for q in 0..n_q as usize {
                job.out_off[q] = fx.alloc_f32_sentinel(E);
                job.ncopy[q][0] = fx.alloc_f32_sentinel(E) as i32;
                let nw = rand_vec(&mut rng, E);
                job.norm_off[q] = fx.push_pb_f32(&nw);
            }
            let mut cpu = fx.arena.clone();
            cpu_ref_mix_jobs(&fx.pb, &mut cpu, &[job], 1);
            let gpu_out = fx.run_gpu(&[job], 1, old_mixb);
            verify_job_outputs(
                &format!("decode_t1(n_s={n_s},n_q={n_q},old={old_mixb})"),
                &job,
                1,
                &gpu_out,
                &cpu,
            );
        }
    }

    // Identity path (q_off < 0 -> k_mix_b_id) for T=1 and T=5, both decode-style and hgemm-prefill-style.
    for &t in &[1usize, 5] {
        for &old_mixb in &[false, true] {
            for &with_half in &[false, true] {
                let mut fx = Fixture::new();
                let mut job = empty_job();
                job.n_s = 1;
                job.n_q = 1;
                job.q_off = -1;
                job.tstride = if t == 1 { 0 } else { E as u32 };
                job.eps = 1e-6;
                let s0 = rand_vec(&mut rng, t * E);
                job.s_off[0] = fx.push_f32_slice(&s0);
                job.out_off[0] = fx.alloc_f32_sentinel(t * E);
                let nw = rand_vec(&mut rng, E);
                job.norm_off[0] = fx.push_pb_f32(&nw);
                if with_half {
                    job.hout[0] = fx.alloc_f16_sentinel(t * E) as i32;
                    job.hcopy[0][0] = fx.alloc_f16_sentinel(t * 2 * E) as i32;
                    job.nof32 = 1;
                } else {
                    job.ncopy[0][0] = fx.alloc_f32_sentinel(t * E) as i32;
                }
                let mut cpu = fx.arena.clone();
                cpu_ref_mix_jobs(&fx.pb, &mut cpu, &[job], t);
                let gpu_out = fx.run_gpu(&[job], t, old_mixb);
                verify_job_outputs(
                    &format!("ident(t={t},old={old_mixb},half={with_half})"),
                    &job,
                    t,
                    &gpu_out,
                    &cpu,
                );
            }
        }
    }
}

/// 6) Unspecialized `(n_s, n_q)` (e.g. `n_q = 2`, which has no `k_mix_b_*_2` specialization)
///    must cleanly fall back to `k_mix` even when `old_mixb = false`, and still produce valid `hout`/`hcopy`/`nof32`.
#[test]
fn fallback_unspecialized_nq2_with_hout_hcopy() {
    let mut rng = Rng::new(0xDEAD_BEEF_0001_0002);
    for &t in &[1usize, 4] {
        let mut fx = Fixture::new();
        let mut job = empty_job();
        job.n_s = 2;
        job.n_q = 2;
        job.tstride = E as u32;
        job.nof32 = 0b0011;
        for i in 0..2 {
            let v = rand_vec(&mut rng, t * E);
            job.s_off[i] = fx.push_f32_slice(&v);
        }
        let qv = rand_vec(&mut rng, 2 * E);
        job.q_off = fx.push_pb_f32(&qv);
        for q in 0..2 {
            job.out_off[q] = fx.alloc_f32_sentinel(t * E);
            job.hout[q] = fx.alloc_f16_sentinel(t * E) as i32;
            job.hcopy[q][0] = fx.alloc_f16_sentinel(t * 2 * E) as i32;
            let nw = rand_vec(&mut rng, E);
            job.norm_off[q] = fx.push_pb_f32(&nw);
        }
        let mut cpu = fx.arena.clone();
        cpu_ref_mix_jobs(&fx.pb, &mut cpu, &[job], t);
        let gpu_out = fx.run_gpu(&[job], t, false);
        verify_job_outputs(&format!("fallback_nq2(t={t})"), &job, t, &gpu_out, &cpu);
    }
}
