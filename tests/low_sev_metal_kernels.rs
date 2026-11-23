//! GPU-vs-CPU-reference tests for Low-Severity Metal kernel defects:
//! - Low-Sev #1: `MTLMathMode::Safe` vs `-INFINITY` sentinels (`metal.rs`, `kernels.metal`, `prefill.metal`)
//! - Low-Sev #3: RMSNorm epsilon in `k_ctrl_apply` & `k_ctrl_apply_b` (`CtrlArgs.eps`, `CtrlBArgs.eps`)
//! - Low-Sev #5: `deq16` correctness across all weight types & zero unreachable duplicate branches

use inkling::engine::{CtrlArgs, CtrlBArgs, CtrlBJob, CtrlJob, MixJob, E};
use inkling::gguf::{T_BF16, T_Q4_K, T_Q5_0, T_Q6_K, T_Q8_0};
use inkling::metal::{Enc, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLMathMode};
use std::mem::{offset_of, size_of};
use std::sync::{Mutex, OnceLock};

const NH: usize = 24;
const NKV: usize = 8;
const HD: usize = 128;
const NREL: usize = 16;
const EXT: usize = 1024;

/// Production Metal shader source (`kernels.metal` + `prefill.metal`).
fn metal_src() -> String {
    format!(
        "{}\n{}",
        include_str!("../src/kernels.metal"),
        include_str!("../src/prefill.metal")
    )
}

fn with_gpu<R>(f: impl FnOnce(&Gpu) -> R) -> R {
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let guard = GPU
        .get_or_init(|| Mutex::new(Gpu::new(&metal_src(), &[])))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    f(&guard)
}

/// Deterministic xorshift64* PRNG for reproducible test vectors.
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
        (u * 2.0 - 1.0) * 1.25
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
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
        return sign | 0x7C00;
    }
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
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

/// Round f32 to bf16 (truncating lower 16 bits, matching `bf16f` in `kernels.metal`).
fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_to_f32(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

fn as_bytes<T>(s: &[T]) -> &[u8] {
    // SAFETY: Caller passes plain-old-data numeric slices (`f32`/`u16`/`u32`); reading
    // their initialized byte representation is valid for `size_of_val(s)` bytes.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn read_buf_f32(buf: &inkling::metal::Buf, len: usize) -> Vec<f32> {
    // SAFETY: Caller waits for GPU completion before calling; `buf` is `StorageModeShared`
    // and allocated with at least `len * 4` bytes.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len).to_vec() }
}

fn read_buf_u16(buf: &inkling::metal::Buf, len: usize) -> Vec<u16> {
    // SAFETY: Caller waits for GPU completion before calling; `buf` is `StorageModeShared`
    // and allocated with at least `len * 2` bytes.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u16, len).to_vec() }
}

fn read_buf_u32(buf: &inkling::metal::Buf, len: usize) -> Vec<u32> {
    // SAFETY: Caller waits for GPU completion before calling; `buf` is `StorageModeShared`
    // and allocated with at least `len * 4` bytes.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u32, len).to_vec() }
}

// ============================================================================
// Low-Severity #3: RMSNorm epsilon in k_ctrl_apply & k_ctrl_apply_b
// ============================================================================

#[test]
fn ctrl_arg_structs_match_metal_abi_layout() {
    assert_eq!(size_of::<CtrlJob>(), 40);
    assert_eq!(size_of::<CtrlArgs>(), 176);
    assert_eq!(offset_of!(CtrlArgs, n_c), 0);
    assert_eq!(offset_of!(CtrlArgs, tstride), 4);
    assert_eq!(offset_of!(CtrlArgs, eps), 8);
    assert_eq!(offset_of!(CtrlArgs, pad), 12);
    assert_eq!(offset_of!(CtrlArgs, job), 16);

    assert_eq!(size_of::<CtrlBJob>(), 32);
    assert_eq!(size_of::<CtrlBArgs>(), 144);
    assert_eq!(offset_of!(CtrlBArgs, n), 0);
    assert_eq!(offset_of!(CtrlBArgs, eps), 4);
    assert_eq!(offset_of!(CtrlBArgs, pad), 8);
    assert_eq!(offset_of!(CtrlBArgs, job), 16);
}

#[test]
fn ctrl_apply_decode_respects_non_default_eps() {
    with_gpu(|gpu| {
        let mut rng = Rng::new(0xC781_0001);
        let n_jobs = 2usize;

        // PB layout (bytes): per job: gain [E x 64 bf16], forget [E x 64 bf16], norm_w [E f32]
        let mut pb: Vec<u8> = Vec::new();
        let mut gain_offs = Vec::new();
        let mut forget_offs = Vec::new();
        let mut norm_offs = Vec::new();
        let mut gains_bf16 = Vec::new();
        let mut forgets_bf16 = Vec::new();
        let mut norm_ws = Vec::new();

        for _ in 0..n_jobs {
            gain_offs.push(pb.len() as u32);
            let gain: Vec<u16> = (0..E * 64)
                .map(|_| f32_to_bf16(rng.next_f32() * 0.25))
                .collect();
            pb.extend_from_slice(as_bytes(&gain));
            gains_bf16.push(gain);

            forget_offs.push(pb.len() as u32);
            let forget: Vec<u16> = (0..E * 64)
                .map(|_| f32_to_bf16(rng.next_f32() * 0.25))
                .collect();
            pb.extend_from_slice(as_bytes(&forget));
            forgets_bf16.push(forget);

            norm_offs.push(pb.len() as i32);
            let nw: Vec<f32> = (0..E).map(|_| 0.5 + rng.next_f32().abs()).collect();
            pb.extend_from_slice(as_bytes(&nw));
            norm_ws.push(nw);
        }

        // A layout (f32 indices): per job: u [E], m [E], g [64], res_out [E], ss [128]
        let mut a_host: Vec<f32> = Vec::new();
        let mut u_offs = Vec::new();
        let mut m_offs = Vec::new();
        let mut g_offs = Vec::new();
        let mut res_offs = Vec::new();
        let mut ss_offs = Vec::new();
        let mut us = Vec::new();
        let mut ms = Vec::new();
        let mut gs = Vec::new();

        for _ in 0..n_jobs {
            u_offs.push(a_host.len() as u32);
            let u: Vec<f32> = (0..E).map(|_| rng.next_f32()).collect();
            a_host.extend_from_slice(&u);
            us.push(u);

            m_offs.push(a_host.len() as u32);
            let m: Vec<f32> = (0..E).map(|_| rng.next_f32()).collect();
            a_host.extend_from_slice(&m);
            ms.push(m);

            g_offs.push(a_host.len() as u32);
            let g: Vec<f32> = (0..64).map(|_| rng.next_f32() * 0.5).collect();
            a_host.extend_from_slice(&g);
            gs.push(g);

            res_offs.push(a_host.len() as u32);
            a_host.extend(std::iter::repeat_n(0.0f32, E));

            ss_offs.push(a_host.len() as u32);
            a_host.extend(std::iter::repeat_n(0.0f32, 128));
        }

        for eps in [1.0f32, 0.25f32, 1e-5f32] {
            let pb_buf = gpu.buffer_with(&pb);
            let a_buf = gpu.buffer_with(as_bytes(&a_host));
            let ctr_buf = gpu.buffer_with(as_bytes(&[0u32; 8]));

            let mut args = CtrlArgs {
                n_c: n_jobs as u32,
                tstride: 0,
                eps,
                pad: 0,
                job: [CtrlJob::default(); 4],
            };
            for j in 0..n_jobs {
                args.job[j] = CtrlJob {
                    u_off: u_offs[j],
                    m_off: m_offs[j],
                    d_off: 0,
                    out_off: g_offs[j],
                    gain_off: gain_offs[j],
                    forget_off: forget_offs[j],
                    res_out: res_offs[j],
                    ss_off: ss_offs[j],
                    norm_w: norm_offs[j],
                    ctr: j as u32,
                };
            }

            let cb = gpu.cmd();
            let enc = Enc::new(&cb);
            enc.pso(&gpu.pso("k_ctrl_apply"));
            enc.buf(&pb_buf, 0, 1);
            enc.buf(&a_buf, 0, 2);
            enc.bytes(&args, 5);
            enc.buf(&ctr_buf, 0, 7);
            enc.dispatch((E / 32, n_jobs, 1), 256);
            enc.end();
            cb.commit();
            cb.waitUntilCompleted();

            let got = read_buf_f32(&a_buf, a_host.len());

            for j in 0..n_jobs {
                let mut raw = vec![0.0f64; E];
                let mut ss = 0.0f64;
                for ch in 0..E {
                    let mut dg = 0.0f64;
                    let mut df = 0.0f64;
                    for k in 0..64 {
                        let gv = gs[j][k] as f64;
                        dg += bf16_to_f32(gains_bf16[j][ch * 64 + k]) as f64 * gv;
                        df += bf16_to_f32(forgets_bf16[j][ch * 64 + k]) as f64 * gv;
                    }
                    let u = us[j][ch] as f64;
                    let m = ms[j][ch] as f64;
                    let o = u - df.tanh() * u + dg.tanh() * m;
                    raw[ch] = o;
                    ss += o * o;
                }
                let r = 1.0 / (ss / (E as f64) + eps as f64).sqrt();
                let mut max_err = 0.0f64;
                for ch in 0..E {
                    let expected = raw[ch] * r * (norm_ws[j][ch] as f64);
                    let actual = got[res_offs[j] as usize + ch] as f64;
                    assert!(actual.is_finite(), "non-finite output at job={j} ch={ch}");
                    max_err = max_err.max((actual - expected).abs());
                }
                assert!(
                    max_err < 1e-3,
                    "k_ctrl_apply eps={eps} job={j} max_err={max_err:.6e}"
                );
            }
        }
    });
}

#[test]
fn ctrl_apply_prefill_respects_non_default_eps() {
    with_gpu(|gpu| {
        let mut rng = Rng::new(0xC781_0002);

        for t in [1usize, 4usize] {
            for pre_applied in [false, true] {
                let norm_w: Vec<f32> = (0..E).map(|_| 0.5 + rng.next_f32().abs()).collect();
                let pb_buf = gpu.buffer_with(as_bytes(&norm_w));

                let mut a_host: Vec<f32> = Vec::new();
                let u_off = a_host.len() as u32;
                let u: Vec<f32> = (0..t * E).map(|_| rng.next_f32()).collect();
                a_host.extend_from_slice(&u);

                let m_off = a_host.len() as u32;
                let m: Vec<f32> = (0..t * E).map(|_| rng.next_f32()).collect();
                a_host.extend_from_slice(&m);

                let g_off = a_host.len() as u32;
                let g: Vec<f32> = (0..t * E).map(|_| rng.next_f32()).collect();
                a_host.extend_from_slice(&g);

                let f_off = a_host.len() as u32;
                let f_in: Vec<f32> = (0..t * E).map(|_| rng.next_f32()).collect();
                a_host.extend_from_slice(&f_in);

                let out_off = a_host.len() as u32;
                a_host.extend(std::iter::repeat_n(0.0f32, t * E));

                let h_off = a_host.len() as i32;
                a_host.extend(std::iter::repeat_n(0.0f32, (t * E) / 2));

                let eps = 1.0f32;
                let a_buf = gpu.buffer_with(as_bytes(&a_host));

                let mut args = CtrlBArgs {
                    n: 1,
                    eps,
                    pad: [0; 2],
                    job: [CtrlBJob::default(); 4],
                };
                args.job[0] = CtrlBJob {
                    u_off,
                    m_off: if pre_applied { u32::MAX } else { m_off },
                    g_off,
                    f_off,
                    out_off,
                    norm_w: 0,
                    h_off,
                    nof32: 0,
                };

                let cb = gpu.cmd();
                let enc = Enc::new(&cb);
                enc.pso(&gpu.pso("k_ctrl_apply_b"));
                enc.buf(&pb_buf, 0, 1);
                enc.buf(&a_buf, 0, 2);
                enc.bytes(&args, 5);
                enc.dispatch((t, 1, 1), 256);
                enc.end();
                cb.commit();
                cb.waitUntilCompleted();

                let got_f32 = read_buf_f32(&a_buf, a_host.len());
                let got_u16 = read_buf_u16(&a_buf, a_host.len() * 2);

                for tok in 0..t {
                    let mut raw = vec![0.0f64; E];
                    let mut ss = 0.0f64;
                    for (i, slot) in raw.iter_mut().enumerate().take(E) {
                        let idx = tok * E + i;
                        let o = if pre_applied {
                            u[idx] as f64
                        } else {
                            let uv = u[idx] as f64;
                            let mv = m[idx] as f64;
                            let fv = f_in[idx] as f64;
                            let gv = g[idx] as f64;
                            uv - fv.tanh() * uv + gv.tanh() * mv
                        };
                        *slot = o;
                        ss += o * o;
                    }
                    let r = 1.0 / (ss / (E as f64) + eps as f64).sqrt();
                    let mut max_err = 0.0f64;
                    let mut max_herr = 0.0f64;
                    for i in 0..E {
                        let expected = raw[i] * r * (norm_w[i] as f64);
                        let actual = got_f32[out_off as usize + tok * E + i] as f64;
                        let actual_h =
                            f16_to_f32(got_u16[(h_off as usize) * 2 + tok * E + i]) as f64;
                        assert!(actual.is_finite() && actual_h.is_finite());
                        max_err = max_err.max((actual - expected).abs());
                        max_herr = max_herr.max((actual_h - expected).abs());
                    }
                    assert!(
                        max_err < 1e-3 && max_herr < 5e-3,
                        "k_ctrl_apply_b T={t} pre_applied={pre_applied} eps={eps} tok={tok} max_err={max_err:.6e} max_herr={max_herr:.6e}"
                    );
                }
            }
        }
    });
}

// ============================================================================
// Low-Severity #1: MTLMathMode::Safe vs -INFINITY sentinels + GPU timing check
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TestAttnArgs {
    pos: u32,
    nkeys: u32,
    first_pos: u32,
    ring: u32,
    cap: u32,
    k_base: u32,
    v_base: u32,
    cs_off: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    r_off: u32,
    sink_off: u32,
    kdyn_off: i32,
    vdyn_off: i32,
    part_off: u32,
    kconv_w: u32,
    vconv_w: u32,
    qclip_w: u32,
    relclip_w: u32,
    relbias_w: u32,
    tau: f32,
    kps: u32,
    nsplit: u32,
    ctr: u32,
    gate_off: u32,
    out_off: u32,
    pad: u32,
    sp0: u32,
    noown: u32,
    gstr: u32,
    istr: u32,
    idx0: u32,
    lastbias: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TestAttnBArgs {
    t: u32,
    p0: u32,
    ring: u32,
    cap: u32,
    k_base: u32,
    v_base: u32,
    window: u32,
    qp_off: u32,
    rp_off: u32,
    sink_off: u32,
    gate_off: u32,
    out_off: u32,
    relbias_w: u32,
    ghalf: u32,
    kpos0: i32,
    pad2: u32,
}

/// Verifies that the production Metal shaders contain zero `-INFINITY` sentinels
/// and that `Gpu::new` (default `MTLMathMode::Fast`) evaluates finite sentinel
/// masking (`NEG_INF_F = -1e30f`, `<= -1e29f`, `exp(NEG_INF_F - mx + 8.0f) == 0.0f`)
/// cleanly without NaN or fast-math miscompilation.
#[test]
fn gpu_default_fast_math_with_finite_sentinels_matches_exact_zero() {
    // Structural guard: neither shader file may reintroduce `-INFINITY` / `INFINITY` literals
    // while compiled under MTLMathMode::Fast.
    for (name, src) in [
        ("src/kernels.metal", include_str!("../src/kernels.metal")),
        ("src/prefill.metal", include_str!("../src/prefill.metal")),
    ] {
        let code_only: String = src
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !code_only.contains("INFINITY"),
            "{name} must use finite sentinels (NEG_INF_F) instead of INFINITY under MTLMathMode::Fast"
        );
    }

    let probe_src = r#"
#include <metal_stdlib>
using namespace metal;
constant float NEG_INF_F   = -1e30f;
constant float NEG_INF_CMP = -1e29f;
kernel void k_sentinel_probe(device const float * in [[buffer(0)]],
                             device float * out [[buffer(1)]],
                             device uint * flags [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float ms[4];
    for (int i = 0; i < 4; i++) ms[i] = in[i];
    // Exact k_attn merge pattern with all-masked NEG_INF_F inputs
    float M = NEG_INF_F;
    for (int i = 0; i < 4; i++) M = max(M, ms[i]);
    float L = 0.0f;
    for (int i = 0; i < 4; i++) {
        const float w = ms[i] <= NEG_INF_CMP ? 0.0f : exp(ms[i] - M);
        L += w;
    }
    out[0] = M;
    out[1] = L;
    flags[0] = (ms[0] <= NEG_INF_CMP) ? 1u : 0u;
    // Exact k_attn_h masked-key pattern under Fast math: exp(NEG_INF_F - finite + 8.0f) == 0.0f
    float fin = in[4];
    float masked = NEG_INF_F;
    float mx = max(fin, masked);
    out[2] = exp(masked - mx + 8.0f);
    // Empty-split combine against zero-logit sink (M_sink = 0.0f): exp(NEG_INF_F - 0.0f) == 0.0f
    out[3] = exp(M - max(0.0f, M));
    flags[1] = isnan(L) ? 1u : 0u;
}
"#;
    let gpu = Gpu::new(probe_src, &["k_sentinel_probe"]);
    assert_eq!(
        gpu.math_mode(),
        MTLMathMode::Fast,
        "Gpu::new must default to MTLMathMode::Fast (using finite sentinels)"
    );

    let input = [-1e30f32, -1e30f32, -1e30f32, -1e30f32, 1.25f32];
    let in_buf = gpu.buffer_with(as_bytes(&input));
    let out_buf = gpu.buffer_with(as_bytes(&[0.0f32; 4]));
    let flag_buf = gpu.buffer_with(as_bytes(&[0u32; 4]));

    let cb = gpu.cmd();
    let enc = Enc::new(&cb);
    enc.pso(&gpu.pso("k_sentinel_probe"));
    enc.buf(&in_buf, 0, 0);
    enc.buf(&out_buf, 0, 1);
    enc.buf(&flag_buf, 0, 2);
    enc.dispatch((1, 1, 1), 1);
    enc.end();
    cb.commit();
    cb.waitUntilCompleted();

    let out = read_buf_f32(&out_buf, 4);
    let flags = read_buf_u32(&flag_buf, 4);
    assert_eq!(flags[0], 1, "NEG_INF_F <= NEG_INF_CMP must be true under Fast math");
    assert_eq!(flags[1], 0, "all-masked split produced NaN in L={}", out[1]);
    assert!(out[0].is_finite() && out[0] <= -1e29f32, "M={}", out[0]);
    assert_eq!(out[1], 0.0f32, "empty split L must be 0.0, got {}", out[1]);
    assert_eq!(out[2], 0.0f32, "masked k_attn_h exp must be 0.0, got {}", out[2]);
    assert_eq!(out[3], 0.0f32, "empty split combine weight must be 0.0, got {}", out[3]);
}

/// Exercises real `k_attn` with an empty split (`sp = 1` has 0 keys so all 4 simdgroups
/// hit `ms_[s2][h] == -INFINITY` and `M == -INFINITY` at `kernels.metal:687`) and real
/// `k_attn_b` with causal + sliding-window `-INFINITY` masking (`prefill.metal:535`),
/// verifying zero NaNs and exact CPU-reference match under `MTL_SHADER_VALIDATION=1`.
#[test]
fn real_attention_kernels_handle_masked_and_empty_splits_without_nan() {
    with_gpu(|gpu| {
        let mut rng = Rng::new(0xA771_0001);

        // ---- 1. Real k_attn with nsplit=2, nkeys=4, kps=4:
        //         split 0 has 4 keys (< 32, so simdgroups 1..3 stay at m=-INFINITY);
        //         split 1 is empty (all 4 simdgroups have m=-INFINITY, M=-INFINITY).
        let cap = 16usize;
        let nkeys = 4usize;
        let nsplit = 2usize;
        let kps = 4usize;

        let mut pb: Vec<u8> = Vec::new();
        let qclip_w = pb.len() as u32;
        let qclip: Vec<f32> = (0..NH * HD)
            .map(|_| 0.8 + 0.2 * rng.next_f32().abs())
            .collect();
        pb.extend_from_slice(as_bytes(&qclip));

        let relclip_w = pb.len() as u32;
        let relclip: Vec<f32> = (0..NH * NREL)
            .map(|_| 0.8 + 0.2 * rng.next_f32().abs())
            .collect();
        pb.extend_from_slice(as_bytes(&relclip));

        let relbias_w = pb.len() as u32;
        let relbias: Vec<f32> = (0..EXT * NREL).map(|_| rng.next_f32() * 0.1).collect();
        pb.extend_from_slice(as_bytes(&relbias));

        let mut a_host: Vec<f32> = Vec::new();
        let q_off = a_host.len() as u32;
        let q_raw: Vec<f32> = (0..NH * HD).map(|_| rng.next_f32() * 0.5).collect();
        a_host.extend_from_slice(&q_raw);

        let r_off = a_host.len() as u32;
        let r_raw: Vec<f32> = (0..NH * NREL).map(|_| rng.next_f32() * 0.5).collect();
        a_host.extend_from_slice(&r_raw);

        let sink_off = a_host.len() as u32;
        let sink: Vec<f32> = (0..NH).map(|_| rng.next_f32() * 0.5).collect();
        a_host.extend_from_slice(&sink);

        let gate_off = a_host.len() as u32;
        let gate: Vec<f32> = (0..NH * HD).map(|_| rng.next_f32()).collect();
        a_host.extend_from_slice(&gate);

        let part_off = a_host.len() as u32;
        a_host.extend(std::iter::repeat_n(0.0f32, NH * nsplit * (HD + 2)));

        let out_off = a_host.len() as u32;
        a_host.extend(std::iter::repeat_n(0.0f32, NH * HD));

        let mut kv_host = vec![0u16; 2 * NKV * cap * HD];
        let k_base = 0usize;
        let v_base = NKV * cap * HD;
        for g in 0..NKV {
            for p in 0..nkeys {
                for d in 0..HD {
                    kv_host[k_base + (g * cap + p) * HD + d] = f32_to_f16(rng.next_f32() * 0.5);
                    kv_host[v_base + (g * cap + p) * HD + d] = f32_to_f16(rng.next_f32() * 0.5);
                }
            }
        }

        let pb_buf = gpu.buffer_with(&pb);
        let a_buf = gpu.buffer_with(as_bytes(&a_host));
        let kv_buf = gpu.buffer_with(as_bytes(&kv_host));
        let cs_buf = gpu.buffer_with(as_bytes(&[0.0f32; 64]));
        let ctr_buf = gpu.buffer_with(as_bytes(&[0u32; NKV]));

        let tau = 1.0f32;
        let args = TestAttnArgs {
            pos: 5,
            nkeys: nkeys as u32,
            first_pos: 0,
            ring: 0,
            cap: cap as u32,
            k_base: k_base as u32,
            v_base: v_base as u32,
            cs_off: 0,
            q_off,
            k_off: 0,
            v_off: 0,
            r_off,
            sink_off,
            kdyn_off: -1,
            vdyn_off: -1,
            part_off,
            kconv_w: 0,
            vconv_w: 0,
            qclip_w,
            relclip_w,
            relbias_w,
            tau,
            kps: kps as u32,
            nsplit: nsplit as u32,
            ctr: 0,
            gate_off,
            out_off,
            pad: 0,
            sp0: 0,
            noown: 1,
            gstr: (cap * HD) as u32,
            istr: HD as u32,
            idx0: 0,
            lastbias: 0.0,
        };

        let cb = gpu.cmd();
        let enc = Enc::new(&cb);
        enc.pso(&gpu.pso("k_attn"));
        enc.buf(&pb_buf, 0, 1);
        enc.buf(&a_buf, 0, 2);
        enc.buf(&kv_buf, 0, 3);
        enc.buf(&cs_buf, 0, 4);
        enc.bytes(&args, 5);
        enc.buf(&ctr_buf, 0, 7);
        enc.dispatch((nsplit, NKV, 1), 128);
        enc.end();
        cb.commit();
        cb.waitUntilCompleted();

        let got = read_buf_f32(&a_buf, a_host.len());

        // Split 1 partial stats must have finite M <= -1e29f, L = 0.0 (zero NaNs)
        for h in 0..NH {
            let p1 = part_off as usize + (h * nsplit + 1) * (HD + 2);
            assert!(
                got[p1].is_finite() && got[p1] <= -1e29f32,
                "empty split M for head {h} should be finite NEG_INF_F (<= -1e29), got {}",
                got[p1]
            );
            assert_eq!(got[p1 + 1], 0.0, "empty split L for head {h} should be 0.0");
        }

        // Compare combined k_attn output against CPU f64 reference
        const QK_CLIP: f64 = 22.627416610717773;
        const REL_CLIP: f64 = 8.0;
        let inv_sqrt_hd = (HD as f64).sqrt().recip();
        let mut max_err = 0.0f64;
        for g in 0..NKV {
            for hl in 0..3 {
                let h = g * 3 + hl;
                let mut qv = [0.0f64; HD];
                let mut qn2 = 0.0f64;
                for d in 0..HD {
                    let v = q_raw[h * HD + d] as f64 * qclip[h * HD + d] as f64;
                    qv[d] = v;
                    qn2 += v * v;
                }
                let qf =
                    (QK_CLIP / (qn2.sqrt() + 1e-6)).clamp(0.0, 1.0) * tau as f64 * inv_sqrt_hd;
                for x in &mut qv {
                    *x *= qf;
                }

                let mut rv = [0.0f64; NREL];
                let mut rn2 = 0.0f64;
                for i in 0..NREL {
                    let v = r_raw[h * NREL + i] as f64 * relclip[h * NREL + i] as f64;
                    rv[i] = v;
                    rn2 += v * v;
                }
                let rf = (REL_CLIP / (rn2.sqrt() + 1e-6)).clamp(0.0, 1.0) * tau as f64;
                for x in &mut rv {
                    *x *= rf;
                }

                let mut logits = vec![0.0f64; nkeys];
                let mut m_max = 0.0f64; // sink logit is 0 after subtracting sink[h]
                for p in 0..nkeys {
                    let mut sc = 0.0f64;
                    for d in 0..HD {
                        sc += qv[d] * f16_to_f32(kv_host[k_base + (g * cap + p) * HD + d]) as f64;
                    }
                    let dist = 5 - p;
                    if dist < EXT {
                        for i in 0..NREL {
                            sc += rv[i] * relbias[dist * NREL + i] as f64;
                        }
                    }
                    sc -= sink[h] as f64;
                    logits[p] = sc;
                    m_max = m_max.max(sc);
                }
                let mut denom = (-m_max).exp();
                let mut out_ref = [0.0f64; HD];
                for p in 0..nkeys {
                    let w = (logits[p] - m_max).exp();
                    denom += w;
                    for d in 0..HD {
                        out_ref[d] +=
                            w * f16_to_f32(kv_host[v_base + (g * cap + p) * HD + d]) as f64;
                    }
                }
                for d in 0..HD {
                    let gval = gate[h * HD + d] as f64;
                    let sig = 1.0 / (1.0 + (-gval).exp());
                    let expected = (out_ref[d] / denom) * sig;
                    let actual = got[out_off as usize + h * HD + d] as f64;
                    assert!(actual.is_finite(), "k_attn NaN/inf at h={h} d={d}");
                    max_err = max_err.max((actual - expected).abs());
                }
            }
        }
        assert!(max_err < 1e-3, "k_attn max_err={max_err:.6e}");

        // ---- 2. Real k_attn_b with T=5 (< AQT=8), window=2 (heavy causal + window -INFINITY masking).
        // Note: `MTL_SHADER_VALIDATION=1` doubles threadgroup allocations in software, pushing
        // `k_attn_b`'s ~26 KiB static threadgroup memory to 54656 B > 32768 B hardware limit.
        // We run `k_attn_b` directly whenever `MTL_SHADER_VALIDATION != 1`, and we also test
        // the exact `k_attn_h` / `k_attn_b` `-INFINITY` masking math in a <= 4 KiB TG kernel
        // unconditionally under `MTL_SHADER_VALIDATION=1` in `gpu_default_math_mode_preserves_neg_infinity_sentinels`.
        if std::env::var("MTL_SHADER_VALIDATION").as_deref() == Ok("1") {
            return;
        }
        let t_b = 5usize;
        let cap_b = 32usize;
        let window_b = 2usize;

        let mut ab_host: Vec<f32> = Vec::new();
        let qp_off = ab_host.len() as u32;
        let qp: Vec<f32> = (0..t_b * NH * HD)
            .map(|_| f16_to_f32(f32_to_f16(rng.next_f32() * 0.4)))
            .collect();
        ab_host.extend_from_slice(&qp);

        let rp_off = ab_host.len() as u32;
        let rp: Vec<f32> = (0..t_b * NH * NREL)
            .map(|_| rng.next_f32() * 0.4)
            .collect();
        ab_host.extend_from_slice(&rp);

        let sink_b_off = ab_host.len() as u32;
        let sink_b: Vec<f32> = (0..t_b * NH).map(|_| rng.next_f32() * 0.5).collect();
        ab_host.extend_from_slice(&sink_b);

        let gate_b_off = ab_host.len() as u32;
        let gate_b: Vec<f32> = (0..t_b * NH * HD).map(|_| rng.next_f32()).collect();
        ab_host.extend_from_slice(&gate_b);

        let out_b_off = ab_host.len() as u32;
        ab_host.extend(std::iter::repeat_n(0.0f32, t_b * NH * HD));

        let mut kv_b = vec![0u16; 2 * NKV * cap_b * HD];
        let kb_base = 0usize;
        let vb_base = NKV * cap_b * HD;
        for g in 0..NKV {
            for p in 0..t_b {
                for d in 0..HD {
                    kv_b[kb_base + (g * cap_b + p) * HD + d] = f32_to_f16(rng.next_f32() * 0.4);
                    kv_b[vb_base + (g * cap_b + p) * HD + d] = f32_to_f16(rng.next_f32() * 0.4);
                }
            }
        }

        let ab_buf = gpu.buffer_with(as_bytes(&ab_host));
        let kvb_buf = gpu.buffer_with(as_bytes(&kv_b));
        let bargs = TestAttnBArgs {
            t: t_b as u32,
            p0: 0,
            ring: 0,
            cap: cap_b as u32,
            k_base: kb_base as u32,
            v_base: vb_base as u32,
            window: window_b as u32,
            qp_off,
            rp_off,
            sink_off: sink_b_off,
            gate_off: gate_b_off,
            out_off: out_b_off,
            relbias_w,
            ghalf: 0,
            kpos0: 0,
            pad2: 0,
        };

        let cb = gpu.cmd();
        let enc = Enc::new(&cb);
        enc.pso(&gpu.pso("k_attn_b"));
        enc.buf(&pb_buf, 0, 1);
        enc.buf(&ab_buf, 0, 2);
        enc.buf(&kvb_buf, 0, 3);
        enc.bytes(&bargs, 5);
        enc.dispatch((t_b.div_ceil(8), NKV, 1), 128);
        enc.end();
        cb.commit();
        cb.waitUntilCompleted();

        let got_b = read_buf_f32(&ab_buf, ab_host.len());
        let mut max_err_b = 0.0f64;
        for tok in 0..t_b {
            for h in 0..NH {
                let g = h / 3;
                let sk = sink_b[tok * NH + h] as f64;
                let mut m_max = 0.0f64; // sink logit = 0 after subtracting sk
                let mut logits = Vec::new();
                for kp in 0..=tok {
                    if tok - kp >= window_b {
                        continue;
                    }
                    let mut sc = 0.0f64;
                    for d in 0..HD {
                        let qd = qp[(tok * NH + h) * HD + d] as f64;
                        let kd = f16_to_f32(kv_b[kb_base + (g * cap_b + kp) * HD + d]) as f64;
                        sc += qd * kd;
                    }
                    let dist = tok - kp;
                    for i in 0..NREL {
                        sc +=
                            rp[(tok * NH + h) * NREL + i] as f64 * relbias[dist * NREL + i] as f64;
                    }
                    sc -= sk;
                    m_max = m_max.max(sc);
                    logits.push((kp, sc));
                }
                let mut denom = (-m_max).exp();
                let mut out_ref = [0.0f64; HD];
                for &(kp, sc) in &logits {
                    let w = (sc - m_max).exp();
                    denom += w;
                    for d in 0..HD {
                        let vd = f16_to_f32(kv_b[vb_base + (g * cap_b + kp) * HD + d]) as f64;
                        out_ref[d] += w * vd;
                    }
                }
                for d in 0..HD {
                    let gval = gate_b[(tok * NH + h) * HD + d] as f64;
                    let sig = 1.0 / (1.0 + (-gval).exp());
                    let expected = (out_ref[d] / denom) * sig;
                    let actual = got_b[out_b_off as usize + (tok * NH + h) * HD + d] as f64;
                    assert!(actual.is_finite(), "k_attn_b NaN/inf at tok={tok} h={h} d={d}");
                    max_err_b = max_err_b.max((actual - expected).abs());
                }
            }
        }
        assert!(max_err_b < 1e-3, "k_attn_b max_err={max_err_b:.6e}");
    });
}

/// Empirical GPU timing benchmark comparing `MTLMathMode::Safe` vs `MTLMathMode::Fast`
/// using interleaved `GPUEndTime - GPUStartTime` measurements on a synthetic
/// `k_mix_b_4_3` + `k_ctrl_apply_b` + `deq16` (`k_gemm` Q4_K) workload.
#[test]
fn math_mode_safe_vs_fast_gpu_timing() {
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct TestGSeg {
        w_lo: u32,
        w_hi: u32,
        row_bytes: u32,
        wtype: u32,
        nrows: u32,
        k: u32,
        tile_start: u32,
        in_off: u32,
        out_off: u32,
        ldo: u32,
        wsrc: u32,
        in2: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct TestGemmArgs {
        nseg: u32,
        t: u32,
        pad0: u32,
        pad1: u32,
        seg: [TestGSeg; 10],
    }

    let src = metal_src();
    let gpu_fast = Gpu::new_with_math_mode(
        &src,
        &["k_mix_b_4_3", "k_ctrl_apply_b", "k_gemm"],
        MTLMathMode::Fast,
    );
    let gpu_safe = Gpu::new_with_math_mode(
        &src,
        &["k_mix_b_4_3", "k_ctrl_apply_b", "k_gemm"],
        MTLMathMode::Safe,
    );
    assert_eq!(gpu_fast.math_mode(), MTLMathMode::Fast);
    assert_eq!(gpu_safe.math_mode(), MTLMathMode::Safe);

    let t = 64usize;
    let nrows = 512usize;
    let k_dim = 1024usize;
    let row_bytes = (k_dim / 256) * 144;
    let mut rng = Rng::new(0xBEEF_5AFE);

    let w_bytes: Vec<u8> = (0..nrows * row_bytes).map(|_| rng.next_u8()).collect();
    let pb_f32: Vec<f32> = (0..7 * E)
        .map(|_| 0.25 + rng.next_f32() * 0.5)
        .collect();

    let mut a_host: Vec<f32> = Vec::new();
    let mut s_off = [0u32; 6];
    for slot in s_off.iter_mut().take(4) {
        *slot = a_host.len() as u32;
        a_host.extend((0..t * E).map(|_| rng.next_f32()));
    }
    let mut mix_out = [0u32; 4];
    for slot in mix_out.iter_mut().take(3) {
        *slot = a_host.len() as u32;
        a_host.extend(std::iter::repeat_n(0.0f32, t * E));
    }
    let ctrl_out = a_host.len() as u32;
    a_host.extend(std::iter::repeat_n(0.0f32, t * E));
    let gemm_out = a_host.len() as u32;
    a_host.extend(std::iter::repeat_n(0.0f32, t * nrows));

    let w_fast = gpu_fast.buffer_with(&w_bytes);
    let pb_fast = gpu_fast.buffer_with(as_bytes(&pb_f32));
    let a_fast = gpu_fast.buffer_with(as_bytes(&a_host));

    let w_safe = gpu_safe.buffer_with(&w_bytes);
    let pb_safe = gpu_safe.buffer_with(as_bytes(&pb_f32));
    let a_safe = gpu_safe.buffer_with(as_bytes(&a_host));

    let mix_job = MixJob {
        n_s: 4,
        n_q: 3,
        q_off: 0,
        eps: 1e-6,
        s_off,
        out_off: mix_out,
        norm_off: [
            (3 * E * 4) as i32,
            (4 * E * 4) as i32,
            (5 * E * 4) as i32,
            -1,
        ],
        ncopy: [[-1; 3]; 4],
        tstride: E as u32,
        ntok: t as u32,
        hcopy: [[-1; 3]; 4],
        hout: [-1; 4],
        hstride: (2 * E) as u32,
        nof32: 0,
    };
    let mut ctrl_args = CtrlBArgs {
        n: 1,
        eps: 1e-6,
        pad: [0; 2],
        job: [CtrlBJob::default(); 4],
    };
    ctrl_args.job[0] = CtrlBJob {
        u_off: s_off[0],
        m_off: s_off[1],
        g_off: s_off[2],
        f_off: s_off[3],
        out_off: ctrl_out,
        norm_w: (6 * E * 4) as u32,
        h_off: -1,
        nof32: 0,
    };
    let mut gemm_args = TestGemmArgs {
        nseg: 1,
        t: t as u32,
        pad0: 0,
        pad1: 0,
        seg: [TestGSeg::default(); 10],
    };
    gemm_args.seg[0] = TestGSeg {
        w_lo: 0,
        w_hi: 0,
        row_bytes: row_bytes as u32,
        wtype: T_Q4_K,
        nrows: nrows as u32,
        k: k_dim as u32,
        tile_start: 0,
        in_off: ctrl_out,
        out_off: gemm_out,
        ldo: nrows as u32,
        wsrc: 0,
        in2: 0,
    };

    let run_once = |gpu: &Gpu,
                    wb: &inkling::metal::Buf,
                    pbb: &inkling::metal::Buf,
                    ab: &inkling::metal::Buf|
     -> f64 {
        let cb = gpu.cmd();
        let enc = Enc::new(&cb);
        enc.buf(wb, 0, 0);
        enc.buf(pbb, 0, 1);
        enc.buf(ab, 0, 2);
        let p_mix = gpu.pso("k_mix_b_4_3");
        let p_ctrl = gpu.pso("k_ctrl_apply_b");
        let p_gemm = gpu.pso("k_gemm");
        for _ in 0..8 {
            enc.pso(&p_mix);
            enc.bytes(&mix_job, 5);
            enc.dispatch((t.div_ceil(4), 1, 1), 128);
            enc.barrier();
            enc.pso(&p_ctrl);
            enc.bytes(&ctrl_args, 5);
            enc.dispatch((t, 1, 1), 256);
            enc.barrier();
            enc.pso(&p_gemm);
            enc.bytes(&gemm_args, 5);
            enc.dispatch((nrows / 64, t / 32, 1), 128);
            enc.barrier();
        }
        enc.end();
        cb.commit();
        cb.waitUntilCompleted();
        (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
    };

    // Warmup both pipelines before measuring
    for _ in 0..4 {
        let _ = run_once(&gpu_fast, &w_fast, &pb_fast, &a_fast);
        let _ = run_once(&gpu_safe, &w_safe, &pb_safe, &a_safe);
    }

    // Interleaved timed runs to eliminate GPU clock ramp bias
    let mut fast_ms = Vec::new();
    let mut safe_ms = Vec::new();
    for _ in 0..12 {
        fast_ms.push(run_once(&gpu_fast, &w_fast, &pb_fast, &a_fast));
        safe_ms.push(run_once(&gpu_safe, &w_safe, &pb_safe, &a_safe));
    }
    fast_ms.sort_by(|a, b| a.total_cmp(b));
    safe_ms.sort_by(|a, b| a.total_cmp(b));
    let fast_med = fast_ms[fast_ms.len() / 2];
    let safe_med = safe_ms[safe_ms.len() / 2];
    eprintln!(
        "MTLMathMode interleaved timing (8x [k_mix_b_4_3 + k_ctrl_apply_b + k_gemm(Q4_K)], T={t}): Fast median={fast_med:.3}ms, Safe median={safe_med:.3}ms, ratio={:.3}x",
        safe_med / fast_med
    );
    assert!(fast_med > 0.0 && safe_med > 0.0);
}

// ============================================================================
// Low-Severity #5: deq16 correctness across all supported weight types + no dead code
// ============================================================================

fn cpu_deq256_q4k(block: &[u8]) -> [u16; 256] {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let sc = &block[4..16];
    let qs = &block[16..144];
    let mut out = [0u16; 256];
    for j in 0..8usize {
        let (s, m) = if j < 4 {
            ((sc[j] & 63) as f32, (sc[j + 4] & 63) as f32)
        } else {
            (
                ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as f32,
                ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f32,
            )
        };
        let df = d * s;
        let mf = dmin * m;
        let sh = (j & 1) * 4;
        for l in 0..32usize {
            let q = ((qs[(j / 2) * 32 + l] >> sh) & 0xF) as f32;
            out[j * 32 + l] = f32_to_f16(q.mul_add(df, -mf));
        }
    }
    out
}

fn cpu_deq256_q6k(block: &[u8]) -> [u16; 256] {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut out = [0u16; 256];
    for n in 0..2usize {
        for g in 0..4usize {
            for l in 0..32usize {
                let sc = scales[8 * n + l / 16 + 2 * g] as i8 as f32;
                let s4 = (g >> 1) * 4;
                let s2 = 2 * g;
                let lo = (ql[64 * n + (g & 1) * 32 + l] >> s4) & 0xF;
                let hi = (qh[32 * n + l] >> s2) & 3;
                let q = ((lo | (hi << 4)) as i32) - 32;
                out[128 * n + 32 * g + l] = f32_to_f16(d * sc * (q as f32));
            }
        }
    }
    out
}

fn cpu_deq256_q8_0(blocks: &[u8]) -> [u16; 256] {
    let mut out = [0u16; 256];
    for b in 0..8usize {
        let blk = &blocks[b * 34..(b + 1) * 34];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..32usize {
            let q = blk[2 + i] as i8 as f32;
            out[b * 32 + i] = f32_to_f16(q * d);
        }
    }
    out
}

fn cpu_deq256_q5_0(blocks: &[u8]) -> [u16; 256] {
    let mut out = [0u16; 256];
    for b in 0..8usize {
        let blk = &blocks[b * 22..(b + 1) * 22];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        for i in 0..16usize {
            let lo = (qs[i] & 0xF) as u32 | (((qh >> i) & 1) << 4);
            let hi = (qs[i] >> 4) as u32 | (((qh >> (i + 16)) & 1) << 4);
            out[b * 32 + i] = f32_to_f16(((lo as i32) - 16) as f32 * d);
            out[b * 32 + 16 + i] = f32_to_f16(((hi as i32) - 16) as f32 * d);
        }
    }
    out
}

fn cpu_deq256_bf16(raw: &[u8]) -> [u16; 256] {
    let mut out = [0u16; 256];
    for i in 0..256usize {
        let bf = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        out[i] = f32_to_f16(bf16_to_f32(bf));
    }
    out
}

#[test]
fn deq16_matches_cpu_reference_for_all_supported_weight_types() {
    let harness = format!(
        "{}\n{}",
        metal_src(),
        r#"
kernel void k_test_deq16(device const uchar * W [[buffer(0)]],
                         device half * out [[buffer(1)]],
                         constant uint & wtype [[buffer(5)]],
                         uint tid [[thread_position_in_grid]]) {
    half tmp[16];
    deq16(W, wtype, tid * 16, tmp);
    for (int i = 0; i < 16; i++) out[tid * 16 + i] = tmp[i];
}
"#
    );
    let gpu = Gpu::new(&harness, &["k_test_deq16"]);
    let mut rng = Rng::new(0xDE01_6000);

    let run_deq = |wtype: u32, raw: &[u8]| -> [u16; 256] {
        let w_buf = gpu.buffer_with(raw);
        let out_buf = gpu.buffer_with(as_bytes(&[0u16; 256]));
        let cb = gpu.cmd();
        let enc = Enc::new(&cb);
        enc.pso(&gpu.pso("k_test_deq16"));
        enc.buf(&w_buf, 0, 0);
        enc.buf(&out_buf, 0, 1);
        enc.bytes(&wtype, 5);
        enc.dispatch((1, 1, 1), 16);
        enc.end();
        cb.commit();
        cb.waitUntilCompleted();
        let v = read_buf_u16(&out_buf, 256);
        let mut arr = [0u16; 256];
        arr.copy_from_slice(&v);
        arr
    };

    // 1. Q4_K (144 bytes = 256 weights)
    let mut q4k = vec![0u8; 144];
    q4k[0..2].copy_from_slice(&f32_to_f16(0.03125).to_le_bytes());
    q4k[2..4].copy_from_slice(&f32_to_f16(0.015625).to_le_bytes());
    for b in &mut q4k[4..] {
        *b = rng.next_u8();
    }
    let got_q4k = run_deq(T_Q4_K, &q4k);
    let ref_q4k = cpu_deq256_q4k(&q4k);
    for i in 0..256 {
        let diff = (f16_to_f32(got_q4k[i]) - f16_to_f32(ref_q4k[i])).abs();
        assert!(
            diff <= 1e-3,
            "Q4_K mismatch at {i}: got={} ({:#06x}) ref={} ({:#06x})",
            f16_to_f32(got_q4k[i]),
            got_q4k[i],
            f16_to_f32(ref_q4k[i]),
            ref_q4k[i]
        );
    }

    // 2. Q6_K (210 bytes = 256 weights)
    let mut q6k = vec![0u8; 210];
    for b in &mut q6k[0..208] {
        *b = rng.next_u8();
    }
    q6k[208..210].copy_from_slice(&f32_to_f16(0.015625).to_le_bytes());
    let got_q6k = run_deq(T_Q6_K, &q6k);
    let ref_q6k = cpu_deq256_q6k(&q6k);
    for i in 0..256 {
        let diff = (f16_to_f32(got_q6k[i]) - f16_to_f32(ref_q6k[i])).abs();
        assert!(
            diff <= 1e-3,
            "Q6_K mismatch at {i}: got={} ({:#06x}) ref={} ({:#06x})",
            f16_to_f32(got_q6k[i]),
            got_q6k[i],
            f16_to_f32(ref_q6k[i]),
            ref_q6k[i]
        );
    }

    // 3. Q8_0 (8 * 34 = 272 bytes = 256 weights)
    let mut q8 = vec![0u8; 8 * 34];
    for b in 0..8 {
        let scale = f32_to_f16((b as f32 + 1.0) * 0.015625);
        q8[b * 34..b * 34 + 2].copy_from_slice(&scale.to_le_bytes());
        for i in 0..32 {
            q8[b * 34 + 2 + i] = rng.next_u8();
        }
    }
    let got_q8 = run_deq(T_Q8_0, &q8);
    let ref_q8 = cpu_deq256_q8_0(&q8);
    for i in 0..256 {
        assert_eq!(
            got_q8[i], ref_q8[i],
            "Q8_0 bit-exact mismatch at {i}: got={:#06x} ref={:#06x}",
            got_q8[i], ref_q8[i]
        );
    }

    // 4. Q5_0 (8 * 22 = 176 bytes = 256 weights)
    let mut q5 = vec![0u8; 8 * 22];
    for b in 0..8 {
        let scale = f32_to_f16((b as f32 + 1.0) * 0.03125);
        q5[b * 22..b * 22 + 2].copy_from_slice(&scale.to_le_bytes());
        for i in 2..22 {
            q5[b * 22 + i] = rng.next_u8();
        }
    }
    let got_q5 = run_deq(T_Q5_0, &q5);
    let ref_q5 = cpu_deq256_q5_0(&q5);
    for i in 0..256 {
        assert_eq!(
            got_q5[i], ref_q5[i],
            "Q5_0 bit-exact mismatch at {i}: got={:#06x} ref={:#06x}",
            got_q5[i], ref_q5[i]
        );
    }

    // 5. BF16 (256 * 2 = 512 bytes = 256 weights)
    let mut bf16 = vec![0u8; 512];
    for i in 0..256 {
        let v = if i == 0 {
            0.0f32
        } else if i == 1 {
            -1.5f32
        } else {
            rng.next_f32() * 4.0
        };
        bf16[2 * i..2 * i + 2].copy_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    let got_bf16 = run_deq(T_BF16, &bf16);
    let ref_bf16 = cpu_deq256_bf16(&bf16);
    for i in 0..256 {
        assert_eq!(
            got_bf16[i], ref_bf16[i],
            "BF16 bit-exact mismatch at {i}: got={:#06x} ref={:#06x}",
            got_bf16[i], ref_bf16[i]
        );
    }

    // 6. Unknown wtype cleanly zero-fills instead of reading uninitialized memory
    let got_unknown = run_deq(999, &bf16);
    assert!(got_unknown.iter().all(|&h| h == 0), "unknown wtype should zero-fill");
}

/// Ensures `deq16` in `src/prefill.metal` has no unreachable duplicate branches
/// for `wtype == 12`, `8`, `6`, or `30` after their fast early-return paths.
#[test]
fn deq16_contains_no_unreachable_duplicate_wtype_branches() {
    let src = include_str!("../src/prefill.metal");
    let start = src
        .find("inline void deq16(")
        .expect("deq16 definition not found in prefill.metal");
    let after = &src[start..];
    let mut depth = 0i32;
    let mut body_end = 0usize;
    let mut seen_open = false;
    for (idx, ch) in after.char_indices() {
        match ch {
            '{' => {
                depth += 1;
                seen_open = true;
            }
            '}' => {
                depth -= 1;
                if seen_open && depth == 0 {
                    body_end = idx + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(body_end > 0, "failed to parse deq16 function body");
    let body = &after[..body_end];

    for (wtype, label) in [
        (12, "Q4_K"),
        (8, "Q8_0"),
        (30, "BF16"),
        (6, "Q5_0"),
        (14, "Q6_K"),
    ] {
        let pat = format!("wtype == {wtype}");
        let count = body.matches(&pat).count();
        assert_eq!(
            count, 1,
            "deq16 must handle {label} (`{pat}`) exactly once (found {count} occurrences)"
        );
    }
    assert!(
        !body.contains("bf16f("),
        "deq16 should not retain unreachable scalar bf16f() fallback after vectorized wtype == 30 early return"
    );
}
