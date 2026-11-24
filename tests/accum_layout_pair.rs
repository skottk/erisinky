//! Regression tests for `cooperative_tensor` accumulator layout detection (`Quad` vs `Pair`)
//! and the M4/Apple9 pair-epilogue prefill kernels (`k_gemm_sg*_p`, `k_moe_sg*_p`, `k_ctrl_gf_p`,
//! `k_attn_h_p`, and `kgmode_0_0`).

use inkling::engine::{classify_att_layout, classify_sg_layout, AccumLayout, E, FF, HD, NH, NKV, NREL};
use inkling::metal::{Enc, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GSeg {
    w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32,
    nrows: u32, k: u32, tile_start: u32, in_off: u32,
    out_off: u32, ldo: u32, wsrc: u32, in2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmArgs {
    nseg: u32, t: u32, pad0: u32, pad1: u32,
    seg: [GSeg; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HSeg {
    w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32,
    nrows: u32, k: u32, tile_start: u32, xh_off: u32,
    out_off: u32, ldo: u32, wsrc: u32, hout: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmHArgs {
    nseg: u32, t: u32, pad0: u32, pad1: u32,
    seg: [HSeg; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MoeGemmArgs {
    w_lo: u32, w_hi: u32, expert_bytes: u32, row_bytes: u32,
    wtype: u32, nrows: u32, k: u32, x_off: u32,
    gather: u32, perm_off: u32, tiles_off: u32, ntiles_off: u32,
    out_off: u32, ldo: u32, situ: u32, pad: u32,
    beta: f32, lbeta: f32, pad2: f32, pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MoeGemmHArgs {
    w_lo: u32, w_hi: u32, expert_bytes: u32, row_bytes: u32,
    wtype: u32, nrows: u32, k: u32, xh_off: u32,
    tiles_off: u32, ntiles_off: u32, out_off: u32, ldo: u32,
    situ: u32, aw_off: u32, beta: f32, lbeta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CtrlGfJob {
    gain_w: u32, forget_w: u32, gh_off: u32, u_off: u32,
    m_off: u32, raw_off: u32, pad0: u32, pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CtrlGfArgs {
    t: u32, njobs: u32, pad0: u32, pad1: u32,
    job: [CtrlGfJob; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AttnBArgs {
    t: u32, p0: u32, ring: u32, cap: u32,
    k_base: u32, v_base: u32, window: u32, qp_off: u32,
    rp_off: u32, sink_off: u32, gate_off: u32, out_off: u32,
    relbias_w: u32, ghalf: u32, kpos0: i32, pad: u32,
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e <= 0 { return sign; }
    if e >= 31 { return sign | 0x7c00; }
    let round_bit = (mant >> 12) & 1;
    let sticky = mant & 0x0fff;
    let mut m = (mant >> 13) as u16;
    if round_bit != 0 && (sticky != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x0400 {
            return sign | (((e + 1) as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | m
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 { f32::from_bits(sign) } else {
            let v = (mant as f32) * 2.0f32.powi(-24);
            if sign != 0 { -v } else { v }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

fn make_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn synth_sg_quad() -> Vec<i32> {
    let mut o = vec![-1i32; 32 * 64];
    for t in 0..32usize {
        o[t * 64 + 63] = 32;
        let tok0 = (t % 16) as i32;
        let grp = (t / 16) as i32;
        for q in 0..4usize {
            let r0 = grp * 16 + (q as i32) * 4;
            for e in 0..4usize {
                o[t * 64 + 4 * q + e] = ((r0 + e as i32) << 16) | tok0;
                o[t * 64 + 4 * (q + 4) + e] = ((r0 + 32 + e as i32) << 16) | tok0;
            }
        }
    }
    o
}

fn synth_sg_pair() -> Vec<i32> {
    let mut o = vec![-1i32; 32 * 64];
    for t in 0..32usize {
        o[t * 64 + 63] = 32;
        let rbase = (((t & 8) >> 1) | ((t & 1) << 1)) as i32;
        let tbase = (((t & 16) >> 2) | ((t & 6) >> 1)) as i32;
        for h in 0..2usize {
            let tok = tbase + (h as i32) * 8;
            for p in 0..8usize {
                let r = rbase + (p as i32) * 8;
                let idx = h * 16 + 2 * p;
                o[t * 64 + idx] = (r << 16) | tok;
                o[t * 64 + idx + 1] = ((r + 1) << 16) | tok;
            }
        }
    }
    o
}

fn synth_att_quad() -> Vec<i32> {
    let mut o = vec![-1i32; 256 * 64];
    for t in 0..256usize {
        o[t * 64 + 63] = 16;
        let row = (t % 32) as i32;
        let sg = (t / 32) as i32;
        for q in 0..4usize {
            let d0 = sg * 16 + (q as i32) * 4;
            for e in 0..4usize {
                o[t * 64 + 4 * q + e] = ((d0 + e as i32) << 16) | row;
            }
        }
    }
    o
}

fn synth_att_pair() -> Vec<i32> {
    let mut o = vec![-1i32; 256 * 64];
    for t in 0..256usize {
        o[t * 64 + 63] = 16;
        let sg = t / 32;
        let lane = t % 32;
        let dbase = ((sg % 4) * 8 + ((lane & 8) >> 1) + ((lane & 1) << 1)) as i32;
        let rbase = ((sg / 4) * 8 + ((lane & 16) >> 2) + ((lane & 6) >> 1)) as i32;
        for h in 0..2usize {
            let row = rbase + (h as i32) * 16;
            for p in 0..4usize {
                let d = dbase + (p as i32) * 32;
                let idx = h * 8 + 2 * p;
                o[t * 64 + idx] = (d << 16) | row;
                o[t * 64 + idx + 1] = ((d + 1) << 16) | row;
            }
        }
    }
    o
}

#[test]
fn classify_sg_and_att_pure_unit_tests() {
    assert_eq!(classify_sg_layout(&synth_sg_quad()), AccumLayout::Quad);
    assert_eq!(classify_sg_layout(&synth_sg_pair()), AccumLayout::Pair);
    assert_eq!(classify_att_layout(&synth_att_quad()), AccumLayout::Quad);
    assert_eq!(classify_att_layout(&synth_att_pair()), AccumLayout::Pair);

    // Truncated buffers rejected
    assert_eq!(classify_sg_layout(&[0; 10]), AccumLayout::Unsupported);
    assert_eq!(classify_att_layout(&[0; 10]), AccumLayout::Unsupported);

    // Corrupting capacity rejects
    let mut bad = synth_sg_pair();
    bad[63] = 16;
    assert_eq!(classify_sg_layout(&bad), AccumLayout::Unsupported);

    // Breaking r+32 pairing rejects
    let mut bad = synth_sg_pair();
    bad[8] = (34 << 16) | 0;
    assert_eq!(classify_sg_layout(&bad), AccumLayout::Unsupported);

    // Breaking shared query row within an 8-element half rejects
    let mut bad = synth_att_pair();
    bad[2] = (32 << 16) | 1;
    assert_eq!(classify_att_layout(&bad), AccumLayout::Unsupported);

    // Odd row/dim start rejects
    let mut bad = synth_att_quad();
    bad[0] = (1 << 16) | 0;
    assert_eq!(classify_att_layout(&bad), AccumLayout::Unsupported);
}

fn gpu_src(kv8: bool) -> String {
    format!(
        "{}{}\n{}\n{}",
        if kv8 { "#define KV8 1\n" } else { "" },
        include_str!("../src/kernels.metal"),
        include_str!("../src/prefill.metal"),
        include_str!("../src/bench_kernels.metal"),
    )
}

fn probe_gpu_layout(gpu: &Gpu) -> AccumLayout {
    let buf_a = gpu.buffer(256 * 64 * 4);
    let buf_kv = gpu.buffer(128 * 128 * 2);
    let zero = 0u32;

    let sg = {
        let cb = gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&gpu.pso("k_layout_sg"));
        e.buf(&buf_a, 0, 2);
        e.bytes(&zero, 5);
        e.dispatch((1, 1, 1), 32);
        e.end(); cb.commit(); cb.waitUntilCompleted();
        let o = unsafe { std::slice::from_raw_parts(buf_a.contents().as_ptr() as *const i32, 32 * 64) };
        classify_sg_layout(o)
    };
    let att = {
        let cb = gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&gpu.pso("k_layout_att"));
        e.buf(&buf_a, 0, 2);
        e.buf(&buf_kv, 0, 3);
        e.bytes(&zero, 5);
        e.dispatch((1, 1, 1), 256);
        e.end(); cb.commit(); cb.waitUntilCompleted();
        let o = unsafe { std::slice::from_raw_parts(buf_a.contents().as_ptr() as *const i32, 256 * 64) };
        classify_att_layout(o)
    };
    assert_eq!(sg, att, "k_layout_sg ({sg:?}) and k_layout_att ({att:?}) must agree");
    sg
}

#[test]
fn gpu_detected_layout_kernels_match_reference() {
    let gpu = Gpu::new(&gpu_src(false), &[]);
    let lay = probe_gpu_layout(&gpu);
    assert_ne!(lay, AccumLayout::Unsupported, "Apple Silicon Metal 4 GPU should match Quad or Pair");

    // Also verify KV8 layout matches fp16 layout
    let gpu_kv8 = Gpu::new(&gpu_src(true), &[]);
    assert_eq!(probe_gpu_layout(&gpu_kv8), lay, "KV8 layout must match fp16 layout");

    let pair = lay == AccumLayout::Pair;
    let dummy_pb = gpu.buffer(4096);

    // 1. k_gemm_sg4 / k_gemm_sg8 (hout=1) vs k_gemm_t on unaligned token counts (T=1, 19, 131)
    for &t in &[1usize, 19, 131] {
        let rows = 128usize;
        let k = 256usize;
        let row_bytes = k * 2; // BF16 (wtype = 30)
        let buf_w = gpu.buffer(rows * row_bytes);
        unsafe {
            let wp = buf_w.contents().as_ptr() as *mut u16;
            for i in 0..rows * k {
                let v = (((i * 7) % 17) as i32 - 8) as f32 * 0.05;
                *wp.add(i) = make_bf16(v);
            }
        }
        let xf_off = 0u32;
        let xh_off = (t * k) as u32;
        let ref_off = xh_off + (t * k).div_ceil(2) as u32;
        let got_off = ref_off + (t * rows) as u32;
        let tot = got_off as usize + (t * rows).div_ceil(2);
        let buf_a = gpu.buffer(tot * 4);
        unsafe {
            let af = buf_a.contents().as_ptr() as *mut f32;
            let ah = af.add(xh_off as usize) as *mut u16;
            for i in 0..t * k {
                let v = (((i * 5) % 13) as i32 - 6) as f32 * 0.1;
                *af.add(xf_off as usize + i) = v;
                *ah.add(i) = f32_to_f16(v);
            }
        }
        let mut ga = GemmArgs { nseg: 1, t: t as u32, ..Default::default() };
        ga.seg[0] = GSeg {
            w_lo: 0, w_hi: 0, row_bytes: row_bytes as u32, wtype: 30,
            nrows: rows as u32, k: k as u32, tile_start: 0, in_off: xf_off,
            out_off: ref_off, ldo: rows as u32, wsrc: 0, in2: 0,
        };
        let mut ha = GemmHArgs { nseg: 1, t: t as u32, ..Default::default() };
        ha.seg[0] = HSeg {
            w_lo: 0, w_hi: 0, row_bytes: row_bytes as u32, wtype: 30,
            nrows: rows as u32, k: k as u32, tile_start: 0, xh_off,
            out_off: got_off, ldo: rows as u32, wsrc: 0, hout: 1,
        };
        let sg_name = match (t > 64, pair) {
            (true, true) => "k_gemm_sg8_p",
            (true, false) => "k_gemm_sg8",
            (false, true) => "k_gemm_sg4_p",
            (false, false) => "k_gemm_sg4",
        };
        let cb = gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&buf_w, 0, 0); e.buf(&dummy_pb, 0, 1); e.buf(&buf_a, 0, 2);
        e.pso(&gpu.pso("k_gemm_t")); e.bytes(&ga, 5);
        e.dispatch((rows / 64, t.div_ceil(32), 1), 128);
        e.barrier();
        e.pso(&gpu.pso(sg_name)); e.bytes(&ha, 5);
        if t > 64 { e.dispatch((t.div_ceil(128), rows / 64, 1), 256); }
        else { e.dispatch((1, rows / 64, 1), 128); }
        e.end(); cb.commit(); cb.waitUntilCompleted();

        unsafe {
            let af = buf_a.contents().as_ptr() as *const f32;
            let got = af.add(got_off as usize) as *const u16;
            let mut max_diff = 0.0f32;
            for i in 0..t * rows {
                let exp_h = f16_to_f32(f32_to_f16(*af.add(ref_off as usize + i)));
                let got_h = f16_to_f32(*got.add(i));
                max_diff = max_diff.max((exp_h - got_h).abs());
            }
            assert!(max_diff <= 1e-3, "k_gemm_sg ({sg_name}, T={t}) max_diff={max_diff}");
        }
    }

    // 2. k_moe_sg4 / k_moe_sg8 (both situ=1 and situ=0) vs k_moe_gemm_t
    for &cnt in &[17usize, 73] {
        let k = 256usize;
        let row_bytes = k * 2; // BF16
        let exp_bytes = 2 * FF * row_bytes;
        let buf_w = gpu.buffer(exp_bytes);
        unsafe {
            let wp = buf_w.contents().as_ptr() as *mut u16;
            for i in 0..(2 * FF * k) {
                let v = (((i * 11) % 19) as i32 - 9) as f32 * 0.02;
                *wp.add(i) = make_bf16(v);
            }
        }
        let xf_off = 0u32;
        let xh_off = (cnt * k) as u32;
        let perm_off = xh_off + (cnt * k).div_ceil(2) as u32;
        let aw_off = perm_off + cnt as u32;
        let t32_off = aw_off + cnt as u32;
        let nt32_off = t32_off + 16;
        let th_off = nt32_off + 4;
        let nth_off = th_off + 4;
        let oref_off = nth_off + 4;
        let ogot_off = oref_off + (cnt * 2 * FF) as u32;
        let tot = ogot_off as usize + (cnt * 2 * FF).div_ceil(2);
        let buf_a = gpu.buffer(tot * 4);
        let ntiles32 = cnt.div_ceil(32);
        unsafe {
            let af = buf_a.contents().as_ptr() as *mut f32;
            let ah = af.add(xh_off as usize) as *mut u16;
            for i in 0..cnt * k {
                let v = (((i * 3) % 15) as i32 - 7) as f32 * 0.08;
                *af.add(xf_off as usize + i) = v;
                *ah.add(i) = f32_to_f16(v);
            }
            let perm = af.add(perm_off as usize) as *mut u32;
            for i in 0..cnt {
                *perm.add(i) = i as u32;
                *af.add(aw_off as usize + i) = 0.25 + (i % 4) as f32 * 0.1;
            }
            let t32 = af.add(t32_off as usize) as *mut u32;
            for j in 0..ntiles32 {
                *t32.add(j * 4) = 0;
                *t32.add(j * 4 + 1) = (j * 32) as u32;
                *t32.add(j * 4 + 2) = 32.min(cnt - j * 32) as u32;
                *t32.add(j * 4 + 3) = 0;
            }
            *(af.add(nt32_off as usize) as *mut u32) = ntiles32 as u32;
            let th = af.add(th_off as usize) as *mut u32;
            *th.add(0) = 0; *th.add(1) = 0; *th.add(2) = cnt as u32; *th.add(3) = 0;
            *(af.add(nth_off as usize) as *mut u32) = 1;
        }
        let moe_name = match (cnt > 64, pair) {
            (true, true) => "k_moe_sg8_p",
            (true, false) => "k_moe_sg8",
            (false, true) => "k_moe_sg4_p",
            (false, false) => "k_moe_sg4",
        };
        let tg = if cnt > 64 { 256 } else { 128 };
        for &situ in &[1u32, 0u32] {
            let ldo = if situ == 1 { FF as u32 } else { (2 * FF) as u32 };
            let y_tiles = if situ == 1 { FF / 32 } else { (2 * FF) / 64 };
            let ma = MoeGemmArgs {
                w_lo: 0, w_hi: 0, expert_bytes: exp_bytes as u32, row_bytes: row_bytes as u32,
                wtype: 30, nrows: (2 * FF) as u32, k: k as u32, x_off: xf_off,
                gather: 1, perm_off, tiles_off: t32_off, ntiles_off: nt32_off,
                out_off: oref_off, ldo, situ, pad: 0, beta: 2.0, lbeta: 4.0, pad2: 0.0, pad3: 0.0,
            };
            let mha = MoeGemmHArgs {
                w_lo: 0, w_hi: 0, expert_bytes: exp_bytes as u32, row_bytes: row_bytes as u32,
                wtype: 30, nrows: (2 * FF) as u32, k: k as u32, xh_off,
                tiles_off: th_off, ntiles_off: nth_off, out_off: ogot_off, ldo,
                situ, aw_off, beta: 2.0, lbeta: 4.0,
            };
            let cb = gpu.cmd();
            let e = Enc::new(&cb);
            e.buf(&buf_w, 0, 0); e.buf(&buf_a, 0, 2);
            e.pso(&gpu.pso("k_moe_gemm_t")); e.bytes(&ma, 5);
            e.dispatch((y_tiles, ntiles32, 1), 128);
            e.barrier();
            e.pso(&gpu.pso(moe_name)); e.bytes(&mha, 5);
            e.dispatch((1, y_tiles, 1), tg);
            e.end(); cb.commit(); cb.waitUntilCompleted();

            unsafe {
                let af = buf_a.contents().as_ptr() as *const f32;
                let got = af.add(ogot_off as usize) as *const u16;
                let mut max_diff = 0.0f32;
                for tok in 0..cnt {
                    let scale = if situ == 1 { *af.add(aw_off as usize + tok) } else { 1.0 };
                    for col in 0..ldo as usize {
                        let idx = tok * ldo as usize + col;
                        let exp_h = f16_to_f32(f32_to_f16(*af.add(oref_off as usize + idx) * scale));
                        let got_h = f16_to_f32(*got.add(idx));
                        max_diff = max_diff.max((exp_h - got_h).abs());
                    }
                }
                assert!(max_diff <= 1e-3, "k_moe_sg ({moe_name}, cnt={cnt}, situ={situ}) max_diff={max_diff}");
            }
        }
    }

    // 3. k_ctrl_gf vs exact CPU reference on unaligned T=37
    {
        let t = 37usize;
        let buf_pb = gpu.buffer(2 * E * 64 * 2);
        let mut gain_w = vec![0.0f32; E * 64];
        let mut forget_w = vec![0.0f32; E * 64];
        unsafe {
            let p = buf_pb.contents().as_ptr() as *mut u16;
            for i in 0..E * 64 {
                let vg = make_bf16((((i * 7) % 13) as i32 - 6) as f32 * 0.04);
                let vf = make_bf16((((i * 11) % 15) as i32 - 7) as f32 * 0.04);
                *p.add(i) = vg;
                *p.add(E * 64 + i) = vf;
                gain_w[i] = bf16_to_f32(vg);
                forget_w[i] = bf16_to_f32(vf);
            }
        }
        let gh_off = 0u32;
        let u_off = (t * 64 / 2) as u32;
        let m_off = u_off + (t * E) as u32;
        let raw_off = m_off + (t * E) as u32;
        let buf_a = gpu.buffer((raw_off as usize + t * E) * 4);
        let mut g_in = vec![0.0f32; t * 64];
        unsafe {
            let af = buf_a.contents().as_ptr() as *mut f32;
            let gh = af.add(gh_off as usize) as *mut u16;
            for i in 0..t * 64 {
                let h = f32_to_f16((((i * 5) % 11) as i32 - 5) as f32 * 0.1);
                *gh.add(i) = h;
                g_in[i] = f16_to_f32(h);
            }
            for i in 0..t * E {
                *af.add(u_off as usize + i) = (((i * 3) % 17) as i32 - 8) as f32 * 0.1;
                *af.add(m_off as usize + i) = (((i * 7) % 19) as i32 - 9) as f32 * 0.1;
            }
        }
        let mut ca = CtrlGfArgs { t: t as u32, njobs: 1, ..Default::default() };
        ca.job[0] = CtrlGfJob {
            gain_w: 0, forget_w: (E * 64 * 2) as u32, gh_off, u_off, m_off, raw_off, pad0: 0, pad1: 0,
        };
        let cname = if pair { "k_ctrl_gf_p" } else { "k_ctrl_gf" };
        let cb = gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&buf_pb, 0, 1); e.buf(&buf_a, 0, 2);
        e.pso(&gpu.pso(cname)); e.bytes(&ca, 5);
        e.dispatch((t.div_ceil(128), E / 32, 1), 256);
        e.end(); cb.commit(); cb.waitUntilCompleted();

        unsafe {
            let af = buf_a.contents().as_ptr() as *const f32;
            let mut max_diff = 0.0f32;
            for tok in 0..t {
                for r in 0..E {
                    let mut gn = 0.0f32;
                    let mut fg = 0.0f32;
                    for kk in 0..64 {
                        let gx = g_in[tok * 64 + kk];
                        gn += gx * gain_w[r * 64 + kk];
                        fg += gx * forget_w[r * 64 + kk];
                    }
                    let u = *af.add(u_off as usize + tok * E + r);
                    let m = *af.add(m_off as usize + tok * E + r);
                    let expv = u - fg.tanh() * u + gn.tanh() * m;
                    let gotv = *af.add(raw_off as usize + tok * E + r);
                    max_diff = max_diff.max((expv - gotv).abs());
                }
            }
            assert!(max_diff <= 1e-4, "k_ctrl_gf ({cname}) max_diff={max_diff}");
        }
    }

    // 4. k_attn_h vs k_attn_bt on unaligned T=37 (crossing 32-query tile boundary)
    {
        let t = 37usize;
        let cap = 64usize;
        let qf_off = 0u32;
        let qh_off = (t * NH * HD) as u32;
        let rp_off = qh_off + (t * NH * HD / 2) as u32;
        let sk_off = rp_off + (t * NH * NREL) as u32;
        let gf_off = sk_off + (t * NH) as u32;
        let gh_off = gf_off + (t * NH * HD) as u32;
        let obt_off = gh_off + (t * NH * HD / 2) as u32;
        let oh_off = obt_off + (t * NH * HD) as u32;
        let buf_a = gpu.buffer((oh_off as usize + t * NH * HD / 2) * 4);
        let buf_kv = gpu.buffer(2 * NKV * cap * HD * 2);
        let buf_rel = gpu.buffer((1024 + 128) * NREL * 4);
        unsafe {
            let af = buf_a.contents().as_ptr() as *mut f32;
            let qh = af.add(qh_off as usize) as *mut u16;
            let gh = af.add(gh_off as usize) as *mut u16;
            for i in 0..t * NH * HD {
                let qv = (((i * 7) % 17) as i32 - 8) as f32 * 0.05;
                *af.add(qf_off as usize + i) = qv;
                *qh.add(i) = f32_to_f16(qv);
                let gv = (((i * 11) % 13) as i32 - 6) as f32 * 0.2;
                *af.add(gf_off as usize + i) = gv;
                *gh.add(i) = f32_to_f16(gv);
            }
            for i in 0..t * NH * NREL {
                *af.add(rp_off as usize + i) = (((i * 3) % 11) as i32 - 5) as f32 * 0.05;
            }
            for i in 0..t * NH {
                *af.add(sk_off as usize + i) = 0.1;
            }
            let kv = buf_kv.contents().as_ptr() as *mut u16;
            for i in 0..2 * NKV * cap * HD {
                *kv.add(i) = f32_to_f16((((i * 5) % 19) as i32 - 9) as f32 * 0.05);
            }
            let rel = buf_rel.contents().as_ptr() as *mut f32;
            for i in 0..1024 * NREL {
                *rel.add(i) = (((i * 3) % 13) as i32 - 6) as f32 * 0.02;
            }
            for i in 1024 * NREL..(1024 + 128) * NREL {
                *rel.add(i) = 0.0;
            }
        }
        let ab = AttnBArgs {
            t: t as u32, p0: 0, ring: 0, cap: cap as u32, k_base: 0, v_base: (NKV * cap * HD) as u32,
            window: 0, qp_off: qf_off, rp_off, sink_off: sk_off, gate_off: gf_off, out_off: obt_off,
            relbias_w: 0, ghalf: 0, kpos0: 0, pad: 0,
        };
        let ah = AttnBArgs {
            qp_off: qh_off, gate_off: gh_off, out_off: oh_off, ghalf: 1, ..ab
        };
        let aname = if pair { "k_attn_h_p" } else { "k_attn_h" };
        let cb = gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&buf_rel, 0, 1); e.buf(&buf_a, 0, 2); e.buf(&buf_kv, 0, 3);
        e.pso(&gpu.pso("k_attn_bt")); e.bytes(&ab, 5);
        e.dispatch((t.div_ceil(32), NH, 1), 256);
        e.barrier();
        e.pso(&gpu.pso(aname)); e.bytes(&ah, 5);
        e.dispatch((t.div_ceil(32), NH, 1), 256);
        e.end(); cb.commit(); cb.waitUntilCompleted();

        unsafe {
            let af = buf_a.contents().as_ptr() as *const f32;
            let got = af.add(oh_off as usize) as *const u16;
            let mut max_diff = 0.0f32;
            for i in 0..t * NH * HD {
                let r = *af.add(obt_off as usize + i);
                let g = f16_to_f32(*got.add(i));
                max_diff = max_diff.max((r - g).abs());
            }
            assert!(max_diff <= 2e-3, "k_attn_h ({aname}) max_diff={max_diff}");
        }
    }
}
