//! Inkling 2 Mini decode engine: weight binding, GPU memory layout and the per-token dispatch schedule.

use crate::gguf::*;
use crate::metal::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use std::cell::{Cell, RefCell};
use std::os::unix::fs::FileExt;

pub const E: usize = 2304;
pub const NH: usize = 24;
pub const NKV: usize = 8;
pub const HD: usize = 128;
pub const NREL: usize = 16;
pub const EXT: usize = 1024;
pub const NEXP: usize = 128;
pub const TOPK: usize = 8;
pub const FF: usize = 1152;
pub const NCH: usize = 2 * NKV * HD + 2 * E; // conv-state channels per slot
pub const N_PHYS: usize = 24;
pub const LOOP_START: usize = 6;
pub const LOOP_END: usize = 18;
pub const N_PASS: usize = 3;
pub const N_SLOT: usize = N_PHYS + (N_PASS - 1) * (LOOP_END - LOOP_START);
pub const BLOCK: usize = 6;
pub const KPS_CHUNK: usize = 32;
/// keys in front of the prefill K/V scratch that the 64-key attention chunks may read (always masked)
const KGUARD: usize = 64;
/// process-wide default for `Engine::load`: keep only part of the (exact, fp16) K/V cache in memory while generating and
/// stream the older part from a temporary file on every token (`--kv-disk`)
pub static KV_DISK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// one position of one slot in the K/V file: K then V, 8 groups of 128 fp16 each
const REC: usize = 2 * NKV * HD * 2;

/// The disk tier of the K/V cache (`--kv-disk`). While generating, the two cache buffers are rings holding only the newest
/// positions ("hot"); every older position a layer still attends to is in `file` and is read into `stg` for each token.
struct Tier {
    file: std::fs::File,
    on: Cell<bool>,
    /// per kind [sliding-window, full]: oldest position the hot ring holds / positions below this are in the file
    lo: [Cell<usize>; 2], cold: [Cell<usize>; 2],
    /// positions below this were written to the file earlier in this session (cached K/V never change, so a later tiering
    /// after a prefill only writes what is new)
    filed: [Cell<usize>; 2],
    /// position at which the sizes are reviewed again (needs an idle GPU)
    next: Cell<usize>,
    /// smallest context that is tiered
    min_n: usize,
    /// staging for the file reads, in two halves: the GPU works on one while the next reads land in the other
    stg: RefCell<Buf>, half: Cell<usize>, pending: RefCell<[Option<CmdBuf>; 2]>,
    /// this step's file reads in execution order, and the round (staging half fill) that is loaded
    plan: RefCell<Vec<Chunk>>, execs: RefCell<Vec<ExecPlan>>, loaded: Cell<usize>,
    pub read_bytes: Cell<u64>, pub read_time: Cell<std::time::Duration>, unaligned: Cell<bool>,
}
#[derive(Clone, Copy)]
struct Chunk { exec: usize, kind: usize, i: usize, c0: usize, c1: usize, round: usize, off: usize, sp0: usize, ns: usize, kps: usize }
/// how one layer execution's keys are divided: file chunks cover the splits before `b`, the hot ring the `ns` splits from `sp0`
#[derive(Clone, Copy)]
struct ExecPlan { b: usize, kps: usize, nsplit: usize, sp0: usize, ns: usize }

/// the encoder a decode step is being written to; with the disk tier a step spans several command buffers
pub struct Cx { e: Enc, cb: Option<CmdBuf> }

/// process-wide default for `Engine::load`: the opt-in 8-bit K/V cache (`--kv8`)
pub static KV8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// initial capacity (positions) of the K/V buffers; they grow with the context
const KV_CAP0: usize = 64;
/// MoE tile classes of the fp16 prefill path: (kernel, max tokens per tile, threads per threadgroup)
pub const NTC: usize = 2;
const MOE_CLS: [(&str, usize, usize); NTC] = [("k_moe_sg8", 128, 256), ("k_moe_sg4", 64, 128)];
pub const MAX_SPLIT: usize = 256;

/// kernel classes, for skip-mask profiling
pub const CLS_NAMES: &[&str] = &["embed", "mix_attn", "qkv", "attn", "combine", "wo", "mix_ffn", "router", "moe_gu", "moe_down",
                                 "ctrl_down", "ctrl_apply", "out_mix", "lm_head", "argmax",
                                 "kvprep", "wo_post", "gather", "moe_comb", "ffn_post", "ctrl_norm", "topk_route"];
const C_KVPREP: u32 = 15; const C_WOPOST: u32 = 16; const C_GATHER: u32 = 17; const C_COMB2: u32 = 18; const C_FFNPOST: u32 = 19;
const C_CTRLN: u32 = 20; const C_TOPK: u32 = 21;
const C_EMBED: u32 = 0; const C_MIXA: u32 = 1; const C_QKV: u32 = 2; const C_ATTN: u32 = 3; const C_COMB: u32 = 4; const C_WO: u32 = 5;
const C_MIXF: u32 = 6; const C_ROUTER: u32 = 7; const C_GU: u32 = 8; const C_DOWN: u32 = 9; const C_CTRLD: u32 = 10; const C_CTRLA: u32 = 11;
const C_OMIX: u32 = 12; const C_LMHEAD: u32 = 13; const C_ARGMAX: u32 = 14;

pub const KERNELS: &[&str] = &[
    "k_embed", "k_mix", "k_mv", "k_ctrl_down", "k_ctrl_apply", "k_attn", "k_attn_combine",
    "k_moe_gu", "k_moe_down", "k_argmax1", "k_argmax2",
    "k_topk", "k_norm",
    "k_gemm", "k_moe_gemm", "k_gemm_t", "k_moe_gemm_t", "k_topk_b", "k_moe_route", "k_moe_comb_b", "k_postconv_b", "k_state_upd", "k_kvprep_b", "k_attn_b", "k_attn_bt", "k_ctrl_apply_b", "k_tohalf", "k_gemm_sg8", "k_gemm_sg4", "k_moe_sg8", "k_moe_sg4", "k_ctrl_gf", "k_moe_route3", "k_moe_comb_h", "k_attn_h", "k_kvcopy", "k_mix_b_1_1", "k_mix_b_2_1", "k_mix_b_3_1", "k_mix_b_4_1", "k_mix_b_5_1", "k_mix_b_6_1", "k_mix_b_1_3", "k_mix_b_2_3", "k_mix_b_3_3", "k_mix_b_4_3", "k_mix_b_5_3", "k_mix_b_6_3", "k_mix_b_1_4", "k_mix_b_2_4", "k_mix_b_3_4", "k_mix_b_4_4", "k_mix_b_5_4", "k_mix_b_6_4", "k_mix_b_id",
    "k_touch",
];

// ----------------------------------------------------------------------------- kernel arg structs (mirror kernels.metal)
#[repr(C)] #[derive(Clone, Copy, Default)]
struct EmbedArgs { emb_lo: u32, emb_hi: u32, row_bytes: u32, norm_off: u32, out0: u32, out1: u32, tok_index: u32, eps: f32 }

#[repr(C)] #[derive(Clone, Copy, Default, Debug)]
pub struct MixJob {
    pub n_s: u32, pub n_q: u32, pub s_off: [u32; 6], pub q_off: i32,
    pub out_off: [u32; 4], pub norm_off: [i32; 4], pub eps: f32, pub tstride: u32,
    pub ncopy: [[i32; 3]; 4], pub ntok: u32,
    pub hcopy: [[i32; 3]; 4], pub hout: [i32; 4], pub hstride: u32, pub nof32: u32,
}

fn mix_b_kernel(j: &MixJob) -> Option<&'static str> {
    if j.q_off < 0 {
        return (j.n_s >= 1).then_some("k_mix_b_id");
    }
    match (j.n_s, j.n_q) {
        (1, 1) => Some("k_mix_b_1_1"), (2, 1) => Some("k_mix_b_2_1"), (3, 1) => Some("k_mix_b_3_1"),
        (4, 1) => Some("k_mix_b_4_1"), (5, 1) => Some("k_mix_b_5_1"), (6, 1) => Some("k_mix_b_6_1"),
        (1, 3) => Some("k_mix_b_1_3"), (2, 3) => Some("k_mix_b_2_3"), (3, 3) => Some("k_mix_b_3_3"),
        (4, 3) => Some("k_mix_b_4_3"), (5, 3) => Some("k_mix_b_5_3"), (6, 3) => Some("k_mix_b_6_3"),
        (1, 4) => Some("k_mix_b_1_4"), (2, 4) => Some("k_mix_b_2_4"), (3, 4) => Some("k_mix_b_3_4"),
        (4, 4) => Some("k_mix_b_4_4"), (5, 4) => Some("k_mix_b_5_4"), (6, 4) => Some("k_mix_b_6_4"),
        _ => None,
    }
}

/// Dispatch depth-attention mix jobs on `e` (buffer 1 = PB, buffer 2 = A must already be bound).
pub fn dispatch_mix_jobs(gpu: &Gpu, e: &Enc, jobs: &[MixJob], t: usize, old_mixb: bool) {
    if jobs.is_empty() || t == 0 { return; }
    // Plain T=1 decode (no fp16 outputs / nof32) runs `k_mix` (768 threads/job, ~2-3x faster
    // than `k_mix_b_*` at t == 1). Batched prefill (t > 1) and fp16 1-token prefill (e.g.
    // 4096k+1 tail chunks) run `k_mix_b_*` so T=1 prefill is bit-identical to batched prefill,
    // while `k_mix` also implements `hout`/`hcopy`/`nof32` for `INK_OLD_MIXB=1` and fallbacks.
    let uses_half = jobs.iter().any(|j| {
        j.nof32 != 0
            || j.hout.iter().any(|&h| h >= 0)
            || j.hcopy.iter().any(|r| r.iter().any(|&h| h >= 0))
    });
    if (t > 1 || uses_half) && !old_mixb && jobs.iter().all(|j| mix_b_kernel(j).is_some()) {
        // independent jobs run concurrently (no barrier between them)
        for j in jobs {
            let j = MixJob { ntok: t as u32, ..*j };
            let name = mix_b_kernel(&j).unwrap();
            e.pso(&gpu.pso(name));
            e.bytes(&j, 5);
            e.dispatch((t.div_ceil(4), 1, 1), 128);
        }
        e.barrier();
        return;
    }
    e.pso(&gpu.pso("k_mix"));
    e.bytes_slice(jobs, 5);
    e.dispatch((jobs.len(), t, 1), 768);
    e.barrier();
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct Seg {
    w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32,
    nrows: u32, k: u32, tg_start: u32, in_off: u32,
    in2_off: i32, norm_off: i32, ss_off: i32, ss_n: i32,
    out_off: u32, epi: u32, res_off: i32, cs_off: u32,
    convw_off: u32, pad: [u32; 3],
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MvArgs { nseg: u32, eps: f32, post: u32, ctr: u32, bias_off: u32, sel_off: u32, wscale: f32, pad: u32, seg: [Seg; 12] }

#[repr(C)] #[derive(Clone, Copy, Default)]
pub struct CtrlJob {
    pub u_off: u32, pub m_off: u32, pub d_off: u32, pub out_off: u32, pub gain_off: u32,
    pub forget_off: u32, pub res_out: u32, pub ss_off: u32, pub norm_w: i32, pub ctr: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
pub struct CtrlArgs { pub n_c: u32, pub tstride: u32, pub eps: f32, pub pad: u32, pub job: [CtrlJob; 4] }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct AttnArgs {
    pos: u32, nkeys: u32, first_pos: u32, ring: u32, cap: u32, k_base: u32, v_base: u32, cs_off: u32,
    q_off: u32, k_off: u32, v_off: u32, r_off: u32, sink_off: u32, kdyn_off: i32, vdyn_off: i32, part_off: u32,
    kconv_w: u32, vconv_w: u32, qclip_w: u32, relclip_w: u32, relbias_w: u32, tau: f32, kps: u32, nsplit: u32,
    ctr: u32, gate_off: u32, out_off: u32, pad: u32,
    sp0: u32, noown: u32, gstr: u32, istr: u32, idx0: u32, lastbias: f32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct DevCopyArgs { src: u32, dst: u32, n: u32, pad: u32 }
/// dev (key eviction experiments): the far keys of the full-attention slots replaced by a selection (`rows` per group,
/// the last one a summary of the evicted keys with score bias `lastbias`); keys from position `r0` on are untouched
pub struct Prune { bag: Buf, rows: usize, r0: usize, lastbias: f32, use_sum: bool }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CombineArgs { part_off: u32, nsplit: u32, gate_off: u32, out_off: u32 }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeArgs {
    logits_off: u32, bias_off: u32, sel_off: u32, x_off: u32,
    norm_off: i32, ss_off: i32, ss_n: i32, eps: f32,
    gu_lo: u32, gu_hi: u32, gu_expert_bytes: u32, gu_row_bytes: u32,
    act_off: u32, wscale: f32, beta: f32, lbeta: f32,
    dn_lo: u32, dn_hi: u32, dn_expert_bytes: u32, dn_row_bytes: u32,
    dn_type: u32, out_off: u32, res_off: i32, cs_off: u32,
    convw_off: u32, pad: [u32; 3],
}

// ---- prefill arg structs (mirror prefill.metal)
#[repr(C)] #[derive(Clone, Copy, Default)]
struct GSeg { w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, tile_start: u32, in_off: u32, out_off: u32, ldo: u32, wsrc: u32, in2: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct ToHalfArgs { src_off: u32, dst_off: u32, k: u32, rows: u32, perm_off: u32, gather: u32, src_stride: u32, dst_stride: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct HSeg { w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, tile_start: u32, xh_off: u32, out_off: u32, ldo: u32, wsrc: u32, hout: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct GemmHArgs { nseg: u32, t: u32, pad: [u32; 2], seg: [HSeg; 10] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeGemmHArgs {
    w_lo: u32, w_hi: u32, expert_bytes: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, xh_off: u32,
    tiles_off: u32, ntiles_off: u32, out_off: u32, ldo: u32, situ: u32, aw_off: u32, beta: f32, lbeta: f32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
pub struct CtrlBJob {
    pub u_off: u32, pub m_off: u32, pub g_off: u32, pub f_off: u32,
    pub out_off: u32, pub norm_w: u32, pub h_off: i32, pub nof32: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlGfJob { gain_w: u32, forget_w: u32, gh_off: u32, u_off: u32, m_off: u32, raw_off: u32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlGfArgs { t: u32, njobs: u32, pad: [u32; 2], job: [CtrlGfJob; 4] }
#[repr(C)] #[derive(Clone, Copy, Default)]
pub struct CtrlBArgs { pub n: u32, pub eps: f32, pub pad: [u32; 2], pub job: [CtrlBJob; 4] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct GemmArgs { nseg: u32, t: u32, pad: [u32; 2], seg: [GSeg; 10] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeGemmArgs {
    w_lo: u32, w_hi: u32, expert_bytes: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, x_off: u32,
    gather: u32, perm_off: u32, tiles_off: u32, ntiles_off: u32, out_off: u32, ldo: u32, situ: u32, pad: u32,
    beta: f32, lbeta: f32, pad2: [f32; 2],
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct TopkBArgs { logits_off: u32, bias_off: u32, sel_off: u32, t: u32, wscale: f32, pad: [u32; 3] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct RouteArgs { sel_off: u32, t: u32, perm_off: u32, inv_off: u32, tiles_off: u32, ntiles_off: u32, tile: u32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct Route3Args { sel_off: u32, t: u32, perm_off: u32, inv_off: u32, aw_off: u32, ind_off: u32, gu_y: u32, dn_y: u32, tiles_off: [u32; NTC], pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeCombArgs { yd_off: u32, inv_off: u32, sel_off: u32, out_off: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct ConvBArgs { y_off: u32, ldy: u32, res_off: u32, out_off: u32, cs_off: u32, convw_off: u32, has_res: u32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct StateArgs { x_off: u32, ldx: u32, cs_off: u32, nch: u32, t: u32, pad: [u32; 3] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct KvPrepArgs {
    t: u32, p0: u32, ring: u32, cap: u32, k_base: u32, v_base: u32, cs_off: u32, q_off: u32,
    k_off: u32, v_off: u32, r_off: u32, kdyn_off: i32, vdyn_off: i32, qp_off: u32, rp_off: u32, qhalf: u32,
    kconv_w: u32, vconv_w: u32, qclip_w: u32, relclip_w: u32, log_alpha: f32, log_floor: f32, is_full: u32, kpos0: i32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct AttnBArgs { t: u32, p0: u32, ring: u32, cap: u32, k_base: u32, v_base: u32, window: u32, qp_off: u32,
                   rp_off: u32, sink_off: u32, gate_off: u32, out_off: u32, relbias_w: u32, ghalf: u32, kpos0: i32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct KvCopyArgs { n: u32, pos0: u32, ring_k: u32, ring_v: u32, cap: u32, sk_off: u32, sv_off: u32, scap: u32, kpos0: i32, to_ring: u32, pad: [u32; 2] }

/// batch (prefill) arena: per-token activation arrays [T][dim], f32 offsets into the same arena buffer
#[derive(Clone, Copy, Default)]
pub struct BArena {
    vec0: u32, q: u32, gate: u32, k: u32, v: u32, rel: u32, sink: u32, kdyn: u32, vdyn: u32, qp: u32, rp: u32, attn_g: u32,
    inp: [u32; 3], msg: [u32; 4], cin: [u32; 3], fin: u32, finc: u32, ctrlg: [u32; 4],
    cat_u: [u32; 4], cat_m: [u32; 4], gain_o: [u32; 4], forget_o: [u32; 4],
    xh: [u32; 3], agh: u32, xgh: u32, acth: u32, cath: [u32; 4], gh: [u32; 4],
    tl: [u32; NTC], ind: u32, aw: u32, fh: u32, sk: u32, sv: u32,
    wo_y: u32, ffn_y: u32, rlog: u32, sel: u32, perm: u32, inv: u32, tiles: u32, ntiles: u32, act: u32, yd: u32, end: u32,
    /// tokens this layout was sized for (0: no batch arena allocated)
    cap: usize,
}
impl BArena {
    /// offsets for a batch of up to `t` tokens. `lean`: the fp16 path, which never touches the f32 intermediates of the
    /// fallback kernels and keeps q / gate / the per-assignment outputs as half.
    fn new(start: u32, t: usize, lean: bool, kvb: usize) -> BArena {
        let mut o = start;
        let mut take = |n: usize| { let r = o; o += ((n + 63) / 64 * 64) as u32; r };
        let fb = |n: usize| if lean { 0 } else { n };      // only the fallback paths use it
        let hf = |n: usize| if lean { n / 2 } else { n };  // half in the fp16 path
        let mut b = BArena { cap: t, ..Default::default() };
        b.vec0 = take(B_NVEC * t * E);
        b.q = take(hf(t * NH * HD)); b.gate = take(hf(t * NH * HD)); b.k = take(t * NKV * HD); b.v = take(t * NKV * HD);
        b.rel = take(t * NH * NREL); b.sink = take(t * NH); b.kdyn = take(t * 32); b.vdyn = take(t * 32);
        b.qp = take(hf(t * NH * HD)); b.rp = take(t * NH * NREL); b.attn_g = take(fb(t * NH * HD));
        for i in 0..3 { b.inp[i] = take(t * E); }
        for i in 0..4 { b.msg[i] = take(t * E); }
        for i in 0..3 { b.cin[i] = take(fb(t * E)); }
        b.fin = take(t * E); b.finc = take(fb(t * E));
        for i in 0..4 { b.cat_u[i] = take(fb(t * E)); b.cat_m[i] = take(fb(t * E)); b.gain_o[i] = take(t * E); b.forget_o[i] = take(fb(t * E)); }
        for i in 0..3 { b.xh[i] = take(t * E / 2); }
        b.agh = take(t * NH * HD / 2);
        b.xgh = take(t * TOPK * E / 2);
        b.acth = take(t * TOPK * FF / 2);
        for i in 0..4 { b.cath[i] = take(t * 2 * E / 2); b.gh[i] = take(t * 64 / 2); }
        for i in 0..4 { b.ctrlg[i] = take(t * 64); }
        b.wo_y = take(t * E); b.ffn_y = take(t * E);
        b.rlog = take(t * NEXP); b.sel = take(t * 16);
        b.tl = [take((t * TOPK / 128 + NEXP + 1) * 4), take((NEXP + 1) * 4)];
        b.ind = take(64); b.aw = take(t * TOPK); b.fh = take(t * E / 2);
        // sliding-window K/V scratch of the batch: [guard | window history | batch keys | guard] per group, half
        b.sk = take(NKV * (EXT + t + 2 * KGUARD) * HD * kvb / 4); b.sv = take(NKV * (EXT + t + 2 * KGUARD) * HD * kvb / 4);
        b.perm = take(t * TOPK); b.inv = take(t * TOPK); b.tiles = take(fb((t * TOPK / 32 + NEXP + 1) * 4)); b.ntiles = take(64);
        b.act = take(fb(t * TOPK * FF)); b.yd = take(hf(t * TOPK * E));
        b.end = o;
        b
    }
    fn vec(&self, i: usize) -> u32 { self.vec0 + (i * self.cap * E) as u32 }
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct ArgmaxArgs { logits_off: u32, n: u32, part_off: u32, nparts: u32, out_index: u32, pad: [u32; 3] }

// ----------------------------------------------------------------------------- weights
#[derive(Clone, Copy, Default, Debug)]
pub struct QW { pub off: u64, pub row_bytes: u32, pub ty: u32, pub rows: u32, pub k: u32, pub expert_bytes: u64 }

fn qw(g: &Gguf, name: &str) -> QW {
    let t = g.tensor(name);
    let rb = t.row_bytes();
    QW { off: t.file_off, row_bytes: rb as u32, ty: t.ty, rows: t.ne[1] as u32, k: t.ne[0] as u32, expert_bytes: rb * t.ne[1] }
}

pub struct Layer {
    pub swa: bool,
    pub wq: QW, pub wgate: QW, pub wk: QW, pub wv: QW, pub wo: QW, pub wrel: QW, pub wsink: QW,
    pub kconvp: Option<QW>, pub vconvp: Option<QW>,
    pub attn_norm: i32, pub q_inp_norm: i32, pub k_inp_norm: i32, pub v_inp_norm: i32,
    pub attn_res_query: i32, pub n_attn_q: u32,
    pub ffn_res_query: u32, pub ffn_norm: u32,
    pub q_clip: u32, pub rel_clip: u32, pub rel_bias: u32, pub k_conv: u32, pub v_conv: u32,
    pub attn_post_conv: u32, pub ffn_post_conv: u32,
    pub router: u32, pub exp_bias: u32,
    pub gu: QW, pub dn: QW,
}

pub struct LoopW { pub msg_query: u32, pub down: [u32; 4], pub gain: [u32; 4], pub forget: [u32; 4] }

struct Pb { data: Vec<u8> }
impl Pb {
    fn align(&mut self) { while !self.data.len().is_multiple_of(256) { self.data.push(0); } }
    fn try_f32(&mut self, g: &Gguf, name: &str) -> std::io::Result<u32> {
        self.align();
        let off = self.data.len() as u32;
        let t = g.tensor(name);
        if t.ty != T_F32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("tensor '{name}' expected F32, got {}", type_name(t.ty)),
            ));
        }
        let raw = g.bytes(t);
        bounds::validate_f32_bytes_finite(name, raw)?;
        self.data.extend_from_slice(raw);
        Ok(off)
    }
    /// F32 -> BF16 (round to nearest even). Lossless for this model's bf16-origin matrices.
    fn try_bf16(&mut self, g: &Gguf, name: &str) -> std::io::Result<u32> {
        self.align();
        let off = self.data.len() as u32;
        let t = g.tensor(name);
        if t.ty != T_F32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("tensor '{name}' expected F32, got {}", type_name(t.ty)),
            ));
        }
        let raw = g.bytes(t);
        bounds::validate_f32_bytes_finite(name, raw)?;
        for &c in raw.as_chunks::<4>().0 {
            let u = u32::from_le_bytes(c);
            let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16;
            self.data.extend_from_slice(&(r as u16).to_le_bytes());
        }
        Ok(off)
    }
}

// ----------------------------------------------------------------------------- activation arena (f32 offsets)
#[derive(Clone, Copy)]
pub struct Arena {
    pub q: u32, pub gate: u32, pub k: u32, pub v: u32, pub rel: u32, pub sink: u32, pub kdyn: u32, pub vdyn: u32,
    pub attn_g: u32, pub inp: [u32; 3], pub msg: [u32; 4], pub cin: [u32; 3], pub ctrlg: [u32; 4], pub ssp: [u32; 4],
    pub cat: [u32; 4], pub fin: u32, pub finc: u32, pub fout: u32, pub rlog: u32, pub sel: u32, pub sel_log: u32, pub act: u32, pub part: u32, pub logits: u32, pub amax: u32, pub total: u32,
}
const N_VEC: usize = 16;
/// hidden-state vectors of the batch arena: at most 4 block states + the current one + 3 pinned + the one being written
const B_NVEC: usize = 10;

impl Arena {
    fn new(max_split: usize, vocab_pad: usize) -> Arena {
        let mut o = (N_VEC * E) as u32;
        let mut take = |n: usize| { let r = o; o += ((n + 63) / 64 * 64) as u32; r };
        let q = take(NH * HD); let gate = take(NH * HD); let k = take(NKV * HD); let v = take(NKV * HD);
        let rel = take(NH * NREL); let sink = take(NH); let kdyn = take(32); let vdyn = take(32);
        let attn_g = take(NH * HD);
        let inp = [take(E), take(E), take(E)];
        let msg = [take(E), take(E), take(E), take(E)];
        let cin = [take(E), take(E), take(E)];
        let ctrlg = [take(64), take(64), take(64), take(64)];
        let ssp = [take(128), take(128), take(128), take(128)];
        let cat = [take(2 * E), take(2 * E), take(2 * E), take(2 * E)];
        let fin = take(E); let finc = take(E); let fout = take(E);
        let rlog = take(NEXP); let sel = take(2 * TOPK); let sel_log = take(64 * 16); let act = take(TOPK * FF);
        let part = take(NH * max_split * (HD + 2));
        let logits = take(vocab_pad);
        let amax = take(2 * 256);
        Arena { q, gate, k, v, rel, sink, kdyn, vdyn, attn_g, inp, msg, cin, ctrlg, ssp, cat, fin, finc, fout, rlog, sel, sel_log, act, part, logits, amax, total: o }
    }
    pub fn vec(i: usize) -> u32 { (i * E) as u32 }
}

// round-robin vector allocator that never hands out a live vector
struct VecAlloc { next: usize, n: usize }
impl VecAlloc {
    fn alloc(&mut self, live: &[usize]) -> usize {
        for _ in 0..self.n {
            let c = self.next;
            self.next = (self.next + 1) % self.n;
            if c != 0 && !live.contains(&c) { return c; }
        }
        panic!("vector pool exhausted");
    }
}

/// Zero-copy read-only guard over the engine's output logits inside `act`.
///
/// Holds the `RefCell<Buf>` borrow for the entire lifetime of the slice so
/// `ensure_batch`, `release_batch`, and `prepare` cannot reallocate and drop
/// the backing `MTLBuffer` while logits are still borrowed.
pub struct Logits<'a> {
    buf: std::cell::Ref<'a, Buf>,
    off: usize,
    len: usize,
}

impl<'a> Logits<'a> {
    /// Construct a bounds-checked `Logits` guard over `buf` starting at `off` `f32` elements
    /// with length `len`.
    pub fn new(buf: std::cell::Ref<'a, Buf>, off: usize, len: usize) -> Self {
        let cap_f32 = buf.length() / std::mem::size_of::<f32>();
        let end = off
            .checked_add(len)
            .expect("logits offset + length overflow");
        assert!(
            end <= cap_f32,
            "logits slice [{off}..{end}) exceeds activation buffer capacity {cap_f32} f32s"
        );
        Self { buf, off, len }
    }
}

impl std::ops::Deref for Logits<'_> {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &[f32] {
        let base = self.buf.contents().as_ptr() as *const f32;
        // SAFETY: `self.buf` keeps the shared `RefCell<Buf>` borrow alive for `&self`,
        // Metal `StorageModeShared` buffers are page-aligned (>= align_of::<f32>()), and
        // `Logits::new` verified `self.off + self.len <= self.buf.length() / 4`.
        unsafe { std::slice::from_raw_parts(base.add(self.off), self.len) }
    }
}

impl AsRef<[f32]> for Logits<'_> {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self
    }
}

/// Enable macOS `F_NOCACHE` (uncached I/O) on an open file descriptor and check the return value.
pub fn set_fd_nocache(fd: std::os::fd::RawFd) -> std::io::Result<()> {
    // SAFETY: `libc::fcntl` with `F_NOCACHE` takes a single `c_int` argument (`1`).
    // Invalid descriptors return `-1` with `EBADF`, which is checked immediately below.
    let rc = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
    if rc == -1 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Create a private (`0o600`), symlink-resistant, immediately-unlinked temporary file
/// in `dir` for the `--kv-disk` KV-cache tier, with `F_NOCACHE` enabled.
pub fn create_kv_disk_file_in(dir: &std::path::Path) -> std::io::Result<std::fs::File> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();

    for _ in 0..64 {
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut rand_bytes = [0u8; 16];
        // SAFETY: `rand_bytes` is a valid, writable 16-byte stack buffer; `arc4random_buf`
        // fills it with CSPRNG bytes and never fails on macOS/BSD.
        unsafe {
            libc::arc4random_buf(rand_bytes.as_mut_ptr() as *mut libc::c_void, rand_bytes.len());
        }
        let hi = u64::from_ne_bytes(rand_bytes[..8].try_into().unwrap());
        let lo = u64::from_ne_bytes(rand_bytes[8..].try_into().unwrap());
        let name = format!("inkling-kv-{pid}-{seq}-{hi:016x}{lo:016x}.bin");
        let path = dir.join(name);

        match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)
        {
            Ok(file) => {
                // Unlink immediately while the FD stays open so no file remains on disk
                // even if the process aborts or crashes.
                std::fs::remove_file(&path)?;
                // Uncached I/O: the point of the file is that its contents stay out of RAM.
                set_fd_nocache(file.as_raw_fd())?;
                return Ok(file);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "failed to create unique kv-disk temporary file after 64 attempts",
    ))
}

pub struct Engine {
    // SAFETY (drop order): Rust drops struct fields in declaration order AFTER `Drop::drop`.
    // 1. `Engine::drop` first waits on all pending tier/GPU command buffers (`gpu.drain()`).
    // 2. All Metal buffers (`w`, `pb`, `cs`, `toks`, `ctr`, `act`, `kv_swa`, `kv_full`, `tier`,
    //    `qcap`, `prune`) drop next — critically releasing `w` (a no-copy `MTLBuffer` aliasing
    //    `g`'s mmap) before `g` is unmapped.
    // 3. `gpu` drops after all buffers (and drains any remaining tracked command buffers).
    // 4. `g: Gguf` MUST remain the LAST field so `Gguf::drop` (`libc::munmap`) executes only
    //    after all GPU work has finished and `w` + `gpu` have been dropped.
    w: Buf, pub pb: Buf, cs: Buf, pub toks: Buf, ctr: Buf,
    /// activations: the small decode arena, followed by the batch (prefill) arena while a prefill needs one
    act: std::cell::RefCell<Buf>,
    /// K/V cache (fp16). Sliding-window layers keep exactly their window (a ring of EXT keys per slot), full-attention
    /// layers every key. Both buffers are sized to the context and grow with it.
    kv_swa: std::cell::RefCell<Buf>, kv_full: std::cell::RefCell<Buf>, swa_cap: std::cell::Cell<usize>, full_cap: std::cell::Cell<usize>,
    /// per slot: (is sliding-window, index among the slots of its kind)
    slot_kind: Vec<(bool, usize)>,
    tier: Option<Tier>,
    /// dev: capture of the full-attention layers' prepared queries and sinks during a prefill (per slot: n x 24 x 128 fp16, n x 24 f32)
    qcap: RefCell<Option<(Buf, usize)>>,
    pub prune: RefCell<Option<Prune>>,
    pub layers: Vec<Layer>,
    loops: Vec<LoopW>,
    tok_embd: QW, tok_norm: u32, output: QW, output_norm: u32, output_res_query: u32,
    pub ar: Arena,
    pub kv8: bool, kvb: usize, ba: std::cell::Cell<BArena>, pub chunk: usize, pub tensor: bool, pub hgemm: bool, sep_comb: bool, sep_topk: bool,
    /// last-layer tail mode of the prefill (dev tools switch it off to profile full layers)
    pub tail: std::cell::Cell<bool>,
    pub skip: std::cell::Cell<u32>,
    pub stop: std::cell::Cell<usize>, done: std::cell::Cell<usize>, pub last_out: std::cell::Cell<usize>, pub last_mid: std::cell::Cell<usize>,
    pub max_ctx: usize,
    pub n_vocab: usize,
    eps: f32, log_alpha: f32, log_floor: f32, situ_beta: f32, situ_lbeta: f32, wscale: f32, pub softcap: f32,
    pub gpu: Gpu,
    /// Backing mmapped GGUF file — MUST stay last so `munmap` runs after `w` and `gpu` drop.
    pub g: Gguf,
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(ref t) = self.tier {
            if let Ok(mut pending) = t.pending.try_borrow_mut() {
                for p in pending.iter_mut() {
                    if let Some(cb) = p.take() {
                        let st = cb.status();
                        if st != objc2_metal::MTLCommandBufferStatus::NotEnqueued
                            && st != objc2_metal::MTLCommandBufferStatus::Completed
                            && st != objc2_metal::MTLCommandBufferStatus::Error
                        {
                            cb.waitUntilCompleted();
                        }
                    }
                }
            }
        }
        self.gpu.drain();
    }
}

pub fn slot_of(pass: usize, ip: usize) -> usize {
    if pass == 0 || !(LOOP_START..LOOP_END).contains(&ip) { ip } else { N_PHYS + (pass - 1) * (LOOP_END - LOOP_START) + (ip - LOOP_START) }
}

/// Pure helpers for benchmark aggregation and dev-tool math (Low-Sev #2 & #4).
pub mod dev_helpers {
    use std::ops::Range;

    /// Select the fastest (maximum) finite bandwidth sample in GB/s.
    pub fn best_bandwidth_gbps(samples: impl IntoIterator<Item = f64>) -> f64 {
        samples
            .into_iter()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .fold(0.0f64, f64::max)
    }

    /// Parse `INK_HEADROOM=a,b` safely, falling back to `(16, 0)` on missing or malformed input.
    pub fn parse_headroom(val: Option<&str>) -> (usize, usize) {
        val.and_then(|v| {
            let (a, b) = v.split_once(',')?;
            Some((a.trim().parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?))
        })
        .unwrap_or((16, 0))
    }

    /// Compute rounded KV-cache reserve capacity without dividing by zero when `hd == 0`.
    pub fn reserve_capacity(need: usize, limit: usize, headroom: bool, hd: usize, hm: usize) -> usize {
        let want = if headroom {
            let frac = if hd == 0 { 0 } else { need / hd };
            need.saturating_add(frac.max(hm))
        } else {
            need
        };
        want.div_ceil(64).saturating_mul(64).min(limit)
    }

    /// Saturating query token range `[n - nq, n)` for `kv_screen`.
    pub fn kv_screen_range(n: usize, nq: usize) -> Range<usize> {
        n.saturating_sub(nq)..n
    }

    /// In-place simulated KV-row quantization for `kv_quant_sim`.
    /// - `bits == 0`: no-op.
    /// - `bits >= 100`: floating format (`100 + m` mantissa bits; `200 + m` with 3-bit exponent).
    /// - `bits == 1`: 1-bit sign quantization (`±max_abs`) without dividing by zero.
    /// - `2..100`: symmetric linear quantization clamped to `<= 24` bits so shifts never overflow.
    pub fn quantize_row_sim(v: &mut [f32], bits: u32) {
        if bits == 0 {
            return;
        }
        if bits >= 100 {
            let (m, lim) = ((bits % 100).min(23) as i32, bits >= 200);
            for x in v.iter_mut() {
                if *x == 0.0 || !x.is_finite() {
                    continue;
                }
                let mut e = x.abs().log2().floor() as i32;
                if lim {
                    e = e.clamp(-2, 4);
                }
                let q = 2f32.powi(e - m);
                if q == 0.0 || !q.is_finite() {
                    continue;
                }
                let mut y = (*x / q).round() * q;
                if lim {
                    y = y.clamp(-(2f32.powi(5) - q), 2f32.powi(5) - q);
                }
                *x = y;
            }
            return;
        }
        let mx = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
        if mx == 0.0 || !mx.is_finite() {
            return;
        }
        if bits == 1 {
            for x in v.iter_mut() {
                if *x != 0.0 {
                    *x = if *x > 0.0 { mx } else { -mx };
                }
            }
            return;
        }
        let eff_bits = bits.min(24);
        let lv = ((1u64 << (eff_bits - 1)) - 1) as f32;
        for x in v.iter_mut() {
            *x = (*x / mx * lv).round() * mx / lv;
        }
    }
}

/// Pure KV-tier math helpers exposed for CPU unit/property testing without a GPU.
pub mod tier_math {
    use super::{EXT, HD, KPS_CHUNK, MAX_SPLIT, NKV, N_SLOT, REC};

    pub fn clamp_tier_piece_bytes(mib: usize) -> usize {
        mib.max(1).saturating_mul(1usize << 20)
    }

    pub fn clamp_tier_threads(n: usize) -> usize {
        n.max(1)
    }

    pub fn clamp_tier_min(n: usize) -> usize {
        n.max(1)
    }

    pub fn clamp_split_target(n: usize) -> usize {
        n.clamp(1, MAX_SPLIT)
    }

    pub fn tier_needs_gpu_drain(upto: usize, done: usize) -> bool {
        upto > done
    }

    pub fn split_kps(nk: usize, split_target: usize) -> usize {
        let st = clamp_split_target(split_target);
        KPS_CHUNK * nk.div_ceil(KPS_CHUNK * st).max(1)
    }

    pub fn kv_exact_bytes(n: usize, n_swa: usize) -> usize {
        let n_swa = n_swa.min(N_SLOT);
        n_swa
            .saturating_mul(n.min(EXT))
            .saturating_add((N_SLOT - n_swa).saturating_mul(n))
            .saturating_mul(REC)
    }

    pub fn tier_targets(
        n: usize,
        n_swa: usize,
        split_target: usize,
        hot_override: Option<usize>,
    ) -> (usize, usize, usize) {
        // Size staging for the next 64-token gen_reserve checkpoint so `half_recs >= split_kps(pos)`
        // stays invariant across every step even above 32K context.
        let s_min = 2usize
            .saturating_mul(split_kps(n.saturating_add(64), split_target))
            .saturating_mul(REC);
        if let Some(h) = hot_override {
            let h = h.max(64);
            return (h, h, (2usize << 20).max(s_min));
        }
        let c = kv_exact_bytes(n, n_swa);
        let s = (((c / 16) >> 20).clamp(2, 32) << 20).max(s_min);
        if c == 0 {
            return (64, 64, s);
        }
        let hot = (c / 2).saturating_sub(s);
        let hs = (((n.min(EXT) as u128 * hot as u128) / c as u128) as usize / 64 * 64).max(64);
        let hf = (((n as u128 * hot as u128) / c as u128) as usize / 64 * 64).max(64);
        (hs, hf, s)
    }

    pub fn plan_read_parts(
        jobs: &[(u64, usize, usize)],
        piece_bytes: usize,
    ) -> Vec<(u64, usize, usize)> {
        let piece = piece_bytes.max(1);
        let mut parts = Vec::new();
        for &(off, bytes, dst) in jobs {
            let mut o = 0usize;
            while o < bytes {
                let n = (bytes - o).min(piece);
                parts.push((off + o as u64, n, dst + o));
                o += n;
            }
        }
        parts
    }

    pub fn tier_foff(kind: usize, i: usize, p: usize, n_swa: usize, max_ctx: usize) -> u64 {
        ((if kind == 0 { i * EXT + p % EXT } else { n_swa * EXT + i * max_ctx + p }) * REC) as u64
    }

    pub fn tier_pieces(kind: usize, i: usize, a: usize, b: usize, n_swa: usize, max_ctx: usize) -> Vec<(u64, usize)> {
        let mut v = Vec::new();
        let mut p = a;
        while p < b {
            let e = if kind == 0 { b.min((p / EXT + 1) * EXT) } else { b };
            v.push((tier_foff(kind, i, p, n_swa, max_ctx), e - p));
            p = e;
        }
        v
    }

    pub fn rows_to_rec(src: &[u8], i: usize, cap: usize, idx: usize, rec: &mut [u8]) {
        assert!(rec.len() >= REC);
        for r in 0..2 * NKV {
            let off = ((i * 2 * NKV + r) * cap + idx) * HD * 2;
            rec[r * HD * 2..(r + 1) * HD * 2].copy_from_slice(&src[off..off + HD * 2]);
        }
    }

    pub fn rec_to_rows(rec: &[u8], dst: &mut [u8], i: usize, cap: usize, idx: usize) {
        assert!(rec.len() >= REC);
        for r in 0..2 * NKV {
            let off = ((i * 2 * NKV + r) * cap + idx) * HD * 2;
            dst[off..off + HD * 2].copy_from_slice(&rec[r * HD * 2..(r + 1) * HD * 2]);
        }
    }
}

/// Pure buffer-bounds validators shared by `Engine` methods and CPU unit tests.
pub mod bounds {
    pub fn validate_set_tokens(
        start: usize,
        toks: &[u32],
        max_ctx: usize,
        vocab_rows: u32,
    ) -> Result<(), String> {
        let cap = max_ctx
            .checked_add(8)
            .ok_or_else(|| format!("set_tokens capacity overflow (max_ctx={max_ctx})"))?;
        let end = start.checked_add(toks.len()).ok_or_else(|| {
            format!(
                "set_tokens range overflow (start={start}, len={})",
                toks.len()
            )
        })?;
        if end > cap {
            return Err(format!(
                "set_tokens range [{start}..{end}) exceeds token buffer capacity {cap} (max_ctx={max_ctx})"
            ));
        }
        for (i, &tok) in toks.iter().enumerate() {
            if tok >= vocab_rows || tok > i32::MAX as u32 {
                return Err(format!(
                    "set_tokens token[{i}]={tok} out of range (vocab_rows={vocab_rows})"
                ));
            }
        }
        Ok(())
    }

    pub fn validate_token_at(idx: usize, max_ctx: usize) -> Result<(), String> {
        let cap = max_ctx
            .checked_add(8)
            .ok_or_else(|| format!("token_at capacity overflow (max_ctx={max_ctx})"))?;
        if idx >= cap {
            return Err(format!(
                "token_at index {idx} out of bounds (capacity {cap}, max_ctx={max_ctx})"
            ));
        }
        Ok(())
    }

    pub fn validate_arena_range(
        off: usize,
        n: usize,
        act_floats: usize,
    ) -> Result<(), String> {
        let end = off
            .checked_add(n)
            .ok_or_else(|| format!("arena offset + length overflow (off={off}, n={n})"))?;
        if end > act_floats {
            return Err(format!(
                "arena slice [{off}..{end}) exceeds buffer capacity {act_floats}"
            ));
        }
        Ok(())
    }

    pub fn validate_restore_kv_len(src_len: usize, dst_len: usize) -> Result<(), String> {
        if src_len > dst_len {
            return Err(format!(
                "restore KV buffer length {src_len} exceeds destination capacity {dst_len}"
            ));
        }
        if src_len != dst_len {
            return Err(format!(
                "restore KV buffer length {src_len} does not match destination capacity {dst_len}"
            ));
        }
        Ok(())
    }

    pub fn validate_restore_range(
        cs_src_len: usize,
        cs_dst_len: usize,
        kv_src_len: usize,
        kv_dst_len: usize,
    ) -> Result<(), String> {
        if cs_src_len > cs_dst_len {
            return Err(format!(
                "restore conv-state length {cs_src_len} exceeds destination capacity {cs_dst_len}"
            ));
        }
        if cs_src_len != cs_dst_len {
            return Err(format!(
                "restore conv-state length {cs_src_len} does not match destination capacity {cs_dst_len}"
            ));
        }
        validate_restore_kv_len(kv_src_len, kv_dst_len)
    }

    pub fn validate_kbench_t(t: usize, chunk: usize) -> Result<(), String> {
        if t == 0 {
            return Err("kbench token count t must be > 0".into());
        }
        if t > chunk {
            return Err(format!(
                "kbench token count t={t} exceeds chunk capacity {chunk}"
            ));
        }
        Ok(())
    }

    pub fn validate_kbench_output_buf(
        t: usize,
        out_rows: usize,
        buf_floats: usize,
    ) -> Result<(), String> {
        let need = t.checked_mul(out_rows).ok_or_else(|| {
            format!("kbench output size overflow (t={t}, out_rows={out_rows})")
        })?;
        if need > buf_floats {
            return Err(format!(
                "kbench output requires {need} floats (t={t} * out_rows={out_rows}) but buffer holds {buf_floats}"
            ));
        }
        Ok(())
    }

    /// Compute `(out_off, out_stride, total_floats)` in 32-bit float units for a dedicated
    /// benchmark scratch buffer holding `t * in_dim` inputs (fp16 if `half_in`, else f32)
    /// followed by `n_out_bufs` output matrices of `t * out_rows` f32s each.
    pub fn kbench_scratch_layout(
        t: usize,
        in_dim: usize,
        out_rows: usize,
        half_in: bool,
        n_out_bufs: usize,
    ) -> Result<(u32, usize, usize), String> {
        if t == 0 || in_dim == 0 || out_rows == 0 || n_out_bufs == 0 {
            return Err(format!(
                "kbench scratch dimensions must be non-zero (t={t}, in_dim={in_dim}, out_rows={out_rows}, n_out_bufs={n_out_bufs})"
            ));
        }
        let in_elems = t
            .checked_mul(in_dim)
            .ok_or_else(|| format!("kbench input size overflow (t={t}, in_dim={in_dim})"))?;
        let in_floats = if half_in { in_elems.div_ceil(2) } else { in_elems };
        let align64 = |n: usize| -> Result<usize, String> {
            n.checked_add(63)
                .map(|v| v / 64 * 64)
                .ok_or_else(|| format!("kbench alignment overflow ({n})"))
        };
        let out_off_usize = align64(in_floats)?;
        let out_off = u32::try_from(out_off_usize)
            .map_err(|_| format!("kbench out_off {out_off_usize} exceeds u32"))?;
        let out_elems = t
            .checked_mul(out_rows)
            .ok_or_else(|| format!("kbench output size overflow (t={t}, out_rows={out_rows})"))?;
        let out_stride = align64(out_elems)?;
        let total_out = out_stride
            .checked_mul(n_out_bufs)
            .ok_or_else(|| format!("kbench total output overflow (n_out_bufs={n_out_bufs})"))?;
        let total_floats = out_off_usize
            .checked_add(total_out)
            .ok_or_else(|| "kbench total scratch floats overflow".to_string())?;
        total_floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "kbench total scratch bytes overflow".to_string())?;
        let last_out_off = out_off_usize
            .checked_add(out_stride * (n_out_bufs - 1))
            .ok_or_else(|| "kbench output offset overflow".to_string())?;
        u32::try_from(last_out_off)
            .map_err(|_| format!("kbench last out_off {last_out_off} exceeds u32"))?;
        Ok((out_off, out_stride, total_floats))
    }

    /// Validate that `bytes` is a sequence of little-endian `f32` values (length is a
    /// multiple of 4) and that every element is finite (neither `NaN` nor `±Inf`).
    pub fn validate_f32_bytes_finite(name: &str, bytes: &[u8]) -> std::io::Result<()> {
        let (chunks, rem) = bytes.as_chunks::<4>();
        if !rem.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "tensor '{name}' byte length {} is not a multiple of 4",
                    bytes.len()
                ),
            ));
        }
        for (idx, &chunk) in chunks.iter().enumerate() {
            let v = f32::from_le_bytes(chunk);
            if !v.is_finite() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "tensor '{name}' contains non-finite f32 value ({v}) at element {idx}"
                    ),
                ));
            }
        }
        Ok(())
    }
}

impl Engine {
    pub fn load(path: &str, max_ctx: usize) -> Engine { Engine::load_with(path, max_ctx, KV8.load(std::sync::atomic::Ordering::Relaxed)) }
    /// `kv8`: keep the K/V cache as 8-bit integers with a per-vector scale (half the memory, not exact) instead of fp16
    pub fn load_with(path: &str, max_ctx: usize, kv8: bool) -> Engine {
        Self::try_load_with(path, max_ctx, kv8).expect("load engine")
    }

    /// Fallible version of `load_with` that validates the GGUF architecture, tensor shapes,
    /// quantization types, and byte bounds via `Gguf::validate_inkling_model()` before
    /// allocating any Metal resources. Respects `INK_SAFE_MATH=1` for selecting `MTLMathMode`.
    pub fn try_load_with(path: &str, max_ctx: usize, kv8: bool) -> std::io::Result<Engine> {
        let math_mode = if std::env::var("INK_SAFE_MATH")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            objc2_metal::MTLMathMode::Safe
        } else {
            objc2_metal::MTLMathMode::Fast
        };
        Self::try_load_with_math_mode(path, max_ctx, kv8, math_mode)
    }

    /// Fallible engine loader with an explicit Metal `MTLMathMode` (`Fast` or `Safe`).
    /// Validates the GGUF metadata, tensor layout, and finiteness of all F32/BF16 parameter
    /// buffer weights before allocating Metal resources.
    pub fn try_load_with_math_mode(
        path: &str,
        max_ctx: usize,
        kv8: bool,
        math_mode: objc2_metal::MTLMathMode,
    ) -> std::io::Result<Engine> {
        let t0 = std::time::Instant::now();
        let g = Gguf::open(path)?;
        g.validate_inkling_model()?;
        if max_ctx == 0 || !max_ctx.is_multiple_of(64) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("context size ({max_ctx}) must be a non-zero multiple of 64"),
            ));
        }
        let chunk: usize = std::env::var("INK_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096).min(max_ctx);
        if chunk == 0 || !chunk.is_multiple_of(32) || chunk > 4096 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid INK_CHUNK ({chunk}): must be a non-zero multiple of 32 <= 4096"),
            ));
        }
        if KV_DISK.load(std::sync::atomic::Ordering::Relaxed) && kv8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--kv-disk keeps the exact fp16 cache; it cannot be combined with --kv8",
            ));
        }
        let a = |k: &str| format!("inkling2_loop.{k}");
        let swa_pat = match &g.kv[&a("attention.sliding_window_pattern")] { Value::AB(v) => v.clone(), _ => unreachable!("validated by validate_inkling_model") };

        let mut pb = Pb { data: Vec::new() };
        let mut layers = Vec::new();
        for i in 0..N_PHYS {
            let n = |s: &str| format!("blk.{i}.{s}");
            let swa = swa_pat[i];
            let opt_f32 = |pb: &mut Pb, s: &str| -> std::io::Result<i32> {
                if g.opt(&n(s)).is_some() {
                    Ok(pb.try_f32(&g, &n(s))? as i32)
                } else {
                    Ok(-1)
                }
            };
            let attn_norm = opt_f32(&mut pb, "attn_norm.weight")?;
            let q_inp_norm = opt_f32(&mut pb, "attn_q_inp_norm.weight")?;
            let k_inp_norm = opt_f32(&mut pb, "attn_k_inp_norm.weight")?;
            let v_inp_norm = opt_f32(&mut pb, "attn_v_inp_norm.weight")?;
            let attn_res_query = opt_f32(&mut pb, "attn_res_query.weight")?;
            let n_attn_q = g.opt(&n("attn_res_query.weight")).map(|t| if t.ne.len() > 1 { t.ne[1] as u32 } else { 1 }).unwrap_or(0);
            let layer = Layer {
                swa,
                wq: qw(&g, &n("attn_q.weight")), wgate: qw(&g, &n("attn_gate.weight")), wk: qw(&g, &n("attn_k.weight")),
                wv: qw(&g, &n("attn_v.weight")), wo: qw(&g, &n("attn_output.weight")), wrel: qw(&g, &n("attn_rel.weight")),
                wsink: qw(&g, &n("attn_sink_proj.weight")),
                kconvp: g.opt(&n("attn_k_conv_proj.weight")).map(|_| qw(&g, &n("attn_k_conv_proj.weight"))),
                vconvp: g.opt(&n("attn_v_conv_proj.weight")).map(|_| qw(&g, &n("attn_v_conv_proj.weight"))),
                attn_norm, q_inp_norm, k_inp_norm, v_inp_norm, attn_res_query, n_attn_q,
                ffn_res_query: pb.try_f32(&g, &n("ffn_res_query.weight"))?,
                ffn_norm: pb.try_f32(&g, &n("ffn_norm.weight"))?,
                q_clip: pb.try_f32(&g, &n("attn_q_clip.weight"))?,
                rel_clip: pb.try_f32(&g, &n("attn_rel_clip.weight"))?,
                rel_bias: pb.try_f32(&g, &n("attn_rel_bias.weight"))?,
                k_conv: pb.try_f32(&g, &n("attn_k_conv.weight"))?,
                v_conv: pb.try_f32(&g, &n("attn_v_conv.weight"))?,
                attn_post_conv: pb.try_f32(&g, &n("attn_post_conv.weight"))?,
                ffn_post_conv: pb.try_f32(&g, &n("ffn_post_conv.weight"))?,
                router: pb.try_bf16(&g, &n("ffn_gate_inp.weight"))?,
                exp_bias: pb.try_f32(&g, &n("exp_probs_b.bias"))?,
                gu: qw(&g, &n("ffn_gate_up_exps.weight")),
                dn: qw(&g, &n("ffn_down_exps.weight")),
            };
            assert!(layer.swa == (layer.attn_norm >= 0));
            assert_eq!(layer.gu.ty, T_Q4_K);
            assert!(layer.dn.ty == T_Q8_0 || layer.dn.ty == T_Q5_0, "down type {}", type_name(layer.dn.ty));
            for w in [&layer.wq, &layer.wgate, &layer.wk, &layer.wo, &layer.wrel, &layer.wsink] { assert_eq!(w.ty, T_Q4_K); }
            assert!(layer.wv.ty == T_Q4_K || layer.wv.ty == T_Q6_K);
            layers.push(layer);
        }
        let mut loops = Vec::new();
        for i in LOOP_START..LOOP_END {
            let n = |s: &str| format!("blk.{i}.{s}");
            let names = ["q", "k", "v", "ffn"];
            let mut lw = LoopW { msg_query: pb.try_f32(&g, &n("loop_msg_query.weight"))?, down: [0; 4], gain: [0; 4], forget: [0; 4] };
            for c in 0..4 {
                lw.down[c] = pb.try_bf16(&g, &n(&format!("loop_{}_ctrl_down.weight", names[c])))?;
                lw.gain[c] = pb.try_bf16(&g, &n(&format!("loop_{}_ctrl_gain.weight", names[c])))?;
                lw.forget[c] = pb.try_bf16(&g, &n(&format!("loop_{}_ctrl_forget.weight", names[c])))?;
            }
            loops.push(lw);
        }
        let tok_norm = pb.try_f32(&g, "token_embd_norm.weight")?;
        let output_norm = pb.try_f32(&g, "output_norm.weight")?;
        let output_res_query = pb.try_f32(&g, "output_res_query.weight")?;
        let tok_embd = qw(&g, "token_embd.weight");
        let output = qw(&g, "output.weight");
        assert_eq!(tok_embd.ty, T_Q4_K);
        assert_eq!(output.ty, T_Q6_K);
        let n_vocab = g.u(&a("vocab_size_unpadded")) as usize;
        let n_vocab_pad = output.rows as usize;

        let src = format!("{}{}
{}
{}", if kv8 { "#define KV8 1
" } else { "" }, include_str!("kernels.metal"), include_str!("prefill.metal"), include_str!("bench_kernels.metal"));
        let kvb = if kv8 { 1 } else { 2 };
        let gpu = Gpu::new_with_math_mode(&src, KERNELS, math_mode);
        let (map, map_len) = g.raw_map().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "gguf is not mmapped")
        })?;
        let w = gpu.buffer_nocopy(map, map_len);
        pb.data.extend(std::iter::repeat_n(0u8, 128 * NREL * 4));   // k_attn_h reads up to 95 rows past a relative-bias table
        let pbuf = gpu.buffer_with(&pb.data);

        let max_split = MAX_SPLIT;
        let ar = Arena::new(max_split, n_vocab_pad);
        // the batch arena is allocated when a prefill needs it and released when generation starts
        let act = gpu.buffer(ar.total as usize * 4);
        // K/V cache layout: per slot [K: NKV x cap x HD][V: NKV x cap x HD] halfs, sliding-window and full slots in separate
        // buffers that both grow with the context (Metal commits a buffer when it is created, so capacity is real memory)
        let mut slot_kind = Vec::new();
        let (mut n_swa, mut n_full) = (0usize, 0usize);
        for s in 0..N_SLOT {
            let phys = if s < N_PHYS { s } else { LOOP_START + (s - N_PHYS) % (LOOP_END - LOOP_START) };
            if layers[phys].swa { slot_kind.push((true, n_swa)); n_swa += 1; } else { slot_kind.push((false, n_full)); n_full += 1; }
        }
        let kv_full_elems = n_full
            .checked_mul(2 * NKV * HD)
            .and_then(|v| v.checked_mul(max_ctx))
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "KV cache size overflow"))?;
        if kv_full_elems >= u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "KV cache too large for 32-bit offsets",
            ));
        }
        let cap0 = KV_CAP0.min(max_ctx);
        let kv_swa = gpu.buffer(n_swa * 2 * NKV * cap0 * HD * kvb);
        let kv_full = gpu.buffer(n_full * 2 * NKV * cap0 * HD * kvb);
        let cs = gpu.buffer(N_SLOT * NCH * 3 * 4);
        let toks = gpu.buffer((max_ctx + 8) * 4);
        let ctr = gpu.buffer(4096 * 4);
        eprintln!("[inkling] loaded {} ({:.1} GiB mapped, {:.1} MiB params) in {:.2?}",
            path.rsplit('/').next().unwrap_or(path), g.file_len as f64 / (1u64 << 30) as f64, pb.data.len() as f64 / (1 << 20) as f64, t0.elapsed());
        let tier = if KV_DISK.load(std::sync::atomic::Ordering::Relaxed) {
            let file = create_kv_disk_file_in(&std::env::temp_dir())?;
            Some(Tier { file, on: Cell::new(false), lo: Default::default(), cold: Default::default(), filed: Default::default(), next: Cell::new(0),
                        min_n: tier_math::clamp_tier_min(std::env::var("INK_TIER_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(512)),
                        stg: RefCell::new(gpu.buffer(16384)), half: Cell::new(0), pending: RefCell::new([None, None]),
                        plan: RefCell::new(Vec::new()), execs: RefCell::new(Vec::new()), loaded: Cell::new(usize::MAX),
                        read_bytes: Cell::new(0), read_time: Cell::new(Default::default()), unaligned: Cell::new(false) })
        } else { None };
        let eng = Engine {
            eps: g.f(&a("attention.layer_norm_rms_epsilon")) as f32,
            log_alpha: g.f(&a("attention.log_scale_alpha")) as f32,
            log_floor: g.f(&a("attention.log_scale_floor")) as f32,
            situ_beta: g.f(&a("activation.situ_beta")) as f32,
            situ_lbeta: g.f(&a("activation.situ_linear_beta")) as f32,
            wscale: g.f(&a("expert_weights_scale")) as f32,
            softcap: g.f(&a("final_logit_softcapping")) as f32,
            w, pb: pbuf, act: std::cell::RefCell::new(act), kv_swa: std::cell::RefCell::new(kv_swa), kv_full: std::cell::RefCell::new(kv_full),
            swa_cap: std::cell::Cell::new(cap0), full_cap: std::cell::Cell::new(cap0), slot_kind, tier, qcap: RefCell::new(None), prune: RefCell::new(None), cs, toks, ctr, layers, loops, tok_embd, tok_norm, output, output_norm, output_res_query,
            ar, kv8, kvb, ba: Default::default(), chunk, tensor: std::env::var("INK_NO_TENSOR").is_err(),
            hgemm: std::env::var("INK_NO_HGEMM").is_err() && std::env::var("INK_NO_TENSOR").is_err(),
            sep_comb: std::env::var("INK_FUSE_COMB").is_err(), sep_topk: std::env::var("INK_SEP_TOPK").is_ok(),
            tail: std::cell::Cell::new(std::env::var("INK_NO_TAIL").is_err()),
            skip: Default::default(), stop: std::cell::Cell::new(usize::MAX), done: std::cell::Cell::new(0),
            last_out: Default::default(), last_mid: Default::default(),
            max_ctx, n_vocab,
            gpu, g,
        };
        let mut eng = eng;
        eng.ensure_batch(64);
        let ok = !eng.hgemm || eng.layout_ok();
        eng.release_batch();
        if !ok {
            eprintln!("[inkling] unexpected tensor accumulator layout on this GPU/OS: using the slower prefill path");
            eng.hgemm = false;
        }
        if std::env::var("INK_NO_WARM").is_err() { eng.warm(); }
        Ok(eng)
    }

    /// (k_base, v_base, cap) of a slot's K/V inside its buffer (offsets in cache elements: fp16, or bytes with --kv8)
    fn kv_slot(&self, slot: usize) -> (u32, u32, u32) {
        let (swa, i) = self.slot_kind[slot];
        let cap = if swa { self.swa_cap.get() } else { self.full_cap.get() };
        ((i * 2 * NKV * cap * HD) as u32, ((i * 2 + 1) * NKV * cap * HD) as u32, cap as u32)
    }
    /// bind the K/V buffer a layer's kernels address through buffer 3
    fn bind_kv(&self, e: &Enc, swa: bool) { if swa { e.buf(&self.kv_swa.borrow(), 0, 3); } else { e.buf(&self.kv_full.borrow(), 0, 3); } }
    /// positions the K/V cache can hold right now
    pub fn kv_capacity(&self) -> usize { if self.swa_cap.get() >= EXT { self.full_cap.get() } else { self.swa_cap.get().min(self.full_cap.get()) } }
    /// first position a generation loop must call `gen_reserve` for (with an idle GPU) before it may run that step
    pub fn gen_capacity(&self) -> usize {
        match &self.tier { Some(t) if t.on.get() => t.next.get(), Some(t) => self.kv_capacity().min(t.min_n), None => self.kv_capacity() }
    }
    /// bytes of K/V cache held in memory (with the disk tier: the hot rings and the read staging)
    pub fn kv_bytes(&self) -> usize {
        self.kv_swa.borrow().length() + self.kv_full.borrow().length() + self.tier.as_ref().map_or(0, |t| if t.on.get() { t.stg.borrow().length() } else { 0 })
    }
    /// Make room for `n` positions in the K/V cache (no-op if they fit). Nothing may be in flight on the GPU.
    /// Sliding-window layers keep one window (their ring is linear until the context reaches it, so it can grow like the
    /// full-attention rows); `headroom` adds slack for the tokens about to be generated so growth stays rare.
    pub fn reserve(&self, n: usize, headroom: bool) {
        assert!(n <= self.max_ctx, "context full ({} tokens)", self.max_ctx);
        let grow = |buf: &std::cell::RefCell<Buf>, capc: &std::cell::Cell<usize>, need: usize, limit: usize, nslots: usize| {
            let old = capc.get();
            if need <= old { return; }
            // slack for the tokens about to be generated: INK_HEADROOM=a,b -> need/a (at least b positions)
            let headroom_env = std::env::var("INK_HEADROOM").ok();
            let (hd, hm) = dev_helpers::parse_headroom(headroom_env.as_deref());
            let cap = dev_helpers::reserve_capacity(need, limit, headroom, hd, hm);
            let eb = self.kvb;
            let nb = self.gpu.buffer(nslots * 2 * NKV * cap * HD * eb);
            // every (slot, K|V, group) row block moves to its place in the larger layout
            unsafe {
                let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
                for blk in 0..nslots * 2 * NKV { std::ptr::copy_nonoverlapping(src.add(blk * old * HD * eb), dst.add(blk * cap * HD * eb), old * HD * eb); }
            }
            *buf.borrow_mut() = nb;
            capc.set(cap);
        };
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        grow(&self.kv_swa, &self.swa_cap, n.min(EXT), EXT, n_swa);
        grow(&self.kv_full, &self.full_cap, n, self.max_ctx, N_SLOT - n_swa);
    }
    /// Allocate (or enlarge) the batch arena for a prefill of `t` tokens. Nothing may be in flight on the GPU.
    pub fn ensure_batch(&self, t: usize) {
        let t = t.div_ceil(64) * 64;
        if t <= self.ba.get().cap { return; }
        let nb = BArena::new(self.ar.total, t, self.hgemm, self.kvb);
        self.realloc_act(nb.end as usize * 4);
        self.ba.set(nb);
    }
    /// Allocate everything a prefill of `t` tokens ending at context length `n` needs and let the GPU map it now.
    /// The first use of a new buffer costs ~20 ms per GiB; a caller that knows its prompt before it starts the clock
    /// (one-shot runs: while the model loads) pays that here instead of inside the prefill.
    pub fn prepare(&self, n: usize, t: usize) {
        self.reserve(n + 1, true);
        self.ensure_batch(t.min(self.chunk));
        let cb = self.gpu.cmd();
        let e = Enc::new(&cb);
        self.bind(&e);
        // one trivial dispatch per buffer (a window-history copy of key 0 into the scratch guard area) makes the driver wire them
        let b = self.ba.get();
        let a = KvCopyArgs { n: 1, pos0: 0, ring_k: 0, ring_v: 0, cap: 64, sk_off: b.sk, sv_off: b.sv, scap: (EXT + b.cap + 2 * KGUARD) as u32,
                             kpos0: 0, to_ring: 0, pad: [0; 2] };
        e.pso(&self.gpu.pso("k_kvcopy")); e.bytes(&a, 5);
        self.bind_kv(&e, true); e.dispatch((1, 1, 1), 32);
        e.barrier();
        self.bind_kv(&e, false); e.dispatch((1, 1, 1), 32);
        e.end(); cb.commit(); cb.waitUntilCompleted();
    }
    /// Give the batch arena back (the decode arena and the last logits stay). Nothing may be in flight on the GPU.
    pub fn release_batch(&self) {
        if self.ba.get().cap == 0 { return; }
        self.realloc_act(self.ar.total as usize * 4);
        self.ba.set(BArena::default());
    }
    fn realloc_act(&self, bytes: usize) {
        let nb = self.gpu.buffer(bytes);
        unsafe { std::ptr::copy_nonoverlapping(self.act.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8, self.ar.total as usize * 4); }
        *self.act.borrow_mut() = nb;
    }
    /// bytes of GPU working memory besides the weights: (K/V cache, activations incl. the batch arena if allocated)
    pub fn mem(&self) -> (usize, usize) { (self.kv_bytes(), self.act.borrow().length()) }

    // ======================================================================== disk tier of the K/V cache (--kv-disk)
    pub fn tier_on(&self) -> bool { self.tier.as_ref().is_some_and(|t| t.on.get()) }
    fn kind_bufs(&self, kind: usize) -> (&RefCell<Buf>, &Cell<usize>, usize) {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        if kind == 0 { (&self.kv_swa, &self.swa_cap, n_swa) } else { (&self.kv_full, &self.full_cap, N_SLOT - n_swa) }
    }
    /// what the cache would hold in memory for `n` positions without the tier, bytes
    pub fn kv_exact_bytes(&self, n: usize) -> usize {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        tier_math::kv_exact_bytes(n, n_swa)
    }
    /// bytes of the cache that are on disk only (read back for every token)
    pub fn kv_disk_bytes(&self, n: usize) -> usize {
        let Some(t) = &self.tier else { return 0 };
        if !t.on.get() { return 0; }
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        let cold = |kind: usize, w0: usize| t.lo[kind].get().saturating_sub(w0);
        (n_swa * cold(0, n.saturating_sub(EXT)) + (N_SLOT - n_swa) * cold(1, 0)) * REC
    }
    /// (bytes read from the K/V file, time spent reading) since the engine was loaded
    pub fn tier_io(&self) -> (u64, std::time::Duration) { self.tier.as_ref().map_or((0, Default::default()), |t| (t.read_bytes.get(), t.read_time.get())) }
    /// hot ring sizes (positions) and staging bytes for a context of `n` positions: together at most half of what the
    /// cache would take in memory
    fn tier_targets(&self, n: usize) -> (usize, usize, usize) {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        // dev: INK_TIER_HOT=positions forces small hot rings so short tests exercise the file (clamped >= 64)
        let hot_override = std::env::var("INK_TIER_HOT").ok().and_then(|v| v.parse::<usize>().ok());
        tier_math::tier_targets(n, n_swa, Self::split_target(), hot_override)
    }
    fn tier_foff(&self, kind: usize, i: usize, p: usize) -> u64 {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        ((if kind == 0 { i * EXT + p % EXT } else { n_swa * EXT + i * self.max_ctx + p }) * REC) as u64
    }
    fn split_target() -> usize {
        tier_math::clamp_split_target(
            std::env::var("INK_SPLITS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize),
        )
    }
    /// keys per split of a decode attention over `nk` keys (the same division with and without the disk tier, so both sum
    /// the same keys in the same order)
    fn split_kps(nk: usize) -> usize {
        tier_math::split_kps(nk, Self::split_target())
    }
    /// file pieces (offset, positions) that hold positions [a, b) of a slot, in position order (a sliding-window slot's
    /// file is a ring of one window)
    fn tier_pieces(&self, kind: usize, i: usize, a: usize, b: usize) -> Vec<(u64, usize)> {
        let mut v = Vec::new();
        let mut p = a;
        while p < b { let e = if kind == 0 { b.min((p / EXT + 1) * EXT) } else { b }; v.push((self.tier_foff(kind, i, p), e - p)); p = e; }
        v
    }
    /// copy one position (16 rows of 128 fp16) between two cache buffers / between a cache buffer and a file record
    unsafe fn rows_copy(src: *const u8, i: usize, cap_s: usize, idx_s: usize, dst: *mut u8, cap_d: usize, idx_d: usize) {
        for r in 0..2 * NKV { std::ptr::copy_nonoverlapping(src.add(((i * 2 * NKV + r) * cap_s + idx_s) * HD * 2), dst.add(((i * 2 * NKV + r) * cap_d + idx_d) * HD * 2), HD * 2); }
    }
    unsafe fn rows_to_rec(src: *const u8, i: usize, cap: usize, idx: usize, rec: *mut u8) {
        for r in 0..2 * NKV { std::ptr::copy_nonoverlapping(src.add(((i * 2 * NKV + r) * cap + idx) * HD * 2), rec.add(r * HD * 2), HD * 2); }
    }
    unsafe fn rec_to_rows(rec: *const u8, dst: *mut u8, i: usize, cap: usize, idx: usize) {
        for r in 0..2 * NKV { std::ptr::copy_nonoverlapping(rec.add(r * HD * 2), dst.add(((i * 2 * NKV + r) * cap + idx) * HD * 2), HD * 2); }
    }
    /// write positions [a, b) of every slot of a kind to the file; `idx(p)` is where position p sits in the kind's buffer now
    fn tier_write(&self, kind: usize, a: usize, b: usize, idx: impl Fn(usize) -> usize) {
        if a >= b { return; }
        let t = self.tier.as_ref().unwrap();
        let (buf, capc, nsl) = self.kind_bufs(kind);
        let (src, cap) = (buf.borrow().contents().as_ptr() as *const u8, capc.get());
        let mut recs = vec![0u8; (b - a).min(2048) * REC];
        for i in 0..nsl {
            let mut p = a;
            while p < b {
                let e = b.min(p + 2048);
                for q in p..e { unsafe { Self::rows_to_rec(src, i, cap, idx(q), recs.as_mut_ptr().add((q - p) * REC)); } }
                let mut o = 0;
                for (off, n) in self.tier_pieces(kind, i, p, e) { t.file.write_all_at(&recs[o..o + n * REC], off).expect("write the K/V file"); o += n * REC; }
                p = e;
            }
        }
    }
    /// read file pieces (offset, bytes, destination) into memory, several at a time (uncached reads; large pieces are split)
    fn tier_read(&self, jobs: &[(u64, usize, *mut u8)]) {
        let t = self.tier.as_ref().unwrap();
        let t0 = std::time::Instant::now();
        let piece = tier_math::clamp_tier_piece_bytes(
            std::env::var("INK_TIER_PIECE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4),
        );
        let nthreads = tier_math::clamp_tier_threads(
            std::env::var("INK_TIER_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4),
        );
        let addr_jobs: Vec<(u64, usize, usize)> = jobs
            .iter()
            .map(|&(off, bytes, dst)| (off, bytes, dst as usize))
            .collect();
        let parts = tier_math::plan_read_parts(&addr_jobs, piece);
        let total: usize = parts.iter().map(|p| p.1).sum();
        let file = &t.file;
        let read = |&(off, n, dst): &(u64, usize, usize)| {
            // SAFETY: Each `job` targets a disjoint slice within the currently-idle staging half
            // of `t.stg` (`tier_load` waited for the prior GPU command buffer using that half),
            // and `plan_read_parts` partitions each job into non-overlapping `[dst, dst + n)` ranges.
            let sl = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, n) };
            file.read_exact_at(sl, off).expect("read the K/V file");
        };
        if parts.len() <= 1 { parts.iter().for_each(read); }
        else {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let (parts, next, read) = (&parts, &next, &read);
            std::thread::scope(|sc| {
                for _ in 0..parts.len().min(nthreads) {
                    sc.spawn(move || loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if k >= parts.len() { break; }
                        read(&parts[k]);
                    });
                }
            });
        }
        t.read_bytes.set(t.read_bytes.get() + total as u64);
        t.read_time.set(t.read_time.get() + t0.elapsed());
    }
    fn tier_wait(&self) {
        let t = self.tier.as_ref().unwrap();
        for p in t.pending.borrow_mut().iter_mut() { if let Some(cb) = p.take() { cb.waitUntilCompleted(); } }
    }
    /// Start keeping only the newest positions in memory: `n` positions are cached, in the normal layout. GPU idle.
    fn tier_enter(&self, n: usize) {
        let t = self.tier.as_ref().unwrap();
        if t.on.get() || n < t.min_n { return; }
        let (hs, hf, s) = self.tier_targets(n);
        for (kind, h) in [(0usize, hs), (1usize, hf)] {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            let old_cap = capc.get();
            let h = h.min(n);
            let (first, lo) = (if kind == 0 { n.saturating_sub(EXT) } else { 0 }, n - h.min(n));
            let idx_old = |p: usize| if kind == 0 { p % old_cap } else { p };
            // the file runs one split ahead of the hot ring's start, so a split boundary always lies where both have the keys
            let cold = (lo + Self::split_kps(n + 1 - first) + 64).min(n);
            self.tier_write(kind, first.max(t.filed[kind].get()), cold, idx_old);
            t.filed[kind].set(cold.max(t.filed[kind].get()));
            let nb = self.gpu.buffer(nsl * 2 * NKV * h * HD * 2);
            unsafe {
                let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
                for i in 0..nsl { for p in lo.max(first)..n { Self::rows_copy(src, i, old_cap, idx_old(p), dst, h, p % h); } }
            }
            *buf.borrow_mut() = nb;
            capc.set(h);
            t.lo[kind].set(lo); t.cold[kind].set(cold);
        }
        *t.stg.borrow_mut() = self.gpu.buffer(s);
        t.half.set(0); t.loaded.set(usize::MAX);
        t.on.set(true);
        t.next.set(n + 64);
    }
    /// Bring the whole cache back into memory in the normal layout (a prefill attends through it directly). GPU idle.
    pub fn tier_leave(&self, n: usize) {
        let Some(t) = &self.tier else { return };
        if !t.on.get() { return; }
        self.tier_wait();
        for kind in 0..2 {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            let (h, lo) = (capc.get(), t.lo[kind].get());
            let first = if kind == 0 { n.saturating_sub(EXT) } else { 0 };
            let cap = if kind == 0 { (n.min(EXT).div_ceil(64) * 64).clamp(KV_CAP0, EXT) } else { (n.div_ceil(64) * 64).clamp(KV_CAP0, self.max_ctx) };
            let idx_new = |p: usize| if kind == 0 { p % cap } else { p };
            let nb = self.gpu.buffer(nsl * 2 * NKV * cap * HD * 2);
            let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
            let mut recs = vec![0u8; 2048 * REC];
            let rp = recs.as_mut_ptr();
            let cold_end = lo.max(first).min(n);
            for i in 0..nsl {
                let mut p = first;
                while p < cold_end {
                    let e = cold_end.min(p + 2048);
                    let mut o = 0;
                    let mut jobs: Vec<(u64, usize, *mut u8)> = Vec::new();
                    for (off, k) in self.tier_pieces(kind, i, p, e) { jobs.push((off, k * REC, unsafe { rp.add(o) })); o += k * REC; }
                    self.tier_read(&jobs);
                    for q in p..e { unsafe { Self::rec_to_rows(rp.add((q - p) * REC), dst, i, cap, idx_new(q)); } }
                    p = e;
                }
                for q in cold_end..n { unsafe { Self::rows_copy(src, i, h, q % h, dst, cap, idx_new(q)); } }
            }
            drop(recs);
            *buf.borrow_mut() = nb;
            capc.set(cap);
        }
        *t.stg.borrow_mut() = self.gpu.buffer(16384);
        t.on.set(false);
    }
    /// forget the cache (a new session)
    fn tier_reset(&self) {
        let Some(t) = &self.tier else { return };
        t.filed[0].set(0); t.filed[1].set(0);
        if !t.on.get() { return; }
        self.tier_wait();
        for kind in 0..2 {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            *buf.borrow_mut() = self.gpu.buffer(nsl * 2 * NKV * KV_CAP0 * HD * 2);
            capc.set(KV_CAP0);
        }
        *t.stg.borrow_mut() = self.gpu.buffer(16384);
        t.on.set(false);
    }
    /// Make room for the step at `pos` (positions 0..pos are cached). Nothing may be in flight on the GPU.
    pub fn gen_reserve(&self, pos: usize) {
        let Some(t) = &self.tier else { self.reserve(pos + 1, true); return };
        if !t.on.get() { self.tier_enter(pos); }
        if !t.on.get() { self.reserve(pos + 1, true); return; }
        // the hot rings follow the context so memory stays at half of the cache
        self.tier_wait();
        let (hs, hf, s) = self.tier_targets(pos + 1);
        for (kind, h) in [(0usize, hs), (1usize, hf)] {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            let cap = capc.get();
            if h < cap + 64 { continue; }
            let nb = self.gpu.buffer(nsl * 2 * NKV * h * HD * 2);
            unsafe {
                let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
                for i in 0..nsl { for p in t.lo[kind].get()..pos { Self::rows_copy(src, i, cap, p % cap, dst, h, p % h); } }
            }
            *buf.borrow_mut() = nb;
            capc.set(h);
        }
        if s > t.stg.borrow().length() { *t.stg.borrow_mut() = self.gpu.buffer(s); t.half.set(0); }
        t.next.set(pos + 64);
    }
    /// One decode step, committed: returns the command buffer whose completion means the step is done.
    pub fn step(&self, pos: usize, logits: bool, argmax: bool) -> CmdBuf {
        if !self.tier_on() {
            let cb = self.gpu.cmd();
            self.encode_step(&cb, pos, logits, argmax);
            cb.commit();
            return cb;
        }
        let t = self.tier.as_ref().unwrap();
        // Rows this step overwrites go to the file first - in fact one split earlier than that (and in batches of 16), so the
        // keys around the hot ring's start are in both places. Rows older than 32 steps are final (2 steps are in flight).
        for kind in 0..2 {
            let (_, capc, _) = self.kind_bufs(kind);
            let cap = capc.get();
            let need = (pos + 1).saturating_sub(cap);
            let lo = t.lo[kind].get().max(need);
            let done = pos.saturating_sub(32);
            let want = (lo + Self::split_kps(if kind == 0 { (pos + 1).min(EXT) } else { pos + 1 })).min(done).max(need);
            if t.cold[kind].get() < want {
                let upto = (want + 15).min(done).max(want);
                if tier_math::tier_needs_gpu_drain(upto, done) {
                    // Safety net: wait for all in-flight GPU KV writes before reading from the hot ring on CPU.
                    self.gpu.drain();
                }
                self.tier_write(kind, t.cold[kind].get().max(t.filed[kind].get()), upto, |p| p % cap);
                t.cold[kind].set(upto);
                t.filed[kind].set(upto.max(t.filed[kind].get()));
            }
            t.lo[kind].set(lo);
        }
        let cb = self.gpu.cmd();
        // Plan this step's file reads: per layer execution the splits older than its hot ring, packed into staging halves.
        // The layer's keys are divided into splits exactly as without the tier (same kps from the window start); the file
        // supplies whole splits up to a boundary b that the hot ring also reaches, so every split sums the same keys in the
        // same order as the in-memory path and the results are bit-identical.
        let half_recs = t.stg.borrow().length() / 2 / REC;
        let (mut plan, mut execs) = (Vec::new(), Vec::new());
        let (mut round, mut room) = (0usize, half_recs);
        let mut order: Vec<usize> = (0..LOOP_START).collect();
        for pass in 0..N_PASS { for ip in LOOP_START..LOOP_END { order.push(slot_of(pass, ip)); } }
        order.extend(LOOP_END..N_PHYS);
        for (exec, &slot) in order.iter().enumerate() {
            let (swa, i) = self.slot_kind[slot];
            let kind = if swa { 0 } else { 1 };
            let w0 = if swa { (pos + 1).saturating_sub(EXT) } else { 0 };
            let lo = t.lo[kind].get().max(w0);
            let nk = pos + 1 - w0;
            let kps = Self::split_kps(nk);
            let b = w0 + (lo - w0).div_ceil(kps) * kps;
            if b <= pos && b <= t.cold[kind].get().max(w0) && kps <= half_recs {
                let (mut c, mut sp) = (w0, 0usize);
                while c < b {
                    if room < kps { round += 1; room = half_recs; }
                    let ns = (room / kps).min((b - c) / kps);
                    plan.push(Chunk { exec, kind, i, c0: c, c1: c + ns * kps, round, off: (half_recs - room) * REC, sp0: sp, ns, kps });
                    c += ns * kps; sp += ns; room -= ns * kps;
                }
                let nsplit = nk.div_ceil(kps);
                execs.push(ExecPlan { b, kps, nsplit, sp0: sp, ns: nsplit - sp });
            } else {
                // (tiny hot rings in tests) no common boundary: divide the file part and the hot part separately - the same
                // keys, summed in a different grouping
                if lo > w0 && !t.unaligned.replace(true) { eprintln!("[inkling] --kv-disk: hot ring shorter than a split, sums are grouped differently from the in-memory path"); }
                let spl = |nk: usize| { let kps = Self::split_kps(nk); (kps, nk.div_ceil(kps)) };
                let (mut c, mut sp) = (w0, 0usize);
                while c < lo {
                    if room < (lo - c).min(64) { round += 1; room = half_recs; }
                    let n = room.min(lo - c);
                    let (kps, ns) = spl(n);
                    plan.push(Chunk { exec, kind, i, c0: c, c1: c + n, round, off: (half_recs - room) * REC, sp0: sp, ns, kps });
                    c += n; sp += ns; room -= n;
                }
                let (kps, ns) = spl(pos + 1 - lo);
                assert!(sp + ns <= MAX_SPLIT, "too many K/V file chunks for one layer");
                execs.push(ExecPlan { b: lo, kps, nsplit: sp + ns, sp0: sp, ns });
            }
        }
        *t.execs.borrow_mut() = execs;
        *t.plan.borrow_mut() = plan;
        t.loaded.set(usize::MAX);
        let mut cx = Cx { e: Enc::new(&cb), cb: Some(cb) };
        self.encode_step_cx(&mut cx, pos, logits, argmax);
        let Cx { e, cb } = cx;
        e.end();
        let cb = cb.unwrap();
        cb.commit();
        // the next step starts in the other staging half
        let h = t.half.get();
        t.pending.borrow_mut()[h] = Some(cb.clone());
        t.half.set(h ^ 1);
        cb
    }
    /// make the staging round of chunk `k` of the plan available: switch halves if it is a new one and read it from the file
    fn tier_load(&self, cx: &mut Cx, k: usize) {
        let t = self.tier.as_ref().unwrap();
        let plan = t.plan.borrow();
        let round = plan[k].round;
        if t.loaded.get() == round { return; }
        if round > 0 {
            // hand the commands written so far to the GPU; they keep reading the half we are leaving
            let ncb = self.gpu.cmd();
            let oe = std::mem::replace(&mut cx.e, Enc::new(&ncb));
            oe.end();
            let ocb = cx.cb.replace(ncb).unwrap();
            ocb.commit();
            let h = t.half.get();
            t.pending.borrow_mut()[h] = Some(ocb);
            t.half.set(h ^ 1);
            self.bind(&cx.e);
        }
        // whatever last used this half must be finished before it is overwritten
        let h = t.half.get();
        let prev = t.pending.borrow_mut()[h].take();
        if let Some(cb) = prev { cb.waitUntilCompleted(); }
        let stg = t.stg.borrow();
        let base = unsafe { (stg.contents().as_ptr() as *mut u8).add(h * (stg.length() / 2)) };
        let mut jobs = Vec::new();
        for c in plan.iter().filter(|c| c.round == round) {
            let mut o = c.off;
            for (off, n) in self.tier_pieces(c.kind, c.i, c.c0, c.c1) { jobs.push((off, n * REC, unsafe { base.add(o) })); o += n * REC; }
        }
        self.tier_read(&jobs);
        t.loaded.set(round);
    }

    // ======================================================================== dev: key eviction experiments
    pub fn qcap_alloc(&self, n: usize) {
        let nf = self.slot_kind.iter().filter(|k| !k.0).count();
        *self.qcap.borrow_mut() = Some((self.gpu.buffer(nf * n * (NH * HD / 2 + NH) * 4), n));
    }
    /// conv state and sliding-window rings, to rerun the same decode steps
    pub fn snapshot(&self) -> (Vec<u8>, Vec<u8>) {
        let g = |b: &Buf| unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const u8, b.length()).to_vec() };
        (g(&self.cs), g(&self.kv_swa.borrow()))
    }
    pub fn restore(&self, s: &(Vec<u8>, Vec<u8>)) {
        let kv = self.kv_swa.borrow();
        bounds::validate_restore_range(s.0.len(), self.cs.length(), s.1.len(), kv.length())
            .unwrap_or_else(|e| panic!("{e}"));
        // SAFETY: `validate_restore_range` verified `s.0.len() == self.cs.length()` and
        // `s.1.len() == kv.length()`. `s.0` and `s.1` are heap `Vec<u8>`s disjoint from the
        // Metal shared buffers `self.cs` and `kv`.
        unsafe {
            std::ptr::copy_nonoverlapping(s.0.as_ptr(), self.cs.contents().as_ptr() as *mut u8, s.0.len());
            std::ptr::copy_nonoverlapping(s.1.as_ptr(), kv.contents().as_ptr() as *mut u8, s.1.len());
        }
    }
    /// Score every far key (older than one window at context `n`) of the full-attention slots by the attention it received
    /// from a sample of the prompt's own queries, keep the best `keep` share per (slot, group) and optionally one summary
    /// key (mean of the evicted keys and values, score bias ln(count)). Returns (far keys, kept per group).
    pub fn prune_build(&self, n: usize, keep: f64, use_sum: bool, nq: usize, mode: &str) -> (usize, usize) {
        assert!(!self.kv8 && !self.tier_on() && n > EXT);
        let qc = self.qcap.borrow();
        let (qbuf, ncap) = qc.as_ref().expect("qcap_alloc + prefill first");
        let r0 = n - EXT;
        let f = ((r0 as f64 * keep).round() as usize).min(r0);
        let rows = f + 1;
        let nf = self.slot_kind.iter().filter(|k| !k.0).count();
        let bag = self.gpu.buffer(nf * 2 * NKV * rows * HD * 2);
        let lut: Vec<f32> = (0..65536u32).map(|h| { let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 }; let e = ((h >> 10) & 31) as i32; let m = (h & 1023) as f32;
            if e == 0 { sg * m * 2f32.powi(-24) } else if e == 31 { 0.0 } else { sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15) } }).collect();
        let f2h = |v: f32| -> u16 { let bits = v.to_bits(); let sign = ((bits >> 16) & 0x8000) as u16; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            if v == 0.0 || exp <= 0 { sign } else { let mant = (bits >> 13) & 0x3ff; let rem = bits & 0x1fff;
                let mut h = ((exp as u32) << 10 | mant) + if rem > 0x1000 || (rem == 0x1000 && mant & 1 == 1) { 1 } else { 0 }; if h >= 0x7c00 { h = 0x7bff; } sign | h as u16 } };
        let cap = self.full_cap.get();
        let (kvp, qp, bp) = (self.kv_full.borrow().contents().as_ptr() as usize, qbuf.contents().as_ptr() as usize, bag.contents().as_ptr() as usize);
        let per = ncap * (NH * HD / 2 + NH) * 4;
        // sampled query positions: spread over the context, denser at the end
        let mut qpos: Vec<usize> = (0..nq).map(|i| (n - 1) - i * (n - 1) / nq.max(1)).collect();
        qpos.dedup();
        let tasks: Vec<(usize, usize)> = (0..nf).flat_map(|fi| (0..NKV).map(move |g| (fi, g))).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (lut, qpos, tasks, f2h) = (&lut, &qpos, &tasks, &f2h);
        std::thread::scope(|sc| {
            for _ in 0..8 { sc.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k >= tasks.len() { break; }
                let (fi, g) = tasks[k];
                let row = |kv: usize, j: usize| unsafe { std::slice::from_raw_parts((kvp as *const u16).add(((fi * 2 + kv) * NKV + g) * cap * HD + j * HD), HD) };
                let keys: Vec<f32> = (0..n).flat_map(|j| row(0, j).iter().map(|&h| lut[h as usize])).collect();
                let mut imp = vec![0f32; n];
                let mut sc_ = vec![0f32; n];
                for &t in qpos.iter() { for hh in 0..3 {
                    let h = g * 3 + hh;
                    let q: Vec<f32> = unsafe { std::slice::from_raw_parts((qp + fi * per) as *const u16, ncap * NH * HD) }[(t * NH + h) * HD..(t * NH + h + 1) * HD].iter().map(|&x| lut[x as usize]).collect();
                    let sink = unsafe { *((qp + fi * per + ncap * NH * HD * 2) as *const f32).add(t * NH + h) };
                    let mut m = sink;
                    for j in 0..=t {
                        let kr = &keys[j * HD..(j + 1) * HD];
                        let mut a = [0f32; 8];
                        for c in (0..HD).step_by(8) { for u in 0..8 { a[u] += q[c + u] * kr[c + u]; } }
                        let s = a.iter().sum::<f32>();
                        sc_[j] = s; if s > m { m = s; }
                    }
                    let mut z = (sink - m).exp();
                    for j in 0..=t { sc_[j] = (sc_[j] - m).exp(); z += sc_[j]; }
                    match mode {
                        "max" => for j in 0..=t { let p = sc_[j] / z; if p > imp[j] { imp[j] = p; } },
                        // only what a key received while it was inside one window of the query / only from further away
                        "near" => for j in t.saturating_sub(EXT - 1)..=t { imp[j] += sc_[j] / z; },
                        "far" => for j in 0..=t.saturating_sub(EXT) { imp[j] += sc_[j] / z; },
                        _ => for j in 0..=t { imp[j] += sc_[j] / z; },
                    }
                } }
                if mode == "rand" { for (j, x) in imp.iter_mut().enumerate() { *x = ((j as u64 * 0x9E3779B97F4A7C15 + (k as u64) * 0xD1B54A32D192ED03) >> 40) as f32; } }
                let mut order: Vec<usize> = (0..r0).collect();
                order.sort_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap());
                let mut kept: Vec<usize> = order[..f].to_vec();
                kept.sort();
                let out = |kv: usize, r: usize| unsafe { std::slice::from_raw_parts_mut((bp as *mut u16).add(((fi * 2 + kv) * NKV + g) * rows * HD + r * HD), HD) };
                for kv in 0..2 {
                    for (r, &j) in kept.iter().enumerate() { out(kv, r).copy_from_slice(row(kv, j)); }
                    let mut mean = vec![0f64; HD];
                    for &j in &order[f..] { for (mm, &h) in mean.iter_mut().zip(row(kv, j)) { *mm += lut[h as usize] as f64; } }
                    let cnt = (r0 - f).max(1) as f64;
                    for (o, mm) in out(kv, f).iter_mut().zip(&mean) { *o = f2h((*mm / cnt) as f32); }
                }
            }); }
        });
        drop(qc);
        *self.prune.borrow_mut() = Some(Prune { bag, rows, r0, lastbias: ((r0 - f).max(1) as f32).ln(), use_sum: use_sum && r0 > f });
        (r0, f)
    }
    /// dev: how peaked is full attention? After a prefill of `n` tokens (tail mode off) the arena still holds the last layer's
    /// prepared queries. For the last `nq` query tokens and every head: how many of its keys score within `margin` logits of
    /// the best one (keys further down contribute less than e^-margin of it). Relative bias is ignored (it only exists within
    /// 1024 tokens). Returns per margin the mean fraction of keys that are within it, and the worst head's fraction.
    pub fn kv_screen(&self, n: usize, nq: usize, margins: &[f32]) -> Vec<(f32, f64, f64)> {
        assert!(!self.kv8, "this dev tool reads the fp16 cache");
        let b = self.ba.get();
        let slot = N_PHYS - 1;
        assert!(!self.slot_kind[slot].0, "last layer is expected to be a full-attention layer");
        let (kb, _, cap) = self.kv_slot(slot);
        let h2f = |h: u16| -> f32 { let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 }; let e = ((h >> 10) & 31) as i32; let m = (h & 1023) as f32;
            if e == 0 { sg * m * 2f32.powi(-24) } else { sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15) } };
        let kv = self.kv_full.borrow();
        let kp = kv.contents().as_ptr() as *const u16;
        let act = self.act.borrow();
        let qp = unsafe { (act.contents().as_ptr() as *const u16).add(b.qp as usize * 2) };
        let sink = unsafe { (act.contents().as_ptr() as *const f32).add(b.sink as usize) };
        let mut within = vec![(0.0f64, 0.0f64); margins.len()];
        let mut cnt = 0usize;
        for g in 0..NKV {
            // keys of this group as f32
            let keys: Vec<f32> = (0..n * HD).map(|i| h2f(unsafe { *kp.add(kb as usize + g * cap as usize * HD + i) })).collect();
            for t in dev_helpers::kv_screen_range(n, nq) { for hh in 0..3 {
                let h = g * 3 + hh;
                let q: Vec<f32> = (0..HD).map(|d| h2f(unsafe { *qp.add((t * NH + h) * HD + d) })).collect();
                let sc: Vec<f32> = (0..=t).map(|j| { let k = &keys[j * HD..(j + 1) * HD]; q.iter().zip(k).map(|(a, c)| a * c).sum::<f32>() }).collect();
                let m = sc.iter().cloned().fold(unsafe { *sink.add(t * NH + h) }, f32::max);
                for (i, &mg) in margins.iter().enumerate() {
                    let f = sc.iter().filter(|&&x| x > m - mg).count() as f64 / sc.len() as f64;
                    within[i].0 += f; within[i].1 = within[i].1.max(f);
                }
                cnt += 1;
            } }
        }
        margins.iter().zip(within).map(|(&m, (s, w))| (m, if cnt == 0 { 0.0 } else { s / cnt as f64 }, w)).collect()
    }
    /// dev: what would storing the cache at `bits` per value cost in accuracy? Rounds the keys and values of positions
    /// [from, to) in place to a per-vector linear grid (the storage stays fp16; nothing gets smaller). kbits / vbits = 0: untouched.
    pub fn kv_quant_sim(&self, from: usize, to: usize, kbits: u32, vbits: u32) {
        assert!(!self.kv8, "this dev tool reads the fp16 cache");
        let h2f = |h: u16| -> f32 { let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 }; let e = ((h >> 10) & 31) as i32; let m = (h & 1023) as f32;
            if e == 0 { sg * m * 2f32.powi(-24) } else { sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15) } };
        let f2h = |v: f32| -> u16 { let bits = v.to_bits(); let sign = ((bits >> 16) & 0x8000) as u16; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            if v == 0.0 || exp <= 0 { sign } else {
                let mant = (bits >> 13) & 0x3ff; let rem = bits & 0x1fff;
                let mut h = ((exp as u32) << 10 | mant) + if rem > 0x1000 || (rem == 0x1000 && mant & 1 == 1) { 1 } else { 0 };
                if h >= 0x7c00 { h = 0x7bff; }
                sign | h as u16 } };
        for slot in 0..N_SLOT {
            let (is_swa, _) = self.slot_kind[slot];
            let (kb, vb, cap) = self.kv_slot(slot);
            let buf = if is_swa { self.kv_swa.borrow() } else { self.kv_full.borrow() };
            let base = buf.contents().as_ptr() as *mut u16;
            for (b, bits) in [(kb, kbits), (vb, vbits)] {
                if bits == 0 { continue; }
                for g in 0..NKV { for pos in from.max(if is_swa { to.saturating_sub(EXT) } else { 0 })..to {
                    let idx = if is_swa { pos % cap as usize } else { pos };
                    // SAFETY: `b` + `(g * cap + idx) * HD + HD` lies within the slot's allocated fp16 KV buffer.
                    let row = unsafe { std::slice::from_raw_parts_mut(base.add(b as usize + (g * cap as usize + idx) * HD), HD) };
                    let mut v: Vec<f32> = row.iter().map(|&h| h2f(h)).collect();
                    dev_helpers::quantize_row_sim(&mut v, bits);
                    for (r, &x) in row.iter_mut().zip(&v) { *r = f2h(x); }
                } }
            }
        }
    }
    /// dev: raw bytes of the K/V cache holding real keys after `n` positions: (sliding-window rings, full-attention rows)
    pub fn kv_dump(&self, n: usize) -> (Vec<u8>, Vec<u8>) {
        assert!(!self.kv8, "this dev tool reads the fp16 cache");
        let (mut swa, mut full) = (Vec::new(), Vec::new());
        for slot in 0..N_SLOT {
            let (is_swa, _) = self.slot_kind[slot];
            let (kb, vb, cap) = self.kv_slot(slot);
            let buf = if is_swa { self.kv_swa.borrow() } else { self.kv_full.borrow() };
            let base = buf.contents().as_ptr() as *const u8;
            let rows = if is_swa { n.min(EXT) } else { n };
            for b in [kb, vb] { for g in 0..NKV {
                let sl = unsafe { std::slice::from_raw_parts(base.add((b as usize + g * cap as usize * HD) * 2), rows * HD * 2) };
                if is_swa { swa.extend_from_slice(sl); } else { full.extend_from_slice(sl); }
            } }
        }
        (swa, full)
    }

    pub fn reset(&self) {
        self.tier_reset();
        unsafe { std::ptr::write_bytes(self.cs.contents().as_ptr() as *mut u8, 0, self.cs.length()); }
    }

    pub fn set_tokens(&self, start: usize, toks: &[u32]) {
        bounds::validate_set_tokens(start, toks, self.max_ctx, self.tok_embd.rows)
            .unwrap_or_else(|e| panic!("{e}"));
        let cap = self.toks.length() / std::mem::size_of::<i32>();
        assert!(
            start + toks.len() <= cap,
            "set_tokens range [{start}..{}) exceeds backing buffer capacity {cap}",
            start + toks.len()
        );
        let p = self.toks.contents().as_ptr() as *mut i32;
        for (i, &t) in toks.iter().enumerate() {
            // SAFETY: `validate_set_tokens` and the backing-buffer check above guarantee
            // `start + i < cap` (`self.toks` holds `cap` `i32`s) and `t <= i32::MAX as u32`.
            unsafe { *p.add(start + i) = t as i32; }
        }
    }
    pub fn arena(&self, off: u32, n: usize) -> Vec<f32> {
        let buf = self.act.borrow();
        let cap_f32 = buf.length() / std::mem::size_of::<f32>();
        bounds::validate_arena_range(off as usize, n, cap_f32)
            .unwrap_or_else(|e| panic!("{e}"));
        // SAFETY: `buf` keeps the `RefCell<Buf>` borrowed and `validate_arena_range`
        // verified `off as usize + n <= cap_f32`.
        unsafe { std::slice::from_raw_parts((buf.contents().as_ptr() as *const f32).add(off as usize), n).to_vec() }
    }
    /// expert ids chosen by each of the 48 layer executions of the last decode step
    pub fn routes(&self) -> Vec<[u8; TOPK]> {
        let v = self.arena(self.ar.sel_log, 48 * 16);
        (0..48).map(|l| { let mut r = [0u8; TOPK]; for k in 0..TOPK { r[k] = v[l * 16 + k] as u8; } r }).collect()
    }
    /// dev: MoE tiles of the last prefilled layer: per width class (128, 64, 32) the per-tile token counts
    pub fn moe_tiles(&self) -> [Vec<u32>; NTC] {
        let a = self.act.borrow().contents().as_ptr() as *const u32;
        let cls = |c: usize| unsafe {
            let n = *a.add(self.ba.get().ind as usize + c * 4) as usize;
            (0..n).map(|i| *a.add(self.ba.get().tl[c] as usize + i * 4 + 2)).collect::<Vec<u32>>()
        };
        [cls(0), cls(1)]
    }
    /// The register epilogues of the fp16 kernels assume how the tensor ops spread their accumulators over threads
    /// (see the comment above k_gemm_sg in prefill.metal).
    fn layout_ok(&self) -> bool {
        {
            let out_off = self.ba.get().yd;
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            e.pso(&self.gpu.pso("k_layout_sg")); e.bytes(&out_off, 5);
            e.dispatch((1, 1, 1), 32);
            e.end(); cb.commit(); cb.waitUntilCompleted();
            let o = unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const i32).add(out_off as usize), 32 * 64) };
            for t in 0..32 {
                if o[t * 64 + 63] != 32 { return false; }
                let rt = |i: usize| (o[t * 64 + i] >> 16, o[t * 64 + i] & 0xffff);
                for q in 0..8 {
                    let (r0, t0) = rt(4 * q);
                    if o[t * 64 + 4 * q] < 0 || r0 % 4 != 0 || t0 >= 16 { return false; }
                    for e in 1..4 { if rt(4 * q + e) != (r0 + e as i32, t0) { return false; } }
                    if q < 4 && (r0 >= 32 || rt(4 * q + 16) != (r0 + 32, t0)) { return false; }
                }
            }
        }
        // attention P.V accumulators: 16 per thread, quads of 4 consecutive dims of one query row
        {
            let out_off = self.ba.get().yd;
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            e.pso(&self.gpu.pso("k_layout_att")); e.bytes(&out_off, 5);
            e.dispatch((1, 1, 1), 256);
            e.end(); cb.commit(); cb.waitUntilCompleted();
            let o = unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const i32).add(out_off as usize), 256 * 64) };
            for t in 0..256 {
                if o[t * 64 + 63] != 16 { return false; }
                for q in 0..4 {
                    let (d0, r0) = (o[t * 64 + 4 * q] >> 16, o[t * 64 + 4 * q] & 0xffff);
                    if o[t * 64 + 4 * q] < 0 || d0 % 4 != 0 { return false; }
                    for e in 1..4 { if o[t * 64 + 4 * q + e] != ((d0 + e as i32) << 16 | r0) { return false; } }
                }
            }
        }
        true
    }
    pub fn token_at(&self, i: usize) -> u32 {
        bounds::validate_token_at(i, self.max_ctx).unwrap_or_else(|e| panic!("{e}"));
        let cap = self.toks.length() / std::mem::size_of::<i32>();
        assert!(i < cap, "token_at index {i} exceeds backing buffer capacity {cap}");
        // SAFETY: `validate_token_at` and the backing-buffer check above guarantee `i < cap`,
        // where `self.toks` holds `cap` `i32` elements.
        unsafe { *(self.toks.contents().as_ptr() as *const i32).add(i) as u32 }
    }
    pub fn logits(&self) -> Logits<'_> {
        Logits::new(self.act.borrow(), self.ar.logits as usize, self.n_vocab)
    }

    /// Read a single logit value directly from the activation buffer without copying.
    /// Returns `f32::NAN` if `tok >= self.n_vocab`.
    #[inline]
    pub fn logit_at(&self, tok: usize) -> f32 {
        if tok >= self.n_vocab {
            return f32::NAN;
        }
        let buf = self.act.borrow();
        let cap_f32 = buf.length() / std::mem::size_of::<f32>();
        let idx = (self.ar.logits as usize)
            .checked_add(tok)
            .expect("logit_at offset overflow");
        assert!(
            idx < cap_f32,
            "logit_at index {idx} exceeds activation buffer capacity {cap_f32} f32s"
        );
        let base = buf.contents().as_ptr() as *const f32;
        // SAFETY: `buf` holds the shared `RefCell<Buf>` borrow for the duration of the read,
        // Metal `StorageModeShared` buffers are page-aligned (>= align_of::<f32>()), and
        // `idx < cap_f32` guarantees `base.add(idx)` is in bounds of `self.act`.
        unsafe { *base.add(idx) }
    }

    /// micro-benchmark a Q4_K matvec variant over the gate_up expert weights of every layer (no cache reuse)
    pub fn kbench(&self, name: &str, nr: usize, nsg: usize) -> f64 {
        #[repr(C)] #[derive(Clone, Copy)]
        struct BenchArgs { w_lo: u32, w_hi: u32, row_bytes: u32, nrows: u32, k: u32, x_off: u32, out_off: u32, pad: u32 }
        let rows = 8 * E; // 8 experts' worth of gate_up rows
        let mut samples = Vec::with_capacity(3);
        for rep in 0..4 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            e.pso(&self.gpu.pso(name));
            let mut bytes = 0usize;
            for l in 0..N_PHYS {
                for chunk in 0..4 {
                    let gu = &self.layers[l].gu;
                    let off = gu.off + ((rep * 4 + chunk) % 16) as u64 * 8 * gu.expert_bytes;
                    let a = BenchArgs { w_lo: off as u32, w_hi: (off >> 32) as u32, row_bytes: gu.row_bytes, nrows: rows as u32, k: E as u32,
                                        x_off: Arena::vec(1), out_off: self.ar.logits, pad: 0 };
                    e.bytes(&a, 5);
                    e.dispatch((rows / (nr * nsg), 1, 1), 32 * nsg);
                    e.barrier();
                    bytes += rows * gu.row_bytes as usize;
                }
            }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            let t = cb.GPUEndTime() - cb.GPUStartTime();
            if rep > 0 && t > 0.0 { samples.push(bytes as f64 / t / 1e9); }
        }
        dev_helpers::best_bandwidth_gbps(samples)
    }

    /// micro-benchmark a MoE-down variant over all layers' down experts (random-ish expert picks)
    pub fn kbench_down(&self, name: &str, rows_per_tg: usize, tg_threads: usize, only_type: u32) -> f64 {
        let ar = self.ar;
        // synthetic act + selection
        let a = self.act.borrow().contents().as_ptr() as *mut f32;
        // SAFETY: `ar.act + TOPK * FF` and `ar.sel + 2 * TOPK` are allocated in the decode arena.
        unsafe {
            for i in 0..TOPK * FF { *a.add(ar.act as usize + i) = ((i * 7919) % 1000) as f32 / 1000.0 - 0.5; }
            for k in 0..TOPK { *a.add(ar.sel as usize + TOPK + k) = 0.125; }
        }
        let mut samples = Vec::with_capacity(3);
        for rep in 0..4 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            let mut bytes = 0usize;
            let mut sels = Vec::new();
            for l in 0..N_PHYS {
                if only_type != 0 && self.layers[l].dn.ty != only_type { continue; }
                for it in 0..4 {
                    let dn = &self.layers[l].dn;
                    let ma = MoeArgs { sel_off: ar.logits + 100000 + ((l * 4 + it) as u32 % 32) * 16, act_off: ar.act, out_off: Arena::vec(2), res_off: Arena::vec(3) as i32,
                                       dn_lo: dn.off as u32, dn_hi: (dn.off >> 32) as u32, dn_expert_bytes: dn.expert_bytes as u32,
                                       dn_row_bytes: dn.row_bytes, dn_type: dn.ty, ..Default::default() };
                    sels.push(ma.sel_off);
                    e.pso(&self.gpu.pso(name));
                    e.bytes(&ma, 5);
                    e.dispatch((E / rows_per_tg, 1, 1), tg_threads);
                    e.barrier();
                    bytes += TOPK * E * dn.row_bytes as usize;
                }
            }
            // SAFETY: each `so` lies within the `ar.logits` scratch region of the decode arena.
            unsafe {
                for (j, &so) in sels.iter().enumerate() {
                    for k in 0..TOPK { *a.add(so as usize + k) = (((j * 37 + k * 13 + rep * 5) % 128) as f32).max(0.0); *a.add(so as usize + TOPK + k) = 0.125; }
                }
            }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            let t = cb.GPUEndTime() - cb.GPUStartTime();
            if rep > 0 && t > 0.0 { samples.push(bytes as f64 / t / 1e9); }
        }
        dev_helpers::best_bandwidth_gbps(samples)
    }

    /// touch every page of the mapped model from the GPU once (wires the GPU page mappings: avoids first-use
    /// page-fault stalls when a rarely used expert is hit for the first time mid-generation)
    pub fn warm(&self) {
        #[repr(C)] #[derive(Clone, Copy)]
        struct TouchArgs { lo: u32, hi: u32, n16: u32, pad: u32 }
        let t0 = std::time::Instant::now();
        let chunk = 1usize << 30;
        let mut off = 0usize;
        let cb = self.gpu.cmd();
        let e = Enc::new(&cb); self.bind(&e);
        e.pso(&self.gpu.pso("k_touch"));
        while off < self.g.file_len {
            let n = chunk.min(self.g.file_len - off) / 16;
            let ta = TouchArgs { lo: off as u32, hi: (off >> 32) as u32, n16: n as u32, pad: 0 };
            e.bytes(&ta, 5); e.dispatch((4096, 1, 1), 256);
            off += chunk;
        }
        e.end(); cb.commit(); cb.waitUntilCompleted();
        eprintln!("[inkling] warmed {:.1} GiB of weights in {:.0?}", self.g.file_len as f64 / (1u64 << 30) as f64, t0.elapsed());
    }

    /// raw GPU read bandwidth over the mapped weights
    pub fn bw_weights(&self) {
        #[repr(C)] #[derive(Clone, Copy)]
        struct TouchArgs { lo: u32, hi: u32, n16: u32, pad: u32 }
        for (gb, tgs) in [(1usize, 1024usize), (2, 2048), (4, 4096), (8, 8192)] {
            let mut best = 0.0f64;
            for rep in 0..3 {
                let off = (1u64 << 30) + rep as u64 * (1u64 << 28);
                let cb = self.gpu.cmd();
                let e = Enc::new(&cb); self.bind(&e);
                let ta = TouchArgs { lo: off as u32, hi: (off >> 32) as u32, n16: ((gb << 30) / 16) as u32, pad: 0 };
                e.pso(&self.gpu.pso("k_touch")); e.bytes(&ta, 5); e.dispatch((tgs, 1, 1), 256);
                e.end(); cb.commit(); cb.waitUntilCompleted();
                best = best.max((gb << 30) as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e9);
            }
            println!("weights read {gb} GB: {best:.1} GB/s");
        }
    }

    /// benchmark the production MoE kernels over every layer with random expert picks (GB/s of expert weights)
    pub fn kbench_moe(&self, which: &str, grid: (usize, usize), tgs: usize) -> f64 {
        let ar = self.ar;
        let a = self.act.borrow().contents().as_ptr() as *mut f32;
        let mut best = 0.0f64;
        for rep in 0..5 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            let mut bytes = 0usize;
            for l in 0..N_PHYS {
                for it in 0..2 {
                    let sel = ar.logits + 100000 + ((l * 2 + it) as u32) * 16;
                    unsafe { for k in 0..TOPK {
                        *a.add(sel as usize + k) = (((l * 2 + it) * 53 + k * 17 + rep * 29) % 128) as f32;
                        *a.add(sel as usize + TOPK + k) = 0.125; } }
                    let ld = &self.layers[l];
                    let ma = MoeArgs { sel_off: sel, x_off: Arena::vec(1), act_off: ar.act, out_off: Arena::vec(2), res_off: Arena::vec(3) as i32,
                        gu_lo: ld.gu.off as u32, gu_hi: (ld.gu.off >> 32) as u32, gu_expert_bytes: ld.gu.expert_bytes as u32, gu_row_bytes: ld.gu.row_bytes,
                        dn_lo: ld.dn.off as u32, dn_hi: (ld.dn.off >> 32) as u32, dn_expert_bytes: ld.dn.expert_bytes as u32, dn_row_bytes: ld.dn.row_bytes,
                        dn_type: ld.dn.ty, beta: 4.0, lbeta: 25.0, cs_off: 0, convw_off: ld.ffn_post_conv, ..Default::default() };
                    e.pso(&self.gpu.pso(which));
                    e.bytes(&ma, 5);
                    e.dispatch((grid.0, grid.1, 1), tgs);
                    e.barrier();
                    bytes += TOPK * if which.contains("gu") { ld.gu.expert_bytes as usize } else { ld.dn.expert_bytes as usize };
                }
            }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            if rep > 0 { best = best.max(bytes as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e9); }
        }
        best
    }

    /// dense GEMM tiling benchmark: QKV-sized Q4_K matrices of all layers, T tokens of random activations
    pub fn kbench_gemm(&self, name: &str, bm: usize, bn: usize, t: usize) -> f64 {
        bounds::validate_kbench_t(t, self.chunk).unwrap_or_else(|e| panic!("{e}"));
        let half_x = name.ends_with("_3") || name.starts_with("kgb") || name.starts_with("kgs");
        let nsg: usize = if name.starts_with("kgb") || name.starts_with("kgs") { name.rsplit('_').next().unwrap().parse().unwrap() } else { 4 };
        let swap = name.starts_with("kgs");
        let max_rows = self.layers.iter().map(|l| l.wq.rows as usize).max().unwrap_or(NH * HD);
        let (out_off, _, total_floats) = bounds::kbench_scratch_layout(t, E, max_rows, half_x, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let scratch = self.gpu.buffer(total_floats * std::mem::size_of::<f32>());
        let a = scratch.contents().as_ptr() as *mut f32;
        // SAFETY: `scratch` holds `total_floats` f32s (>= `out_off` f32s for input); `kbench_scratch_layout`
        // guarantees `t * E` elements (f16 when `half_x`, f32 otherwise) fit within `[0, out_off)` floats.
        unsafe {
            for i in 0..t * E {
                let v = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
                if half_x {
                    // f32 -> f16 bits (simple round-to-nearest for normal range)
                    let bits = v.to_bits(); let sign = (bits >> 16) & 0x8000; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
                    let h: u16 = if v == 0.0 || exp <= 0 { sign as u16 } else { (sign | ((exp as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16 };
                    *(a as *mut u16).add(i) = h;
                } else { *a.add(i) = v; }
            }
        }
        let mut best = 0.0f64;
        for _ in 0..3 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            e.buf(&scratch, 0, 2);
            let tt = t as u32;
            let mut flops = 0.0;
            for l in &self.layers {
                let w = &l.wq;
                let sg = GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
                                in_off: 0, out_off, ldo: w.rows, ..Default::default() };
                e.pso(&self.gpu.pso(name)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                {
                if swap { e.dispatch((t.div_ceil(bn), (w.rows as usize).div_ceil(bm), 1), 32 * nsg); }
                else { e.dispatch(((w.rows as usize).div_ceil(bm), t.div_ceil(bn), 1), 32 * nsg); }
                e.barrier();
                flops += 2.0 * t as f64 * w.rows as f64 * w.k as f64;
                }
            }
            e.end(); cb.commit(); cb.waitUntilCompleted();
            best = best.max(flops / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
        }
        best
    }

    /// dev: after a prefill of `t` tokens, re-run only the MoE GEMM kernels of every layer on the buffers of the last layer
    /// (real routing, gathered inputs already in place): (gate/up TFLOPs, down TFLOPs)
    pub fn kbench_moe_pf(&self, t: usize) -> (f64, f64) {
        let b = self.ba.get();
        let mut res = [0.0f64; 2];
        for which in 0..2 {
            for _ in 0..3 {
                let cb = self.gpu.cmd();
                let e = Enc::new(&cb);
                self.bind(&e);
                for l in &self.layers {
                    for (c, (name, _, tg)) in MOE_CLS.iter().enumerate() {
                        let g = if which == 0 { MoeGemmHArgs {
                            w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
                            wtype: l.gu.ty, nrows: (2 * FF) as u32, k: E as u32, xh_off: b.xgh, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                            out_off: b.acth, ldo: FF as u32, situ: 1, aw_off: b.aw, beta: self.situ_beta, lbeta: self.situ_lbeta,
                        } } else { MoeGemmHArgs {
                            w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
                            wtype: l.dn.ty, nrows: E as u32, k: FF as u32, xh_off: b.acth, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                            out_off: b.yd, ldo: E as u32, situ: 0, aw_off: 0, beta: 0.0, lbeta: 0.0,
                        } };
                        e.pso(&self.gpu.pso(name)); e.bytes(&g, 5);
                        e.dispatch_indirect(&self.act.borrow(), (b.ind as usize + which * 4 * NTC + c * 4) * 4, *tg);
                    }
                    e.barrier();
                }
                e.end(); cb.commit(); cb.waitUntilCompleted();
                let flops = self.layers.len() as f64 * (t * TOPK) as f64 * 2.0 * (E * FF) as f64 * if which == 0 { 2.0 } else { 1.0 };
                res[which] = res[which].max(flops / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
            }
        }
        (res[0], res[1])
    }

    /// per-simdgroup tile benchmark: `nact` of `nsg` simdgroups per tile do the matmul. Returns TFLOPs over the ACTIVE tokens.
    /// tensor-unit ceiling probe (fp16 x fp16 from device memory, no weight staging): TFLOPs
    pub fn kbench_peak(&self, name: &str, nsg: usize, t: usize) -> f64 {
        bounds::validate_kbench_t(t, self.chunk).unwrap_or_else(|e| panic!("{e}"));
        assert!(nsg > 0 && t > 64 && t >= 16 * nsg, "kbench_peak requires nsg > 0, t > 64, t >= 16*nsg (got nsg={nsg}, t={t})");
        let rows = NH * HD;
        let (out_off, _, total_floats) = bounds::kbench_scratch_layout(t, E, rows, true, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let scratch = self.gpu.buffer(total_floats * std::mem::size_of::<f32>());
        let h = scratch.contents().as_ptr() as *mut u16;
        // SAFETY: `scratch` holds `total_floats` f32s; `kbench_scratch_layout` guarantees
        // `t * E` f16 values fit within the input prefix `[0, out_off)` floats.
        unsafe { for i in 0..t * E { *h.add(i) = 0x3800 + ((i * 7919) % 512) as u16; } }
        let mut best = 0.0f64;
        for _ in 0..3 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            e.buf(&scratch, 0, 2);
            let tt = t as u32;
            for _ in 0..8 {
                let sg = GSeg { k: E as u32, in_off: 0, out_off, ldo: rows as u32, nrows: rows as u32, ..Default::default() };
                e.pso(&self.gpu.pso(name)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                e.dispatch((t / (16 * nsg), rows / 64, 1), 32 * nsg);
                e.barrier();
            }
            e.end(); cb.commit(); cb.waitUntilCompleted();
            best = best.max(8.0 * 2.0 * (t * rows * E) as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
        }
        best
    }
    /// precision-mode probe: TFLOPs of a tile-op variant and its error against the exact-accumulation variant (layer 0 wq, realistic x)
    pub fn kbench_mode(&self, name: &str, t: usize) -> (f64, f64, f64) {
        bounds::validate_kbench_t(t, self.chunk).unwrap_or_else(|e| panic!("{e}"));
        assert!(t >= 128, "kbench_mode requires t >= 128 (got t={t})");
        let w = &self.layers[0].wq;
        let (out0, out_stride, total_floats) = bounds::kbench_scratch_layout(t, E, w.rows as usize, true, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        let out1 = out0 + out_stride as u32;
        let scratch = self.gpu.buffer(total_floats * std::mem::size_of::<f32>());
        let a = scratch.contents().as_ptr() as *const f32;
        let h = scratch.contents().as_ptr() as *mut u16;
        // x ~ unit-variance pseudo-normal values, as fp16
        let f16 = |v: f32| -> u16 { let bits = v.to_bits(); let sign = (bits >> 16) & 0x8000; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            if v == 0.0 || exp <= 0 { sign as u16 } else { (sign | ((exp as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16 } };
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 40) as f32 / (1u64 << 24) as f32 };
        // SAFETY: `scratch` holds `total_floats` f32s; `kbench_scratch_layout` guarantees
        // `t * E` f16 inputs fit in `[0, out0)` floats.
        unsafe { for i in 0..t * E { let v = rnd() + rnd() + rnd() + rnd() - 2.0; *h.add(i) = f16(v * 1.7); } }
        let run = |nm: &str, out: u32| -> f64 {
            let mut best = 0.0f64;
            for _ in 0..3 {
                let cb = self.gpu.cmd();
                let e = Enc::new(&cb);
                self.bind(&e);
                e.buf(&scratch, 0, 2);
                let tt = t as u32;
                for _ in 0..8 {
                    let sg = GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
                                    in_off: 0, out_off: out, ldo: w.rows, ..Default::default() };
                    e.pso(&self.gpu.pso(nm)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                    e.dispatch((t / 128, w.rows as usize / 64, 1), 256);
                    e.barrier();
                }
                e.end(); cb.commit(); cb.waitUntilCompleted();
                best = best.max(8.0 * 2.0 * t as f64 * w.rows as f64 * w.k as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
            }
            best
        };
        run("kgmode_0_0", out0);
        let tf = run(name, out1);
        let n = t * w.rows as usize;
        let (mut se, mut sr, mut mx) = (0.0f64, 0.0f64, 0.0f64);
        // SAFETY: GPU work has completed (`waitUntilCompleted` above) and `kbench_scratch_layout`
        // sized `scratch` for `n` f32s at both `out0` and `out1`.
        unsafe { for i in 0..n { let (x, y) = (*a.add(out1 as usize + i) as f64, *a.add(out0 as usize + i) as f64); se += (x - y) * (x - y); sr += y * y; mx = mx.max((x - y).abs()); } }
        (tf, (se / sr).sqrt(), mx / (sr / n as f64).sqrt())
    }
    pub fn kbench_sgn(&self, name: &str, nsg: usize, nact: usize, t: usize) -> f64 {
        bounds::validate_kbench_t(t, self.chunk).unwrap_or_else(|e| panic!("{e}"));
        assert!(nsg > 0 && nact <= nsg && t >= 16 * nsg && t <= 0xFFFF,
                "kbench_sgn invalid params (nsg={nsg}, nact={nact}, t={t})");
        let max_rows = self.layers.iter().map(|l| l.wq.rows as usize).max().unwrap_or(NH * HD);
        let (out_off, _, total_floats) = bounds::kbench_scratch_layout(t, E, max_rows, true, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let scratch = self.gpu.buffer(total_floats * std::mem::size_of::<f32>());
        let a = scratch.contents().as_ptr() as *mut u16;
        // SAFETY: `scratch` holds `total_floats` f32s; `kbench_scratch_layout` guarantees
        // `t * E` f16 inputs fit within `[0, out_off)` floats.
        unsafe { for i in 0..t * E { *a.add(i) = 0x3800 + ((i * 7919) % 512) as u16; } }
        let mut best = 0.0f64;
        for _ in 0..3 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind(&e);
            e.buf(&scratch, 0, 2);
            let tt = (t | nact << 16) as u32;
            let mut flops = 0.0;
            for l in &self.layers {
                let w = &l.wq;
                let sg = GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
                                in_off: 0, out_off, ldo: w.rows, ..Default::default() };
                e.pso(&self.gpu.pso(name)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                e.dispatch((t / (16 * nsg), w.rows as usize / 64, 1), 32 * nsg);
                e.barrier();
                flops += 2.0 * (t * nact / nsg) as f64 * w.rows as f64 * w.k as f64;
            }
            e.end(); cb.commit(); cb.waitUntilCompleted();
            best = best.max(flops / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
        }
        best
    }

    fn bind(&self, e: &Enc) {
        e.buf(&self.w, 0, 0); e.buf(&self.pb, 0, 1); e.buf(&self.act.borrow(), 0, 2); e.buf(&self.kv_swa.borrow(), 0, 3); e.buf(&self.cs, 0, 4); e.buf(&self.toks, 0, 6);
        e.buf(&self.ctr, 0, 7);
    }

    fn seg(w: &QW, in_off: u32, out_off: u32) -> Seg {
        debug_assert!(w.ty == T_Q4_K || w.ty == T_Q6_K);
        Seg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
              in_off, in2_off: -1, norm_off: -1, ss_off: -1, ss_n: 0, out_off, ..Default::default() }
    }
    fn seg_pb(off: u32, rows: u32, k: u32, in_off: u32, out_off: u32) -> Seg {
        Seg { w_lo: off, w_hi: 0, row_bytes: k * 2, wtype: T_BF16, nrows: rows, k, in_off, in2_off: -1, norm_off: -1, ss_off: -1, ss_n: 0, out_off, ..Default::default() }
    }
    #[inline]
    fn go<T>(&self, e: &Enc, cls: u32, pso: &str, args: &T, grid: (usize, usize, usize), tg: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        e.pso(&self.gpu.pso(pso));
        e.bytes(args, 5);
        e.dispatch(grid, tg);
        e.barrier();
    }
    fn mv(&self, e: &Enc, cls: u32, segs: &[Seg]) { self.mv_post(e, cls, segs, MvArgs::default()); }
    fn mv_post(&self, e: &Enc, cls: u32, segs: &[Seg], post: MvArgs) {
        let mut a = MvArgs { nseg: segs.len() as u32, eps: self.eps, ..post };
        let mut tg = 0u32;
        for (i, s) in segs.iter().enumerate() {
            a.seg[i] = *s;
            a.seg[i].tg_start = tg;
            tg += s.nrows.div_ceil(8);
        }
        self.go(e, cls, "k_mv", &a, (tg as usize, 1, 1), 128);
    }
    fn mix(&self, e: &Enc, cls: u32, jobs: &[MixJob]) { self.mix_t(e, cls, jobs, 1); }
    fn mix_t(&self, e: &Enc, cls: u32, jobs: &[MixJob], t: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        dispatch_mix_jobs(&self.gpu, e, jobs, t, std::env::var("INK_OLD_MIXB").is_ok());
    }
    fn mix_job(&self, states: &[usize], q_off: i32, n_q: u32, outs: &[u32], norms: &[i32]) -> MixJob {
        let offs: Vec<u32> = states.iter().map(|&s| Arena::vec(s)).collect();
        self.mix_job_off(&offs, q_off, n_q, outs, norms, 0)
    }
    fn mix_job_off(&self, states: &[u32], q_off: i32, n_q: u32, outs: &[u32], norms: &[i32], tstride: u32) -> MixJob {
        let mut j = MixJob { n_s: states.len() as u32, n_q, q_off, eps: self.eps, tstride, ..Default::default() };
        assert!(states.len() <= 6);
        for (i, &s) in states.iter().enumerate() { j.s_off[i] = s; }
        for i in 0..4 { j.norm_off[i] = -1; j.ncopy[i] = [-1; 3]; j.hcopy[i] = [-1; 3]; j.hout[i] = -1; }
        j.hstride = (2 * E) as u32;
        for (i, &o) in outs.iter().enumerate() { j.out_off[i] = o; }
        for (i, &n) in norms.iter().enumerate() { j.norm_off[i] = n; }
        j
    }
    /// loop controllers, then RMSNorm of each controlled vector (norm weight w, written to out)
    /// decode loop controllers: g = D.cat[j] (bf16 matvec on the mix kernel's normalized copies), then gain/forget apply
    /// with the RMSNorm fused into the last threadgroup of each controller
    fn ctrl_dec(&self, e: &Enc, jobs: &[CtrlJob]) {
        let ar = self.ar;
        let segs: Vec<Seg> = jobs.iter().map(|j| {
            let c = ar.ctrlg.iter().position(|&o| o == j.out_off).unwrap();
            Self::seg_pb(j.d_off, 64, (2 * E) as u32, ar.cat[c], j.out_off)
        }).collect();
        self.mv(e, C_CTRLD, &segs);
        let mut a = CtrlArgs { n_c: jobs.len() as u32, eps: self.eps, ..Default::default() };
        for (i, j) in jobs.iter().enumerate() { a.job[i] = *j; }
        self.go(e, C_CTRLA, "k_ctrl_apply", &a, (E / 32, jobs.len(), 1), 256);
    }

    /// Encode one full decode step for token at `pos` (input token read from toks[pos]).
    /// If `logits` is set, also computes lm_head and writes argmax to toks[pos+1].
    pub fn encode_step(&self, cb: &CmdBuf, pos: usize, logits: bool, argmax: bool) {
        assert!(!self.tier_on(), "with the disk tier a step spans several command buffers: use step()");
        assert!(pos < self.kv_capacity(), "K/V capacity: call reserve() first");
        let mut cx = Cx { e: Enc::new(cb), cb: None };
        self.encode_step_cx(&mut cx, pos, logits, argmax);
        cx.e.end();
    }
    fn encode_step_cx(&self, cx: &mut Cx, pos: usize, logits: bool, argmax: bool) {
        let e = &cx.e;
        self.bind(e);
        let ar = self.ar;
        // embedding -> vec 0 (x0)
        let ea = EmbedArgs { emb_lo: self.tok_embd.off as u32, emb_hi: (self.tok_embd.off >> 32) as u32, row_bytes: self.tok_embd.row_bytes,
                             norm_off: self.tok_norm, out0: Arena::vec(0), out1: Arena::vec(0), tok_index: pos as u32, eps: self.eps };
        self.go(e, C_EMBED, "k_embed", &ea, (1, 1, 1), 256);

        self.done.set(0);
        let mut va = VecAlloc { next: 1, n: N_VEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for ip in 0..LOOP_START { cur = self.layer(cx, pos, ip, ip, cur, &mut blocks, None, &mut va, &[]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = self.layer(cx, pos, slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned);
            }
            bank = (*blocks.get(prelude_blocks.len() + 1).unwrap_or(&0), cur);
        }
        for ip in LOOP_END..N_PHYS { cur = self.layer(cx, pos, ip, ip, cur, &mut blocks, None, &mut va, &[]); }

        let e = &cx.e;
        if logits {
            let mut st = blocks.clone(); st.push(cur);
            let j = self.mix_job(&st, self.output_res_query as i32, 1, &[ar.fout], &[self.output_norm as i32]);
            self.mix(e, C_OMIX, &[j]);
            self.mv(e, C_LMHEAD, &[Self::seg(&self.output, ar.fout, ar.logits)]);
        }
        if logits && argmax {
            let nparts = 128u32;
            let am = ArgmaxArgs { logits_off: ar.logits, n: self.n_vocab as u32, part_off: ar.amax, nparts, out_index: pos as u32 + 1, pad: [0; 3] };
            self.go(e, C_ARGMAX, "k_argmax1", &am, (nparts as usize, 1, 1), 1024);
            self.go(e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, cx: &mut Cx, pos: usize, slot: usize, ip: usize, cur: usize, blocks: &mut Vec<usize>,
             msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize]) -> usize {
        if self.done.get() >= self.stop.get() { return cur; }
        self.done.set(self.done.get() + 1);
        let e = &cx.e;
        let l = &self.layers[ip];
        let ar = self.ar;
        let ctl = msg.map(|_| &self.loops[ip - LOOP_START]);
        let cs_slot = (slot * NCH * 3) as u32;

        // ---- attention inputs: depth mix (+ loop control) -> normalized q/k/v inputs
        let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
        let mut st = blocks.clone(); st.push(cur);
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        let xin: [u32; 3];
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            // raw mixed inputs u_j -> inp[j]; RMSNorm'd copies -> cat[j][0:E]; messages m_j -> msg[j], normalized -> cat[j][E:2E]
            let mut ja = self.mix_job(&st, l.attn_res_query, nq, &ar.inp[..nq as usize], &[]);
            if l.swa { ja.ncopy[0] = [ar.cat[0] as i32, ar.cat[1] as i32, ar.cat[2] as i32]; }
            else { for j in 0..3 { ja.ncopy[j][0] = ar.cat[j] as i32; } }
            let mut jm = self.mix_job(&[b0, b1], lw.msg_query as i32, 4, &ar.msg, &[]);
            for j in 0..4 { jm.ncopy[j][0] = (ar.cat[j] + E as u32) as i32; }
            self.mix(e, C_MIXA, &[ja, jm]);
            let mut cj = Vec::new();
            for j in 0..3 {
                let u = if l.swa { ar.inp[0] } else { ar.inp[j] };
                cj.push(CtrlJob { u_off: u, m_off: ar.msg[j], d_off: lw.down[j], out_off: ar.ctrlg[j], gain_off: lw.gain[j],
                                  forget_off: lw.forget[j], res_out: ar.cin[j], ss_off: ar.ssp[j], norm_w: norms[j], ctr: 8 + j as u32 });
            }
            self.ctrl_dec(e, &cj);
            xin = ar.cin;
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            self.mix(e, C_MIXA, &[self.mix_job(&st, l.attn_res_query, nq, &ar.inp[..nq as usize], &nn)]);
            xin = if l.swa { [ar.inp[0]; 3] } else { [ar.inp[0], ar.inp[1], ar.inp[2]] };
        }
        // ---- fused projections (one dispatch)
        let mut segs = vec![Self::seg(&l.wq, xin[0], ar.q), Self::seg(&l.wgate, xin[0], ar.gate), Self::seg(&l.wrel, xin[0], ar.rel),
                            Self::seg(&l.wsink, xin[0], ar.sink), Self::seg(&l.wk, xin[1], ar.k), Self::seg(&l.wv, xin[2], ar.v)];
        if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) { segs.push(Self::seg(kp, xin[1], ar.kdyn)); segs.push(Self::seg(vp, xin[2], ar.vdyn)); }
        self.mv(e, C_QKV, &segs);
        // ---- attention (split-K) with the split combine + output gate fused into the last split per kv group
        let (kb, vb, kcap) = self.kv_slot(slot);
        let tau = if l.swa { 1.0 } else { 1.0 + self.log_alpha * (((pos + 1) as f32 / self.log_floor).max(1.0)).ln() };
        let base = AttnArgs {
            pos: pos as u32, cap: kcap, k_base: kb, v_base: vb, cs_off: cs_slot,
            q_off: ar.q, k_off: ar.k, v_off: ar.v, r_off: ar.rel, sink_off: ar.sink,
            kdyn_off: if l.swa { -1 } else { ar.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { ar.vdyn as i32 }, part_off: ar.part,
            kconv_w: l.k_conv, vconv_w: l.v_conv, qclip_w: l.q_clip, relclip_w: l.rel_clip, relbias_w: l.rel_bias, tau,
            ctr: 0, gate_off: ar.gate, out_off: ar.attn_g, gstr: kcap * HD as u32, istr: HD as u32, ..Default::default()
        };
        let nsplit;
        if self.tier_on() {
            // disk tier: the keys older than the hot ring come from the file (staged in chunks, each its own dispatch writing
            // its share of the split partials); the hot ring is one more dispatch and also stores this token's K/V
            let t = self.tier.as_ref().unwrap();
            let exec = self.done.get() - 1;
            let x = t.execs.borrow()[exec];
            let chunks: Vec<(usize, Chunk)> = t.plan.borrow().iter().cloned().enumerate().filter(|(_, c)| c.exec == exec).collect();
            for (k, c) in chunks {
                self.tier_load(cx, k);
                let e = &cx.e;
                let off = ((t.half.get() * (t.stg.borrow().length() / 2) + c.off) / 2) as u32;       // in fp16 elements
                let a = AttnArgs { nkeys: (c.c1 - c.c0) as u32, first_pos: c.c0 as u32, ring: 0, idx0: c.c0 as u32, k_base: off, v_base: off + (NKV * HD) as u32,
                                   gstr: HD as u32, istr: (2 * NKV * HD) as u32, kps: c.kps as u32, nsplit: x.nsplit as u32, sp0: c.sp0 as u32, noown: 1, pad: 1, ..base };
                e.buf(&t.stg.borrow(), 0, 3);
                self.go(e, C_ATTN, "k_attn", &a, (c.ns, NKV, 1), 128);
            }
            let e = &cx.e;
            let a = AttnArgs { nkeys: (pos + 1 - x.b) as u32, first_pos: x.b as u32, ring: 1, kps: x.kps as u32, nsplit: x.nsplit as u32, sp0: x.sp0 as u32, pad: 1, ..base };
            self.bind_kv(e, l.swa);
            self.go(e, C_ATTN, "k_attn", &a, (x.ns, NKV, 1), 128);
            let total = x.nsplit;
            nsplit = total;
        } else if let (false, Some(pr)) = (l.swa, self.prune.borrow().as_ref()) {
            // dev: far keys from the selection, recent keys from the cache
            let fi = self.slot_kind[slot].1;
            let spl = |nk: usize| { let kps = Self::split_kps(nk); (kps, nk.div_ceil(kps)) };
            let nb = if pr.use_sum { pr.rows } else { pr.rows - 1 };
            let ((k1, n1), (k2, n2)) = (if nb > 0 { spl(nb) } else { (32, 0) }, spl(pos + 1 - pr.r0));
            if nb > 0 {
                let kb2 = (fi * 2 * NKV * pr.rows * HD) as u32;
                let a = AttnArgs { nkeys: nb as u32, first_pos: 0, ring: 0, cap: pr.rows as u32, k_base: kb2, v_base: kb2 + (NKV * pr.rows * HD) as u32,
                                   gstr: (pr.rows * HD) as u32, kps: k1 as u32, nsplit: (n1 + n2) as u32, sp0: 0, noown: 1, pad: 1,
                                   lastbias: if pr.use_sum { pr.lastbias } else { 0.0 }, ..base };
                e.buf(&pr.bag, 0, 3);
                self.go(e, C_ATTN, "k_attn", &a, (n1, NKV, 1), 128);
            }
            let a = AttnArgs { nkeys: (pos + 1 - pr.r0) as u32, first_pos: pr.r0 as u32, ring: 0, kps: k2 as u32, nsplit: (n1 + n2) as u32, sp0: n1 as u32, pad: 1, ..base };
            self.bind_kv(e, false);
            self.go(e, C_ATTN, "k_attn", &a, (n2, NKV, 1), 128);
            nsplit = n1 + n2;
        } else {
            // ---- attention (split-K) with the split combine + output gate fused into the last split per kv group
            self.bind_kv(e, l.swa);
            let nkeys = if l.swa { (pos + 1).min(EXT) } else { pos + 1 };
            let kps = Self::split_kps(nkeys);
            nsplit = nkeys.div_ceil(kps);
            let aa = AttnArgs { nkeys: nkeys as u32, first_pos: (pos + 1 - nkeys) as u32, ring: l.swa as u32, kps: kps as u32, nsplit: nsplit as u32,
                                pad: self.sep_comb as u32, ..base };
            self.go(e, C_ATTN, "k_attn", &aa, (nsplit, NKV, 1), 128);
        }
        let e = &cx.e;
        if self.sep_comb || self.tier_on() || self.prune.borrow().is_some() {
            let ca = CombineArgs { part_off: ar.part, nsplit: nsplit as u32, gate_off: ar.gate, out_off: ar.attn_g };
            self.go(e, C_COMB, "k_attn_combine", &ca, (NH, 1, 1), 128);
        }
        // ---- output projection + post-conv (+ residual unless this layer starts a new block)
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone(); live.push(cur); live.extend_from_slice(pinned);
        let mid = va.alloc(&live);
        let mut so = Self::seg(&l.wo, ar.attn_g, Arena::vec(mid));
        so.epi = 1; so.res_off = if block_start { -1 } else { Arena::vec(cur) as i32 };
        so.cs_off = cs_slot + (2 * NKV * HD * 3) as u32; so.convw_off = l.attn_post_conv;
        self.mv(e, C_WO, &[so]);
        if block_start { blocks.push(cur); }
        // ---- FFN input: depth mix (+ control) -> normalized
        let mut st = blocks.clone(); st.push(mid);
        let x_off = if let Some(lw) = ctl {
            let mut jf = self.mix_job(&st, l.ffn_res_query as i32, 1, &[ar.fin], &[]);
            jf.ncopy[0][0] = ar.cat[3] as i32;
            self.mix(e, C_MIXF, &[jf]);
            self.ctrl_dec(e, &[CtrlJob { u_off: ar.fin, m_off: ar.msg[3], d_off: lw.down[3], out_off: ar.ctrlg[3], gain_off: lw.gain[3],
                                         forget_off: lw.forget[3], res_out: ar.finc, ss_off: ar.ssp[3], norm_w: l.ffn_norm as i32, ctr: 11 }]);
            ar.finc
        } else {
            self.mix(e, C_MIXF, &[self.mix_job(&st, l.ffn_res_query as i32, 1, &[ar.fin], &[l.ffn_norm as i32])]);
            ar.fin
        };
        // ---- router with fused top-k (last threadgroup)
        let sel = ar.sel_log + ((self.done.get() - 1) * 16) as u32;
        let rpost = MvArgs { post: (!self.sep_topk) as u32, ctr: 12, bias_off: l.exp_bias, sel_off: sel, wscale: self.wscale, ..Default::default() };
        self.mv_post(e, C_ROUTER, &[Self::seg_pb(l.router, NEXP as u32, E as u32, x_off, ar.rlog)], rpost);
        let mut live: Vec<usize> = blocks.clone(); live.push(mid); live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        let topk_args = MoeArgs { logits_off: ar.rlog, bias_off: l.exp_bias, sel_off: sel, wscale: self.wscale, ..Default::default() };
        if self.sep_topk { self.go(e, C_ROUTER, "k_topk", &topk_args, (1, 1, 1), 32); }
        let ma = MoeArgs {
            logits_off: ar.rlog, bias_off: l.exp_bias, sel_off: sel, x_off,
            norm_off: -1, ss_off: -1, ss_n: 0, eps: self.eps,
            gu_lo: l.gu.off as u32, gu_hi: (l.gu.off >> 32) as u32, gu_expert_bytes: l.gu.expert_bytes as u32, gu_row_bytes: l.gu.row_bytes,
            act_off: ar.act, wscale: self.wscale, beta: self.situ_beta, lbeta: self.situ_lbeta,
            dn_lo: l.dn.off as u32, dn_hi: (l.dn.off >> 32) as u32, dn_expert_bytes: l.dn.expert_bytes as u32, dn_row_bytes: l.dn.row_bytes,
            dn_type: l.dn.ty, out_off: Arena::vec(out), res_off: Arena::vec(mid) as i32, cs_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32,
            convw_off: l.ffn_post_conv, pad: [0; 3],
        };
        // ---- experts
        self.go(e, C_GU, "k_moe_gu", &ma, (FF / 4, TOPK, 1), 128);
        self.go(e, C_DOWN, "k_moe_down", &ma, (E / 4, 1, 1), 64);
        self.last_out.set(out); self.last_mid.set(mid);
        out
    }

    // ======================================================================== batched prefill
    fn gemm(&self, e: &Enc, cls: u32, segs: &[GSeg], t: usize) {
        let mut a = GemmArgs { nseg: segs.len() as u32, t: t as u32, ..Default::default() };
        let mut tiles = 0u32;
        for (i, sg) in segs.iter().enumerate() { a.seg[i] = *sg; a.seg[i].tile_start = tiles; tiles += sg.nrows.div_ceil(64); }
        self.go(e, cls, if self.tensor { "k_gemm_t" } else { "k_gemm" }, &a, (tiles as usize, t.div_ceil(32), 1), 128);
    }
    /// batched loop controllers as GEMMs: g = D.[u^;m^] (split-input GEMM over the mix kernel's normalized copies),
    /// gain/forget pre-activations = G.g, F.g (K=64 GEMM), then a fused apply + RMSNorm per token
    /// jobs: (controller index c, loop weights, u_off, m_off, out_off, norm weight)
    /// fp16 path: the mix kernels already wrote [u^ | m^] as half into cath; `hout[i]` receives job i's output as half
    /// (`nof32[i]`: only as half)
    fn ctrl_b(&self, e: &Enc, lw: &LoopW, jobs: &[(usize, u32, u32, u32, u32)], t: usize, hout: &[(u32, bool)]) {
        let b = self.ba.get();
        let pb = |off: u32, rows: u32, k: u32, in_off: u32, in2: u32, out: u32| GSeg {
            w_lo: off, w_hi: 0, row_bytes: k * 2, wtype: T_BF16, nrows: rows, k, in_off, out_off: out, ldo: rows, wsrc: 1, in2, ..Default::default() };
        if self.hgemm {
            let down: Vec<HSeg> = jobs.iter().map(|&(c, _, _, _, _)| Self::hseg_pb(lw.down[c], 64, (2 * E) as u32, b.cath[c], b.ctrlg[c])).collect();
            self.gemm_h(e, C_CTRLD, &down, t);
            for &(c, _, _, _, _) in jobs { self.tohalf(e, C_CTRLA, b.ctrlg[c], 64, b.gh[c], 64, 64, t, None); }
            e.barrier();
            // gain/forget GEMM with the apply fused into its epilogue: raw (un-normalized) outputs land in gain_o
            let mut ga = CtrlGfArgs { t: t as u32, njobs: jobs.len() as u32, ..Default::default() };
            for (i, &(c, u, m, _, _)) in jobs.iter().enumerate() {
                ga.job[i] = CtrlGfJob { gain_w: lw.gain[c], forget_w: lw.forget[c], gh_off: b.gh[c], u_off: u, m_off: m, raw_off: b.gain_o[c], pad: [0; 2] };
            }
            self.go(e, C_CTRLA, "k_ctrl_gf", &ga, (t.div_ceil(128), jobs.len() * (E / 32), 1), 256);
        } else {
            let down: Vec<GSeg> = jobs.iter().map(|&(c, _, _, _, _)| pb(lw.down[c], 64, (2 * E) as u32, b.cat_u[c], b.cat_m[c], b.ctrlg[c])).collect();
            self.gemm(e, C_CTRLD, &down, t);
            let mut gf = Vec::new();
            for &(c, _, _, _, _) in jobs {
                gf.push(pb(lw.gain[c], E as u32, 64, b.ctrlg[c], 0, b.gain_o[c]));
                gf.push(pb(lw.forget[c], E as u32, 64, b.ctrlg[c], 0, b.forget_o[c]));
            }
            self.gemm(e, C_CTRLA, &gf, t);
        }
        let mut a = CtrlBArgs { n: jobs.len() as u32, eps: self.eps, ..Default::default() };
        for (i, &(c, u, m, out, nw)) in jobs.iter().enumerate() {
            let (h, nof) = hout.get(i).map(|&(h, n)| (h as i32, n as u32)).unwrap_or((-1, 0));
            a.job[i] = if self.hgemm { CtrlBJob { u_off: b.gain_o[c], m_off: u32::MAX, g_off: 0, f_off: 0, out_off: out, norm_w: nw, h_off: h, nof32: nof } }
                       else { CtrlBJob { u_off: u, m_off: m, g_off: b.gain_o[c], f_off: b.forget_o[c], out_off: out, norm_w: nw, h_off: h, nof32: nof } };
        }
        self.go(e, C_CTRLN, "k_ctrl_apply_b", &a, (t, jobs.len(), 1), 256);
    }

    /// f32 [rows][k] (row stride src_stride, optional row gather) -> half [rows][dst_stride] at dst
    #[allow(clippy::too_many_arguments)]
    fn tohalf(&self, e: &Enc, cls: u32, src: u32, src_stride: usize, dst: u32, dst_stride: usize, k: usize, rows: usize, perm: Option<u32>) {
        self.tohalf_x(e, cls, src, src_stride, dst, dst_stride, k, rows, perm, false)
    }
    /// `half_src`: the source is already half (row stride in halves): a plain (gathered) copy
    #[allow(clippy::too_many_arguments)]
    fn tohalf_x(&self, e: &Enc, cls: u32, src: u32, src_stride: usize, dst: u32, dst_stride: usize, k: usize, rows: usize, perm: Option<u32>, half_src: bool) {
        let a = ToHalfArgs { src_off: src, dst_off: dst, k: k as u32, rows: rows as u32, perm_off: perm.unwrap_or(0), gather: perm.is_some() as u32 | (half_src as u32) << 1,
                             src_stride: src_stride as u32, dst_stride: dst_stride as u32 };
        if self.skip.get() & (1 << cls) != 0 { return; }
        e.pso(&self.gpu.pso("k_tohalf")); e.bytes(&a, 5); e.dispatch((k.div_ceil(1024), rows, 1), 256);
    }
    /// fp16-activation tensor GEMM over segments (weight, wsrc, half input, output, ldo); barrier after
    fn gemm_h(&self, e: &Enc, cls: u32, segs: &[HSeg], t: usize) {
        let mut a = GemmHArgs { nseg: segs.len() as u32, t: t as u32, ..Default::default() };
        let mut tiles = 0u32;
        for (i, sg) in segs.iter().enumerate() { a.seg[i] = *sg; a.seg[i].tile_start = tiles; tiles += sg.nrows.div_ceil(64); }
        if t > 64 { self.go(e, cls, "k_gemm_sg8", &a, (t.div_ceil(128), tiles as usize, 1), 256); }
        else { self.go(e, cls, "k_gemm_sg4", &a, (1, tiles as usize, 1), 128); }
    }
    fn hseg(w: &QW, xh: u32, out: u32) -> HSeg {
        HSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k, xh_off: xh, out_off: out, ldo: w.rows, ..Default::default() }
    }
    fn hseg_pb(off: u32, rows: u32, k: u32, xh: u32, out: u32) -> HSeg {
        HSeg { w_lo: off, w_hi: 0, row_bytes: k * 2, wtype: T_BF16, nrows: rows, k, xh_off: xh, out_off: out, ldo: rows, wsrc: 1, ..Default::default() }
    }

    fn gseg(w: &QW, in_off: u32, out_off: u32) -> GSeg {
        GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
               in_off, out_off, ldo: w.rows, ..Default::default() }
    }

    /// Encode prefill of `t` tokens (toks[p0..p0+t]) as one batch. Optionally computes logits/argmax for the last token.
    pub fn encode_prefill(&self, cb: &CmdBuf, p0: usize, t: usize, logits: bool, argmax: bool) {
        assert!(t >= 1 && t <= self.chunk && p0 + t <= self.kv_capacity(), "K/V capacity: call reserve() first");
        assert!(t <= self.ba.get().cap, "batch arena: call ensure_batch() first");
        let e = Enc::new(cb);
        self.bind(&e);
        let b = self.ba.get();
        let ea = EmbedArgs { emb_lo: self.tok_embd.off as u32, emb_hi: (self.tok_embd.off >> 32) as u32, row_bytes: self.tok_embd.row_bytes,
                             norm_off: self.tok_norm, out0: b.vec(0), out1: b.vec(0), tok_index: p0 as u32, eps: self.eps };
        self.go(&e, C_EMBED, "k_embed", &ea, (t, 1, 1), 256);
        // Last layer: nothing after the batch reads its outputs except the final token (logits) and the conv states (last 3
        // rows of the two branch outputs). It is a full-attention layer, so K/V still cover every token, but queries, attention,
        // output projection and FFN only run from row t-8: the conv/residual chain makes rows >= t-2 of the result exact.
        let mut trim = [0usize; N_PHYS];
        if self.hgemm && self.tail.get() && t > 16 { trim[N_PHYS - 1] = t - 8; }
        let mut va = VecAlloc { next: 1, n: B_NVEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for ip in 0..LOOP_START { cur = self.layer_b(&e, p0, t, ip, ip, cur, &mut blocks, None, &mut va, &[], trim[ip]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END { cur = self.layer_b(&e, p0, t, slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned, 0); }
            bank = (blocks[prelude_blocks.len() + 1], cur);
        }
        for ip in LOOP_END..N_PHYS { cur = self.layer_b(&e, p0, t, ip, ip, cur, &mut blocks, None, &mut va, &[], trim[ip]); }
        if logits {
            let ar = self.ar;
            let last = ((t - 1) * E) as u32;
            let mut st: Vec<u32> = blocks.iter().map(|&v| b.vec(v) + last).collect();
            st.push(b.vec(cur) + last);
            let j = self.mix_job_off(&st, self.output_res_query as i32, 1, &[ar.fout], &[self.output_norm as i32], 0);
            self.mix(&e, C_OMIX, &[j]);
            self.mv(&e, C_LMHEAD, &[Self::seg(&self.output, ar.fout, ar.logits)]);
            if argmax {
                let nparts = 128u32;
                let am = ArgmaxArgs { logits_off: ar.logits, n: self.n_vocab as u32, part_off: ar.amax, nparts, out_index: (p0 + t) as u32, pad: [0; 3] };
                self.go(&e, C_ARGMAX, "k_argmax1", &am, (nparts as usize, 1, 1), 1024);
                self.go(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
            }
        }
        e.end();
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_b(&self, e: &Enc, p0: usize, t: usize, slot: usize, ip: usize, cur: usize, blocks: &mut Vec<usize>,
               msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize], ts: usize) -> usize {
        let l = &self.layers[ip];
        let b = self.ba.get();
        let vo = |i: usize| b.vec(i);
        let ctl = msg.map(|_| &self.loops[ip - LOOP_START]);
        let cs_slot = (slot * NCH * 3) as u32;
        let tt = t as u32;
        let es = E as u32;
        let hatt = self.hgemm;      // fp16 path: k_attn_h, half q / gate, router on the fp16 FFN input
        let (qgh, rth) = (hatt, hatt);
        // the query side (queries, attention, output projection, FFN) covers rows [ts, t); the key side always all rows
        assert!(ts == 0 || (ctl.is_none() && hatt));
        let n = t - ts;
        let nn_ = n as u32;
        let sh = |off: u32, dim: usize| off + (ts * dim) as u32;
        // ---- attention inputs
        let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
        let mut st: Vec<u32> = blocks.iter().map(|&v| vo(v)).collect(); st.push(vo(cur));
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        let xin: [u32; 3];
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            let mut ja = self.mix_job_off(&st, l.attn_res_query, nq, &b.inp[..nq as usize], &[], es);
            let mut jm = self.mix_job_off(&[vo(b0), vo(b1)], lw.msg_query as i32, 4, &b.msg, &[], es);
            if self.hgemm {
                // normalized copies straight into the controllers' fp16 inputs [u^ | m^]
                if l.swa { ja.hcopy[0] = [b.cath[0] as i32, b.cath[1] as i32, b.cath[2] as i32]; }
                else { for j in 0..3 { ja.hcopy[j][0] = b.cath[j] as i32; } }
                for j in 0..4 { jm.hcopy[j][0] = (b.cath[j] + (E / 2) as u32) as i32; }
            } else {
                if l.swa { ja.ncopy[0] = [b.cat_u[0] as i32, b.cat_u[1] as i32, b.cat_u[2] as i32]; }
                else { for j in 0..3 { ja.ncopy[j][0] = b.cat_u[j] as i32; } }
                for j in 0..4 { jm.ncopy[j][0] = b.cat_m[j] as i32; }
            }
            self.mix_t(e, C_MIXA, &[ja, jm], t);
            let jobs: Vec<(usize, u32, u32, u32, u32)> = (0..3).map(|j| (j, if l.swa { b.inp[0] } else { b.inp[j] }, b.msg[j], b.cin[j], norms[j] as u32)).collect();
            let ho: Vec<(u32, bool)> = if self.hgemm { (0..3).map(|j| (b.xh[j], true)).collect() } else { vec![] };
            self.ctrl_b(e, lw, &jobs, t, &ho);
            xin = b.cin;
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            let mut ja = self.mix_job_off(&st, l.attn_res_query, nq, &b.inp[..nq as usize], &nn, es);
            if self.hgemm { for q in 0..nq as usize { ja.hout[q] = b.xh[q] as i32; } ja.nof32 = 0xF; }
            self.mix_t(e, C_MIXA, &[ja], t);
            xin = if l.swa { [b.inp[0]; 3] } else { [b.inp[0], b.inp[1], b.inp[2]] };
        }
        // ---- projections (one batched GEMM over all segments)
        let mut segs = vec![Self::gseg(&l.wq, xin[0], b.q), Self::gseg(&l.wgate, xin[0], b.gate), Self::gseg(&l.wrel, xin[0], b.rel),
                            Self::gseg(&l.wsink, xin[0], b.sink), Self::gseg(&l.wk, xin[1], b.k), Self::gseg(&l.wv, xin[2], b.v)];
        if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) { segs.push(Self::gseg(kp, xin[1], b.kdyn)); segs.push(Self::gseg(vp, xin[2], b.vdyn)); }
        if self.hgemm {
            // the fp16 inputs b.xh[..] were written by the mix / controller kernels: one shared input in an uncontrolled
            // sliding-window layer, separate q / k / v inputs otherwise
            let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
            let mut hq = vec![Self::hseg(&l.wq, b.xh[xi[0]], b.q), Self::hseg(&l.wgate, b.xh[xi[0]], b.gate), Self::hseg(&l.wrel, b.xh[xi[0]], b.rel),
                              Self::hseg(&l.wsink, b.xh[xi[0]], b.sink)];
            // q and the output gate are only read once more (K/V prep, attention epilogue): write them as half
            if hatt && qgh { hq[0].hout = 1; hq[1].hout = 1; }
            let mut hs = vec![Self::hseg(&l.wk, b.xh[xi[1]], b.k), Self::hseg(&l.wv, b.xh[xi[2]], b.v)];
            if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) { hs.push(Self::hseg(kp, b.xh[xi[1]], b.kdyn)); hs.push(Self::hseg(vp, b.xh[xi[2]], b.vdyn)); }
            if ts == 0 { hq.extend_from_slice(&hs); self.gemm_h(e, C_QKV, &hq, t); }
            else {
                // query-side projections for the tail rows only (the two GEMMs are independent)
                for sg in hq.iter_mut() { sg.xh_off = sh(sg.xh_off, E / 2); sg.out_off = sh(sg.out_off, sg.nrows as usize / (1 + sg.hout as usize)); }
                self.gemm_h(e, C_QKV, &hs, t);
                self.gemm_h(e, C_QKV, &hq, n);
            }
        } else {
            self.gemm(e, C_QKV, &segs, t);
        }
        // ---- K/V conv + cache write, Q/R prep
        // Sliding-window layers: the batch attends through a linear scratch [window history | batch keys] in the arena, and the
        // ring (exactly one window) only receives the last window of keys. Full layers address their cache directly.
        let (kb, vb, kcap) = self.kv_slot(slot);
        let scap = (EXT + b.cap + 2 * KGUARD) as u32;
        let (pk, pv, cap, kpos0) = if l.swa { (b.sk * (4 / self.kvb) as u32, b.sv * (4 / self.kvb) as u32, scap, p0 as i32 - (EXT + KGUARD) as i32) } else { (kb, vb, kcap, 0) };
        let kcopy = |n: usize, pos0: usize, to_ring: u32| {
            let a = KvCopyArgs { n: n as u32, pos0: pos0 as u32, ring_k: kb, ring_v: vb, cap: kcap, sk_off: b.sk, sv_off: b.sv, scap, kpos0, to_ring, pad: [0; 2] };
            e.buf(&self.kv_swa.borrow(), 0, 3);
            self.go(e, C_KVPREP, "k_kvcopy", &a, (n, NKV, 1), 32);
            e.buf(&self.act.borrow(), 0, 3);
        };
        if l.swa { if p0 > 0 { kcopy(p0.min(EXT), p0 - p0.min(EXT), 0); } e.buf(&self.act.borrow(), 0, 3); } else { self.bind_kv(e, false); }
        let ka = KvPrepArgs {
            t: tt, p0: p0 as u32, ring: 0, cap, k_base: pk, v_base: pv, cs_off: cs_slot, q_off: b.q,
            k_off: b.k, v_off: b.v, r_off: b.rel, kdyn_off: if l.swa { -1 } else { b.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { b.vdyn as i32 },
            qp_off: b.qp, rp_off: b.rp, qhalf: hatt as u32 | ((hatt && qgh) as u32) << 1, kconv_w: l.k_conv, vconv_w: l.v_conv, qclip_w: l.q_clip, relclip_w: l.rel_clip,
            log_alpha: self.log_alpha, log_floor: self.log_floor, is_full: (!l.swa) as u32, kpos0,
        };
        self.go(e, C_KVPREP, "k_kvprep_b", &ka, (t, NKV, 1), 128);
        if let (false, Some((qc, ncap))) = (l.swa, self.qcap.borrow().as_ref()) {
            assert!(hatt && ts == 0 && p0 + t <= *ncap, "query capture needs the fp16 attention path, no tail trimming and room");
            let (fi, per) = (self.slot_kind[slot].1, ncap * (NH * HD / 2 + NH));
            e.buf(qc, 0, 3);
            let a = DevCopyArgs { src: b.qp, dst: (fi * per + p0 * (NH * HD / 2)) as u32, n: (t * NH * HD / 2) as u32, pad: 0 };
            self.go(e, C_KVPREP, "k_devcopy", &a, ((t * NH * HD / 2).div_ceil(1024), 1, 1), 1024);
            let a = DevCopyArgs { src: b.sink, dst: (fi * per + ncap * (NH * HD / 2) + p0 * NH) as u32, n: (t * NH) as u32, pad: 0 };
            self.go(e, C_KVPREP, "k_devcopy", &a, ((t * NH).div_ceil(1024), 1, 1), 1024);
            self.bind_kv(e, false);
        }
        if l.swa { kcopy(t.min(EXT), p0 + t - t.min(EXT), 1); }
        let su = StateArgs { x_off: b.k, ldx: (NKV * HD) as u32, cs_off: cs_slot, nch: (NKV * HD) as u32, t: tt, pad: [0; 3] };
        self.go(e, C_KVPREP, "k_state_upd", &su, (NKV * HD / 256, 1, 1), 256);
        let su = StateArgs { x_off: b.v, ldx: (NKV * HD) as u32, cs_off: cs_slot + (NKV * HD * 3) as u32, nch: (NKV * HD) as u32, t: tt, pad: [0; 3] };
        self.go(e, C_KVPREP, "k_state_upd", &su, (NKV * HD / 256, 1, 1), 256);
        // ---- attention (causal, query-tiled)
        let aa = AttnBArgs { t: tt, p0: p0 as u32, ring: 0, cap, k_base: pk, v_base: pv,
                             window: if l.swa { EXT as u32 } else { 0 }, qp_off: b.qp, rp_off: b.rp, sink_off: b.sink, gate_off: b.gate,
                             out_off: b.attn_g, relbias_w: l.rel_bias, ghalf: 0, kpos0, pad: 0 };
        if hatt {
            let ah = AttnBArgs { t: nn_, p0: (p0 + ts) as u32, qp_off: sh(b.qp, NH * HD / 2), rp_off: sh(b.rp, NH * NREL), sink_off: sh(b.sink, NH),
                                 gate_off: sh(b.gate, if qgh { NH * HD / 2 } else { NH * HD }), out_off: sh(b.agh, NH * HD / 2), ghalf: qgh as u32, ..aa };
            self.go(e, C_ATTN, "k_attn_h", &ah, (n.div_ceil(32), NH, 1), 256);
        }
        else if self.tensor { self.go(e, C_ATTN, "k_attn_bt", &aa, (t.div_ceil(32), NH, 1), 256); }
        else { self.go(e, C_ATTN, "k_attn_b", &aa, (t.div_ceil(8), NKV, 1), 128); }
        // ---- output projection + post-conv (+ residual)
        if self.hgemm {
            if !hatt { self.tohalf(e, C_WO, b.attn_g, NH * HD, b.agh, NH * HD, NH * HD, t, None); e.barrier(); }
            self.gemm_h(e, C_WO, &[Self::hseg(&l.wo, sh(b.agh, NH * HD / 2), sh(b.wo_y, E))], n);
        } else {
            self.gemm(e, C_WO, &[Self::gseg(&l.wo, b.attn_g, b.wo_y)], t);
        }
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone(); live.push(cur); live.extend_from_slice(pinned);
        let mid = va.alloc(&live);
        // (tail mode: the first 3 tail rows see the conv state instead of their real predecessors; they are never used)
        let cv = ConvBArgs { y_off: sh(b.wo_y, E), ldy: es, res_off: sh(vo(cur), E), out_off: sh(vo(mid), E), cs_off: cs_slot + (2 * NKV * HD * 3) as u32,
                             convw_off: l.attn_post_conv, has_res: (!block_start) as u32, pad: 0 };
        self.go(e, C_WOPOST, "k_postconv_b", &cv, (E / 256, n, 1), 256);
        let su = StateArgs { x_off: b.wo_y, ldx: es, cs_off: cs_slot + (2 * NKV * HD * 3) as u32, nch: es, t: tt, pad: [0; 3] };
        self.go(e, C_WOPOST, "k_state_upd", &su, (E / 256, 1, 1), 256);
        if block_start { blocks.push(cur); }
        // ---- FFN input
        let mut st: Vec<u32> = blocks.iter().map(|&v| sh(vo(v), E)).collect(); st.push(sh(vo(mid), E));
        let x = if let Some(lw) = ctl {
            let mut jf = self.mix_job_off(&st, l.ffn_res_query as i32, 1, &[b.fin], &[], es);
            if self.hgemm { jf.hcopy[0][0] = b.cath[3] as i32; } else { jf.ncopy[0][0] = b.cat_u[3] as i32; }
            self.mix_t(e, C_MIXF, &[jf], t);
            let ho: Vec<(u32, bool)> = if self.hgemm { vec![(b.fh, rth)] } else { vec![] };
            self.ctrl_b(e, lw, &[(3, b.fin, b.msg[3], b.finc, l.ffn_norm)], t, &ho);
            b.finc
        } else {
            let mut jf = self.mix_job_off(&st, l.ffn_res_query as i32, 1, &[sh(b.fin, E)], &[l.ffn_norm as i32], es);
            if self.hgemm { jf.hout[0] = sh(b.fh, E / 2) as i32; jf.nof32 = rth as u32; }
            self.mix_t(e, C_MIXF, &[jf], n);
            b.fin
        };
        // ---- router + top-k + expert grouping
        let rs = GSeg { w_lo: l.router, w_hi: 0, row_bytes: es * 2, wtype: T_BF16, nrows: NEXP as u32, k: es, in_off: sh(x, E), out_off: b.rlog,
                        ldo: NEXP as u32, wsrc: 1, ..Default::default() };
        if rth { self.gemm_h(e, C_ROUTER, &[Self::hseg_pb(l.router, NEXP as u32, es, sh(b.fh, E / 2), b.rlog)], n); }
        else { self.gemm(e, C_ROUTER, &[rs], n); }
        let ta = TopkBArgs { logits_off: b.rlog, bias_off: l.exp_bias, sel_off: b.sel, t: nn_, wscale: self.wscale, pad: [0; 3] };
        self.go(e, C_TOPK, "k_topk_b", &ta, (n, 1, 1), 32);
        let tile = 32usize;
        if self.hgemm {
            let ra = Route3Args { sel_off: b.sel, t: nn_, perm_off: b.perm, inv_off: b.inv, aw_off: b.aw, ind_off: b.ind, gu_y: (FF / 32) as u32,
                                  dn_y: (E / 64) as u32, tiles_off: b.tl, pad: [0; 2] };
            self.go(e, C_TOPK, "k_moe_route3", &ra, (1, 1, 1), 1024);
        } else {
            let ra = RouteArgs { sel_off: b.sel, t: tt, perm_off: b.perm, inv_off: b.inv, tiles_off: b.tiles, ntiles_off: b.ntiles, tile: tile as u32, pad: 0 };
            self.go(e, C_ROUTER, "k_moe_route", &ra, (1, 1, 1), 1024);
        }
        let max_tiles = (t * TOPK).div_ceil(tile) + NEXP;
        let gu = MoeGemmArgs {
            w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
            wtype: l.gu.ty, nrows: (2 * FF) as u32, k: es, x_off: x, gather: 1, perm_off: b.perm, tiles_off: b.tiles, ntiles_off: b.ntiles,
            out_off: b.act, ldo: FF as u32, situ: 1, pad: 0, beta: self.situ_beta, lbeta: self.situ_lbeta, pad2: [0.0; 2],
        };
        let mg = if self.tensor { "k_moe_gemm_t" } else { "k_moe_gemm" };
        if self.hgemm {
            // gather the FFN input rows of every (token, expert) assignment into contiguous half rows
            // gather the FFN input rows of every (token, expert) assignment into contiguous half rows
            self.tohalf_x(e, C_GATHER, sh(b.fh, E / 2), E, b.xgh, E, E, n * TOPK, Some(b.perm), true);
            e.barrier();
            // tiles in four widths, sized on the GPU by the routing kernel (indirect dispatch); classes run concurrently
            if self.skip.get() & (1 << C_GU) == 0 {
                for (c, (name, _, tg)) in MOE_CLS.iter().enumerate() {
                    let g = MoeGemmHArgs {
                        w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
                        wtype: l.gu.ty, nrows: (2 * FF) as u32, k: es, xh_off: b.xgh, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                        out_off: b.acth, ldo: FF as u32, situ: 1, aw_off: b.aw, beta: self.situ_beta, lbeta: self.situ_lbeta,
                    };
                    e.pso(&self.gpu.pso(name)); e.bytes(&g, 5);
                    e.dispatch_indirect(&self.act.borrow(), (b.ind as usize + c * 4) * 4, *tg);
                }
                e.barrier();
            }
            if self.skip.get() & (1 << C_DOWN) == 0 {
                for (c, (name, _, tg)) in MOE_CLS.iter().enumerate() {
                    let d = MoeGemmHArgs {
                        w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
                        wtype: l.dn.ty, nrows: es, k: FF as u32, xh_off: b.acth, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                        out_off: b.yd, ldo: es, situ: 0, aw_off: 0, beta: 0.0, lbeta: 0.0,
                    };
                    e.pso(&self.gpu.pso(name)); e.bytes(&d, 5);
                    e.dispatch_indirect(&self.act.borrow(), (b.ind as usize + 4 * NTC + c * 4) * 4, *tg);
                }
                e.barrier();
            }
        } else {
        self.go(e, C_GU, mg, &gu, (FF / 32, max_tiles, 1), 128);
        let dn = MoeGemmArgs {
            w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
            wtype: l.dn.ty, nrows: es, k: FF as u32, x_off: b.act, gather: 0, perm_off: b.perm, tiles_off: b.tiles, ntiles_off: b.ntiles,
            out_off: b.yd, ldo: es, situ: 0, pad: 0, beta: 0.0, lbeta: 0.0, pad2: [0.0; 2],
        };
        self.go(e, C_DOWN, mg, &dn, (E / 64, max_tiles, 1), 128);
        }
        let ca = MoeCombArgs { yd_off: b.yd, inv_off: b.inv, sel_off: b.sel, out_off: sh(b.ffn_y, E) };
        if self.hgemm { self.go(e, C_COMB2, "k_moe_comb_h", &ca, (n, E.div_ceil(1024), 1), 256); }
        else { self.go(e, C_DOWN, "k_moe_comb_b", &ca, (E / 256, t, 1), 256); }
        let mut live: Vec<usize> = blocks.clone(); live.push(mid); live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        let cv = ConvBArgs { y_off: sh(b.ffn_y, E), ldy: es, res_off: sh(vo(mid), E), out_off: sh(vo(out), E), cs_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32,
                             convw_off: l.ffn_post_conv, has_res: 1, pad: 0 };
        self.go(e, C_FFNPOST, "k_postconv_b", &cv, (E / 256, n, 1), 256);
        let su = StateArgs { x_off: b.ffn_y, ldx: es, cs_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32, nch: es, t: tt, pad: [0; 3] };
        self.go(e, C_FFNPOST, "k_state_upd", &su, (E / 256, 1, 1), 256);
        out
    }
}
