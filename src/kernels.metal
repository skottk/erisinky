// Inkling 2 Mini (inkling2_loop) fused Metal kernels.
// Buffers: 0 = W (mmapped GGUF, quantized weights), 1 = PB (small f32 params + bf16 matrices),
//          2 = A (f32 activation arena), 3 = KV (K/V cache), 4 = CS (f32 conv state), 5 = args.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// Finite negative sentinel safe under MTLMathMode::Fast (avoids finite-math-only
// miscompilation of -INFINITY / == -INFINITY while exp(NEG_INF_F - m) == 0.0f).
constant float NEG_INF_F   = -1e30f;
constant float NEG_INF_CMP = -1e29f;

#define E      2304
#define NH     24
#define NKV    8
#define HD     128
#define NREL   16
#define EXT    1024
#define NEXP   128
#define TOPK   8
#define FF     1152
#define QK_CLIP 22.627416610717773f
#define REL_CLIP 8.0f

// ---- K/V cache element type.
// Default: fp16, the values the model computed. With KV8 (the opt-in "--kv8" cache, half the size, NOT exact): one signed
// byte per value, v[i] = q[i] * s per 128-value vector. The scale costs no space: s is one of 256 steps
// s(c) = (8 + (c & 7)) * 2^((c >> 3) - 20) (so every q * s is exact in fp16 and f32), and the 8 bits of c are the lowest bits of
// the vector's first 8 bytes (bit 7 - i in byte i) - those 8 values are rounded to the even or odd integers.
// Every reader, including the token's own step, uses the stored values, so batched and single-token paths agree.
#ifdef KV8
typedef int8_t kv_t;
typedef uint kvw_t;            // 4 consecutive cache elements, for copies
inline float kv_scale(uint c) { return float(8u + (c & 7u)) * as_type<float>(((c >> 3) + 107u) << 23); }
// smallest step that represents a vector whose largest magnitude is mx (126 steps: the first 8 values may only reach the even 126)
inline uint kv_code(float mx) {
    if (!(mx > 0.0f)) return 0u;
    const float t = mx / 126.0f;
    int ex = clamp((int) floor(log2(t)), -17, 14);
    int m = clamp((int) ceil(t * exp2((float) -ex) * 8.0f), 8, 16);
    if (m == 16) { m = 8; ex += 1; }
    uint c = (uint) min(((ex + 17) << 3) | (m - 8), 255);
    if (kv_scale(c) * 126.0f < mx && c < 255u) c += 1u;      // log2 / exp2 rounding must never make the largest value clip
    return c;
}
// the scale of the stored vector that starts at p
inline float kv_vscale(device const kv_t * p) {
    device const uint * w = (device const uint *) p;
    const uint hi = ((w[0] & 0x01010101u) * 0x08040201u) >> 24, lo = ((w[1] & 0x01010101u) * 0x08040201u) >> 24;
    return kv_scale(((hi & 15u) << 4) | (lo & 15u));
}
// element i of a vector with scale code c: the stored integer (as float) nearest to x / s
inline float kv_quant(float x, float s, uint c, ushort i) {
    const float y = x / s;
    if (i < 8) { const float b = float((c >> (7 - i)) & 1u); return clamp(2.0f * rint((y - b) * 0.5f) + b, -127.0f, 127.0f); }
    return clamp(rint(y), -127.0f, 127.0f);
}
#else
typedef half kv_t;
typedef half4 kvw_t;
#endif

struct block_q4_K { half d; half dmin; uchar scales[12]; uchar qs[128]; };
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
struct block_q8_0 { half d; char qs[32]; };
struct block_q5_0 { half d; uchar qh[4]; uchar qs[16]; };

inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// sum over the whole threadgroup; every thread gets the result. sh must hold >= 32 floats.
inline float tg_sum(float v, threadgroup float * sh, ushort sg, ushort lane, ushort nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? sh[lane] : 0.0f;
    return simd_sum(r);
}
inline float tg_max(float v, threadgroup float * sh, ushort sg, ushort lane, ushort nsg) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? sh[lane] : NEG_INF_F;
    return simd_max(r);
}

// "last threadgroup done" detection: every TG bumps a device counter after its writes; the TG that observes
// count == total-1 knows all others finished (device-scope seq_cst fences make their writes visible) and resets it.
inline bool last_tg(device atomic_uint * ctr, uint total, ushort tid, threadgroup uint * flag) {
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint prev = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        *flag = (prev == total - 1) ? 1u : 0u;
        if (*flag) atomic_store_explicit(ctr, 0u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *flag != 0;
}

inline void scale_min_k4(int j, device const uchar * q, thread float & sc, thread float & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// ---------------------------------------------------------------------------------------------
// quantized row dot products against x in threadgroup memory. NR rows, row pointers w[r].
// all 32 lanes participate; returns per-lane partial sums in sumf[r] (caller simd_sums).

template <int NR>
inline void dot_q4k(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
    const short ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
    float yl[16], yh[16];
    for (int ib = ix; ib < nb; ib += 4) {
        device const float * y4 = x + ib * 256 + 64 * iq + 8 * ir;
        float4 sumy = 0.0f;
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }
        for (short row = 0; row < NR; ++row) {
            device const block_q4_K * b = (device const block_q4_K *) w[row] + ib;
            device const ushort * sc = (device const ushort *) b->scales + iq;
            device const ushort * q1 = (device const ushort *) b->qs + 16 * iq + 4 * ir;
            device const ushort * q2 = q1 + 32;
            ushort sc16[4];
            thread const uchar * sc8 = (thread const uchar *) sc16;
            sc16[0] = sc[0] & 0x3f3f;
            sc16[1] = sc[2] & 0x3f3f;
            sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
            sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
            float4 acc1 = 0.0f, acc2 = 0.0f;
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += float(b->d) * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                        (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                        (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                        (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         float(b->dmin) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
    }
}

template <int NR>
inline void dot_q6k(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
    const short tid = lane / 2, ix = lane % 2, ip = tid / 8, il = tid % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
    float yl[16];
    for (int i = ix; i < nb; i += 2) {
        device const float * y = x + i * 256 + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l + 0]; yl[4 * l + 1] = y[l + 32]; yl[4 * l + 2] = y[l + 64]; yl[4 * l + 3] = y[l + 96];
        }
        for (short row = 0; row < NR; ++row) {
            device const block_q6_K * b = (device const block_q6_K *) w[row] + i;
            device const uchar * q1 = b->ql + q_offset_l;
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = b->qh + q_offset_h;
            device const char  * sc = b->scales + is;
            float4 sums = 0.0f;
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((int8_t) ((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((int8_t) ((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((int8_t) ((q1[l] >> 4)  | ((qh[l] & 0x30) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((int8_t) ((q2[l] >> 4)  | ((qh[l] & 0xC0) >> 2)) - 32);
            }
            sumf[row] += float(b->d) * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }
}

template <int NR>
inline void dot_bf16(device const uchar * const * w, int K, device const float * x, ushort lane, thread float * sumf) {
    for (int i = lane * 8; i < K; i += 256) {
        const float4 xa = *(device const float4 *) (x + i), xb = *(device const float4 *) (x + i + 4);
        for (short row = 0; row < NR; ++row) {
            const ushort4 a = *(device const ushort4 *) (w[row] + 2 * i);
            const ushort4 b = *(device const ushort4 *) (w[row] + 2 * i + 8);
            sumf[row] += dot(xa, float4(bf16f(a.x), bf16f(a.y), bf16f(a.z), bf16f(a.w))) +
                         dot(xb, float4(bf16f(b.x), bf16f(b.y), bf16f(b.z), bf16f(b.w)));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// embedding lookup (Q4_K row) + RMSNorm(token_embd_norm)

struct EmbedArgs { uint emb_lo, emb_hi, row_bytes, norm_off, out0, out1, tok_index; float eps; };

kernel void k_embed(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                    constant EmbedArgs & a [[buffer(5)]], device const int * toks [[buffer(6)]], uint tok [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    const int t = toks[a.tok_index + tok];
    const ulong off = (ulong(a.emb_hi) << 32 | a.emb_lo) + (ulong) t * a.row_bytes;
    device const block_q4_K * row = (device const block_q4_K *) (W + off);
    float v[9];
    float ss = 0.0f;
    for (int k = 0; k < 9; k++) {
        const int j = tid, pair = j / 64, l = j % 64;
        device const block_q4_K & b = row[k];
        const int sub = pair * 2 + (l >= 32 ? 1 : 0);
        const uchar q = b.qs[pair * 32 + (l & 31)];
        const int qv = l >= 32 ? (q >> 4) : (q & 0xF);
        float sc, m;
        scale_min_k4(sub, b.scales, sc, m);
        v[k] = float(b.d) * sc * qv - float(b.dmin) * m;
        ss += v[k] * v[k];
    }
    ss = tg_sum(ss, sh, sg, lane, 8);
    const float r = rsqrt(ss / E + a.eps);
    device const float * w = (device const float *) (PB + a.norm_off);
    for (int k = 0; k < 9; k++) {
        const int i = tid + k * 256;
        const float o = v[k] * r * w[i];
        A[a.out0 + tok * E + i] = o;
        A[a.out1 + tok * E + i] = o;
    }
}

// ---------------------------------------------------------------------------------------------
// depth attention ("block attention residuals"): out_q = sum_i softmax_i(q . n_i / sqrt(E)) n_i, n_i = rmsnorm(s_i)
// optionally followed by RMSNorm with a weight. q_off < 0: identity (out = raw s_0). One threadgroup per job, 256 threads.

// hcopy / hout: fp16 destinations (normalized copies with row stride hstride halves; (normed) outputs
// with row stride E halves). nof32 bit q: output q is only wanted in fp16.
struct MixJob { uint nS, nQ; uint s_off[6]; int q_off; uint out_off[4]; int norm_off[4]; float eps; uint tstride; int ncopy[4][3]; uint pad_t;
                int hcopy[4][3]; int hout[4]; uint hstride, nof32; };

kernel void k_mix(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant MixJob * jobs [[buffer(5)]],
                  uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    // 768 threads, 3 channels per thread; all register arrays statically indexed (no spills)
    constant MixJob & J = jobs[tgp.x];
    const uint toff = tgp.y * J.tstride;
    threadgroup float part[24][32];
    threadgroup float red[32];
    threadgroup float sh2[24][4];
    const uint nS = J.nS, nQ = J.nQ;
    float s[6][3];
    #pragma unroll
    for (uint i = 0; i < 6; i++) {
        #pragma unroll
        for (int k = 0; k < 3; k++) s[i][k] = i < nS ? A[J.s_off[i] + toff + tid + k * 768] : 0.0f;
    }
    float o[4][3];
    if (J.q_off < 0) {
        #pragma unroll
        for (int k = 0; k < 3; k++) { o[0][k] = s[0][k]; o[1][k] = o[2][k] = o[3][k] = 0; }
    } else {
        device const float * Q = (device const float *) (PB + J.q_off);
        float qv[4][3];
        #pragma unroll
        for (uint q = 0; q < 4; q++) {
            #pragma unroll
            for (int k = 0; k < 3; k++) qv[q][k] = q < nQ ? Q[q * E + tid + k * 768] : 0.0f;
        }
        // v index: i*5 + 0 -> ss_i ; i*5 + 1 + q -> dot_iq
        #pragma unroll
        for (uint i = 0; i < 6; i++) {
            float ss = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            #pragma unroll
            for (int k = 0; k < 3; k++) { ss += s[i][k] * s[i][k]; d0 += s[i][k] * qv[0][k]; d1 += s[i][k] * qv[1][k]; d2 += s[i][k] * qv[2][k]; d3 += s[i][k] * qv[3][k]; }
            ss = simd_sum(ss); d0 = simd_sum(d0); d1 = simd_sum(d1); d2 = simd_sum(d2); d3 = simd_sum(d3);
            if (lane == 0) { part[sg][i * 5 + 0] = ss; part[sg][i * 5 + 1] = d0; part[sg][i * 5 + 2] = d1; part[sg][i * 5 + 3] = d2; part[sg][i * 5 + 4] = d3; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 30) { float t = 0; for (int g = 0; g < 24; g++) t += part[g][tid]; red[tid] = t; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rr[6];
        #pragma unroll
        for (uint i = 0; i < 6; i++) rr[i] = rsqrt(red[i * 5] / E + 1e-6f);
        #pragma unroll
        for (uint q = 0; q < 4; q++) {
            float sc[6], mx = NEG_INF_F;
            #pragma unroll
            for (uint i = 0; i < 6; i++) { sc[i] = i < nS ? rr[i] * red[i * 5 + 1 + q] * (1.0f / 48.0f) : NEG_INF_F; mx = max(mx, sc[i]); }
            float den = 0;
            #pragma unroll
            for (uint i = 0; i < 6; i++) { sc[i] = i < nS ? exp(sc[i] - mx) : 0.0f; den += sc[i]; }
            #pragma unroll
            for (uint i = 0; i < 6; i++) sc[i] = sc[i] / den * rr[i];
            #pragma unroll
            for (int k = 0; k < 3; k++) { float acc = 0; for (uint i = 0; i < 6; i++) acc += sc[i] * s[i][k]; o[q][k] = acc; }
        }
    }
    const uint nOut = J.q_off < 0 ? 1 : nQ;
    // optional RMSNorm per output: one reduction round for all outputs
    float ssq[4];
    #pragma unroll
    for (uint q = 0; q < 4; q++) { float t = 0; for (int k = 0; k < 3; k++) t += o[q][k] * o[q][k]; ssq[q] = simd_sum(t); }
    if (lane == 0) for (int q = 0; q < 4; q++) sh2[sg][q] = ssq[q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint q = 0; q < 4; q++) {
        if (q >= nOut) break;
        float t = 0; for (int g = 0; g < 24; g++) t += sh2[g][q];
        // plain RMSNorm copies (eps 1e-6, no weight) for the loop controllers' inputs
        const float rn = rsqrt(t / E + 1e-6f);
        for (int c = 0; c < 3; c++) {
            if (J.ncopy[q][c] >= 0) for (int k = 0; k < 3; k++) A[J.ncopy[q][c] + toff + tid + k * 768] = o[q][k] * rn;
            if (J.hcopy[q][c] >= 0) {
                device half * dst = (device half *) (A + J.hcopy[q][c]) + (ulong) tgp.y * J.hstride;
                for (int k = 0; k < 3; k++) dst[tid + k * 768] = half(o[q][k] * rn);
            }
        }
        float out_v[3];
        if (J.norm_off[q] >= 0) {
            const float r = rsqrt(t / E + J.eps);
            device const float * w = (device const float *) (PB + J.norm_off[q]);
            #pragma unroll
            for (int k = 0; k < 3; k++) out_v[k] = o[q][k] * r * w[tid + k * 768];
        } else {
            #pragma unroll
            for (int k = 0; k < 3; k++) out_v[k] = o[q][k];
        }
        if (J.hout[q] >= 0) {
            device half * dst = (device half *) (A + J.hout[q]) + (ulong) tgp.y * E;
            #pragma unroll
            for (int k = 0; k < 3; k++) dst[tid + k * 768] = half(out_v[k]);
        }
        if (!((J.nof32 >> q) & 1u)) {
            #pragma unroll
            for (int k = 0; k < 3; k++) A[J.out_off[q] + toff + tid + k * 768] = out_v[k];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// generic segmented matvec: several weight matrices (possibly different quant types and inputs) in one dispatch.
// x is staged into threadgroup memory with optional transforms: * sigmoid(in2), RMSNorm(norm weight).
// epilogue 1: causal short conv (post-conv) with state + optional residual.

struct Seg {
    uint w_lo, w_hi, row_bytes, wtype;
    uint nrows, K, tg_start, in_off;
    int  in2_off, norm_off, ss_off, ss_n;
    uint out_off, epi; int res_off; uint cs_off;
    uint convw_off, pad0, pad1, pad2;
};
struct MvArgs { uint nseg; float eps; uint post, ctr; uint bias_off, sel_off; float wscale; uint pad; Seg seg[12]; };

// top-k over NEXP router logits (sigmoid scores + selection bias), normalized weights * wscale. one simdgroup.
inline void topk_sg(device float * A, device const uchar * PB, uint logits_off, uint bias_off, uint sel_off, float wscale, ushort lane) {
    device const float * lg = A + logits_off;
    device const float * bias = (device const float *) (PB + bias_off);
    float pr[4], sel[4];
    for (int i = 0; i < 4; i++) { const int e = lane * 4 + i; pr[i] = sigm(lg[e]); sel[i] = pr[i] + bias[e]; }
    float wsum = 0, myw = 0; int myid = 0;
    for (int k = 0; k < TOPK; k++) {
        float bv = sel[0]; int bi = 0;
        for (int i = 1; i < 4; i++) if (sel[i] > bv) { bv = sel[i]; bi = i; }
        const float mv = simd_max(bv);
        const int win = simd_min((bv == mv) ? (int) (lane * 4 + bi) : 1 << 20);
        const float pw = simd_shuffle(pr[win % 4], (ushort) (win / 4));
        if ((int) (lane * 4 + bi) == win) sel[bi] = NEG_INF_F;
        if (lane == (ushort) k) { myid = win; myw = pw; }
        wsum += pw;
    }
    if (lane < TOPK) {
        A[sel_off + lane] = (float) myid;
        A[sel_off + TOPK + lane] = myw / (wsum + 1e-20f) * wscale;
    }
}

#define MV_NSG 4
#define MV_NR  2
#define MV_ROWS (MV_NSG * MV_NR)

inline void mv_epilogue(constant Seg & S, uint row, float y, device float * A, device float * CS, device const uchar * PB) {
    if (S.epi == 0) { A[S.out_off + row] = y; return; }
    // post short-conv (static taps) over channel `row`
    device float * h = CS + S.cs_off + row * 3;
    device const float * w = (device const float *) (PB + S.convw_off) + row * 4;
    const float h0 = h[0], h1 = h[1], h2 = h[2];
    const float conv = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * y;
    h[0] = h1; h[1] = h2; h[2] = y;
    float o = (y + conv) * 0.4472135954999579f;  // 1/sqrt(5)
    if (S.res_off >= 0) o += A[S.res_off + row];
    A[S.out_off + row] = o;
}

// inputs are pre-normalized; x is read straight from device memory (L1/L2 cached), no staging/barriers
kernel void k_mv(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                 device float * CS [[buffer(4)]], constant MvArgs & args [[buffer(5)]], device atomic_uint * CTR [[buffer(7)]],
                 uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]],
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint flag;
    uint si = 0;
    while (si + 1 < args.nseg && tg >= args.seg[si + 1].tg_start) si++;
    constant Seg & S = args.seg[si];
    const uint row0 = (tg - S.tg_start) * MV_ROWS + sg * MV_NR;
    if (row0 < S.nrows) {
    const ulong wbase = ulong(S.w_hi) << 32 | S.w_lo;
    device const uchar * base = (S.wtype == 30) ? PB : W;
    device const uchar * wp[MV_NR];
    for (int rr = 0; rr < MV_NR; rr++) wp[rr] = base + wbase + (ulong) min(row0 + rr, S.nrows - 1) * S.row_bytes;
    float sumf[MV_NR] = {0, 0};
    device const float * x = A + S.in_off;
    if (S.wtype == 12) dot_q4k<MV_NR>(wp, S.K / 256, x, lane, sumf);
    else if (S.wtype == 14) dot_q6k<MV_NR>(wp, S.K / 256, x, lane, sumf);
    else dot_bf16<MV_NR>(wp, S.K, x, lane, sumf);
    for (int rr = 0; rr < MV_NR; rr++) {
        const float y = simd_sum(sumf[rr]);
        if (lane == 0 && row0 + rr < S.nrows) mv_epilogue(S, row0 + rr, y, A, CS, PB);
    }
    }
    if (args.post == 1 && last_tg(CTR + args.ctr, ntg, sg * 32 + lane, &flag) && sg == 0)
        topk_sg(A, PB, args.seg[0].out_off, args.bias_off, args.sel_off, args.wscale, lane);
}

// RMSNorm of up to 3 vectors from precomputed partial sums of squares (written by k_ctrl_apply). 1 TG per vector.
struct NormJob { uint in_off, out_off, w_off, ss_off; };
struct NormArgs { uint n, nparts; float eps; uint tstride; NormJob job[4]; };
kernel void k_norm(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant NormArgs & a [[buffer(5)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constant NormJob & J = a.job[tgp.x];
    const uint tok = tgp.y, toff = tok * a.tstride;
    float ss = 0;
    for (uint i = 0; i < a.nparts; i++) ss += A[J.ss_off + tok * 128 + i];
    const float r = rsqrt(ss / E + a.eps);
    device const float * w = (device const float *) (PB + J.w_off);
    for (uint i = tid; i < E; i += 256) A[J.out_off + toff + i] = A[J.in_off + toff + i] * r * w[i];
}

// ---------------------------------------------------------------------------------------------
// loop controller: g = D . [rmsnorm(u); rmsnorm(m)]  (D: 64 x 4608 bf16). grid: nC*16 TGs of 128 threads (1 row / simdgroup)

struct CtrlJob { uint u_off, m_off, d_off, out_off, gain_off, forget_off, res_out, ss_off; int norm_w; uint ctr; };
struct CtrlArgs { uint nC; uint tstride; float eps; uint pad; CtrlJob job[4]; };

kernel void k_ctrl_down(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant CtrlArgs & a [[buffer(5)]],
                        uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[2 * E];
    threadgroup float sh[32];
    const uint tg = tgp.x, tok = tgp.y, toff = tok * a.tstride;
    const uint c = tg / 16;
    constant CtrlJob & J = a.job[c];
    float su = 0, sm = 0;
    for (uint i = tid; i < E; i += 128) { float u = A[J.u_off + toff + i], m = A[J.m_off + toff + i]; su += u * u; sm += m * m; }
    su = tg_sum(su, sh, sg, lane, 4);
    sm = tg_sum(sm, sh, sg, lane, 4);
    const float ru = rsqrt(su / E + 1e-6f), rm = rsqrt(sm / E + 1e-6f);
    for (uint i = tid; i < E; i += 128) { xs[i] = A[J.u_off + toff + i] * ru; xs[E + i] = A[J.m_off + toff + i] * rm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = (tg % 16) * 4 + sg;
    device const uchar * wr = PB + J.d_off + row * (2 * E) * 2;
    float acc = 0;
    for (int i = lane * 8; i < 2 * E; i += 256) {
        const float4 xa = *(threadgroup const float4 *) (xs + i), xb = *(threadgroup const float4 *) (xs + i + 4);
        const ushort4 p = *(device const ushort4 *) (wr + 2 * i), q = *(device const ushort4 *) (wr + 2 * i + 8);
        acc += dot(xa, float4(bf16f(p.x), bf16f(p.y), bf16f(p.z), bf16f(p.w))) + dot(xb, float4(bf16f(q.x), bf16f(q.y), bf16f(q.z), bf16f(q.w)));
    }
    const float y = simd_sum(acc);
    if (lane == 0) A[J.out_off + tok * 64 + row] = y;
}

// out = u*(1 - tanh(F.g)) + tanh(G.g)*m ; G,F: E x 64 bf16. 8 lanes per channel; 256-thread TG = 32 channels.
// writes per-TG partial sum of squares of out to ss_off[tg] (for a downstream RMSNorm).
kernel void k_ctrl_apply(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant CtrlArgs & a [[buffer(5)]],
                         device atomic_uint * CTR [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ntg [[threadgroups_per_grid]],
                         ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    threadgroup uint flag;
    constant CtrlJob & J = a.job[tg.y];
    const uint tok = tg.z, toff = tok * a.tstride;
    const uint ch = tg.x * 32 + sg * 4 + lane / 8;
    const uint p = lane % 8;
    float g[8];
    for (int i = 0; i < 8; i++) g[i] = A[J.out_off + tok * 64 + p * 8 + i];
    ushort4 ga = *(device const ushort4 *) (PB + J.gain_off + ch * 128 + p * 16);
    ushort4 gb = *(device const ushort4 *) (PB + J.gain_off + ch * 128 + p * 16 + 8);
    ushort4 fa = *(device const ushort4 *) (PB + J.forget_off + ch * 128 + p * 16);
    ushort4 fb = *(device const ushort4 *) (PB + J.forget_off + ch * 128 + p * 16 + 8);
    float dg = bf16f(ga.x) * g[0] + bf16f(ga.y) * g[1] + bf16f(ga.z) * g[2] + bf16f(ga.w) * g[3] +
               bf16f(gb.x) * g[4] + bf16f(gb.y) * g[5] + bf16f(gb.z) * g[6] + bf16f(gb.w) * g[7];
    float df = bf16f(fa.x) * g[0] + bf16f(fa.y) * g[1] + bf16f(fa.z) * g[2] + bf16f(fa.w) * g[3] +
               bf16f(fb.x) * g[4] + bf16f(fb.y) * g[5] + bf16f(fb.z) * g[6] + bf16f(fb.w) * g[7];
    dg += simd_shuffle_xor(dg, 1); dg += simd_shuffle_xor(dg, 2); dg += simd_shuffle_xor(dg, 4);
    df += simd_shuffle_xor(df, 1); df += simd_shuffle_xor(df, 2); df += simd_shuffle_xor(df, 4);
    float o = 0;
    if (p == 0) {
        const float u = A[J.u_off + toff + ch], m = A[J.m_off + toff + ch];
        o = u - tanh(df) * u + tanh(dg) * m;
        A[J.res_out + toff + ch] = o;
    }
    const float ss = tg_sum(o * o, sh, sg, lane, 8);
    if (tid == 0) A[J.ss_off + tok * 128 + tg.x] = ss;
    if (J.norm_w >= 0 && last_tg(CTR + J.ctr, ntg.x, tid, &flag)) {
        float t = 0;
        for (uint i = 0; i < ntg.x; i++) t += A[J.ss_off + tok * 128 + i];
        const float r = rsqrt(t / E + a.eps);
        device const float * w = (device const float *) (PB + J.norm_w);
        for (uint i = tid; i < E; i += 256) A[J.res_out + toff + i] *= r * w[i];
    }
}

// ---------------------------------------------------------------------------------------------
// decode attention for one token, one kv head per TG column, keys split across TG rows (flash-decoding).
// fuses: K/V short conv (static + dynamic taps) with state update, K clip, V RMSNorm, Q clip, R clip,
// log-scale tau, relative-position bias, per-head sink, partial softmax.

struct AttnArgs {
    uint pos, nkeys, first_pos, ring, cap, k_base, v_base, cs_off;
    uint q_off, k_off, v_off, r_off, sink_off; int kdyn_off, vdyn_off; uint part_off;
    uint kconv_w, vconv_w, qclip_w, relclip_w, relbias_w; float tau; uint kps, nsplit;
    uint ctr, gate_off, out_off, pad;
    // sp0: index of this dispatch's first split among the layer's nsplit partials; noown: keys only (another dispatch computes
    // and stores the token's own K/V); row (g, idx) of the bound K/V lives at g * gstr + idx * istr; idx = pos - idx0 if !ring
    uint sp0, noown, gstr, istr, idx0;
    float lastbias;      // dev (key eviction experiments): added to the score of the dispatch's last key
};
#define KVROW(g, idx) ((g) * a.gstr + (idx) * a.istr)
#define KVIDX(p) (a.ring ? ((p) % a.cap) : ((p) - a.idx0))

#define KPS_MAX 256

kernel void k_attn(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                   device float * CS [[buffer(4)]], constant AttnArgs & a [[buffer(5)]], device atomic_uint * CTR [[buffer(7)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    // one TG = (key split, kv head g) serving q heads 3g..3g+2; 128 threads; keys processed in chunks of 32
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float kc[HD], vc[HD];
    threadgroup float red[4][8];
    const uint sp = tgp.x, g = tgp.y;
    const uint j0 = sp * a.kps, j1 = min(a.nkeys, j0 + a.kps);
    const bool owns = (j1 == a.nkeys) && a.noown == 0;
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    // ---- phase 1: raw values + partial norms
    float q[3];
    for (int h = 0; h < 3; h++) { const uint H = g * 3 + h; q[h] = A[a.q_off + H * HD + tid] * qclip[H * HD + tid]; }
    float kx = 0, vx = 0;
    if (owns) {
        const uint c = g * HD + tid;
        const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            const uint ch = which * (NKV * HD) + c;
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_w : a.vconv_w)) + c * 4;
            device float * hh = CS + a.cs_off + ch * 3;
            const float x = A[(which == 0 ? a.k_off : a.v_off) + c], h0 = hh[0], h1 = hh[1], h2 = hh[2];
            float acc = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * x;
            const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
            if (dyn >= 0) { device const float * t = A + dyn + g * 4; acc += t[0] * x + t[1] * h2 + t[2] * h1 + t[3] * h0; }
            hh[0] = h1; hh[1] = h2; hh[2] = x;
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[a.r_off + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        const float s5 = simd_max(fabs(kx)), s6 = simd_max(fabs(vx));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    // logit = (q.k)/sqrt(HD) + r.B[d] - sink  (bias + sink are unscaled, as llama.cpp folds them into the mask)
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * a.tau * 0.08838834764831845f;
        ((threadgroup float *) qs4[h])[tid] = q[h] * f;
    }
    if (tid < 48) rs[tid / 16][tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * a.tau;
    if (owns) {
        const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
        const float k = kx * kf, v = vx * vf;
        const uint idx = KVIDX(a.pos);
#ifdef KV8
        const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
        const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
        const float sk = kv_scale(ck), sv = kv_scale(cv);
        const float qk = kv_quant(k, sk, ck, tid), qv = kv_quant(v, sv, cv, tid);
        kc[tid] = qk * sk; vc[tid] = qv * sv;
        KV[a.k_base + KVROW(g, idx) + tid] = (kv_t) (int) qk;
        KV[a.v_base + KVROW(g, idx) + tid] = (kv_t) (int) qv;
#else
        kc[tid] = k; vc[tid] = v;
        KV[a.k_base + KVROW(g, idx) + tid] = half(k);
        KV[a.v_base + KVROW(g, idx) + tid] = half(v);
#endif
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float * relb = (device const float *) (PB + a.relbias_w);
    // ---- barrier-free per-simdgroup online softmax over 8-key blocks (4 lanes per key for the scores)
    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {NEG_INF_F, NEG_INF_F, NEG_INF_F}, l[3] = {0, 0, 0};
    float4 o[3] = {0, 0, 0};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);
    for (uint jb = j0 + sg * 8; jb < j1; jb += 32) {
        const uint j = jb + kk;
        const bool valid = j < j1;
        const uint p = a.first_pos + min(j, j1 - 1);
        const uint idx = KVIDX(p);
        float acc[3] = {0, 0, 0};
        if (p == a.pos) {
            for (int c = 0; c < 8; c++) {
                const float4 kf = *(threadgroup const float4 *) (&kc[pp * 32 + c * 4]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
        } else {
#ifdef KV8
            device const kv_t * krow = KV + a.k_base + KVROW(g, idx);
            device const packed_char4 * kr = (device const packed_char4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(char4(kr[c]));
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
            const float ks = kv_vscale(krow);
            for (int h = 0; h < 3; h++) acc[h] *= ks;
#else
            device const half4 * kr = (device const half4 *) (KV + a.k_base + KVROW(g, idx) + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
#endif
        }
        const uint d = a.pos - p;
        if (d < EXT) {
            const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4);
            for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
        }
        if (a.lastbias != 0.0f && j == a.nkeys - 1) for (int h = 0; h < 3; h++) acc[h] += a.lastbias * 0.25f;
        float pr[3], sc_[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? t - sink[h] : NEG_INF_F;
            const float mx = max(m[h], simd_max(t));
            sc_[h] = exp(m[h] - mx);
            pr[h] = exp(t - mx);
            l[h] = l[h] * sc_[h] + simd_sum(pr[h]) * 0.25f;
            m[h] = mx;
            o[h] *= sc_[h];
        }
#ifdef KV8
        // each stored value vector's scale goes into its key's probability (the token's own vector is already scaled)
        { const float vsc = (p == a.pos) ? 1.0f : kv_vscale(KV + a.v_base + KVROW(g, idx)); for (int h = 0; h < 3; h++) pr[h] *= vsc; }
#endif
        // V: lane owns dims [4*lane, 4*lane+4); 8 keys of this block
        for (ushort q = 0; q < 8; q++) {
            const uint jq = jb + q;
            if (jq >= j1) break;
            const uint pq = a.first_pos + jq;
            const uint iq = KVIDX(pq);
            const float4 v = (pq == a.pos) ? *(threadgroup const float4 *) (&vc[lane * 4])
#ifdef KV8
                                           : float4(char4(*(device const packed_char4 *) (KV + a.v_base + KVROW(g, iq) + lane * 4)));
#else
                                           : float4(*(device const half4 *) (KV + a.v_base + KVROW(g, iq) + lane * 4));
#endif
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q * 4) * v;
        }
    }
    // ---- merge the 4 simdgroups' partial softmax states
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int h = 0; h < 3; h++) {
        float M = NEG_INF_F;
        for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
        float L = 0, O = 0;
        for (int s2 = 0; s2 < 4; s2++) {
            const float w = ms_[s2][h] <= NEG_INF_CMP ? 0.0f : exp(ms_[s2][h] - M);
            L += w * ls_[s2][h];
            O += w * ((threadgroup const float *) os_[s2][h])[tid];
        }
        device float * P = A + a.part_off + ((g * 3 + h) * a.nsplit + a.sp0 + sp) * (HD + 2);
        if (tid == 0) { P[0] = M; P[1] = L; }
        P[2 + tid] = O;
    }
    // the last split of this kv group combines all splits (+ the zero-logit sink) and applies the sigmoid gate
    threadgroup uint flag;
    if (a.pad == 0 && last_tg(CTR + a.ctr + g, a.nsplit, tid, &flag)) {
        for (int h = 0; h < 3; h++) {
            const uint H = g * 3 + h;
            device const float * P = A + a.part_off + H * a.nsplit * (HD + 2);
            float M = 0.0f;
            for (uint s2 = 0; s2 < a.nsplit; s2++) M = max(M, P[s2 * (HD + 2)]);
            float L = exp(-M), O = 0;
            for (uint s2 = 0; s2 < a.nsplit; s2++) {
                const float w = exp(P[s2 * (HD + 2)] - M);
                L += w * P[s2 * (HD + 2) + 1];
                O += w * P[s2 * (HD + 2) + 2 + tid];
            }
            const uint i = H * HD + tid;
            A[a.out_off + i] = O / L * sigm(A[a.gate_off + i]);
        }
    }
}

// combine split partials (+ zero-logit sink) and apply the sigmoid output gate. grid: NH TGs x 128 threads
struct CombineArgs { uint part_off, nsplit, gate_off, out_off; };
kernel void k_attn_combine(device float * A [[buffer(2)]], constant CombineArgs & a [[buffer(5)]],
                           uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    device const float * P = A + a.part_off + h * a.nsplit * (HD + 2);
    float M = 0.0f;  // sink logit
    for (uint s = 0; s < a.nsplit; s++) M = max(M, P[s * (HD + 2)]);
    float L = exp(-M), o = 0;
    for (uint s = 0; s < a.nsplit; s++) {
        const float w = exp(P[s * (HD + 2)] - M);
        L += w * P[s * (HD + 2) + 1];
        o += w * P[s * (HD + 2) + 2 + tid];
    }
    const uint i = h * HD + tid;
    A[a.out_off + i] = o / L * sigm(A[a.gate_off + i]);
}

// ---------------------------------------------------------------------------------------------
// MoE: top-k (sigmoid + selection bias), gate_up (Q4_K, fused [gate; up] rows), SiTU activation.
// grid: (FF / (NSG*2), TOPK) TGs of NSG simdgroups; each simdgroup does 2 gate rows + 2 up rows.

struct MoeArgs {
    uint logits_off, bias_off, sel_off, x_off;
    int  norm_off, ss_off, ss_n; float eps;
    uint gu_lo, gu_hi, gu_expert_bytes, gu_row_bytes;
    uint act_off; float wscale, beta, lbeta;
    uint dn_lo, dn_hi, dn_expert_bytes, dn_row_bytes;
    uint dn_type, out_off; int res_off; uint cs_off;
    uint convw_off, pad0, pad1, pad2;
};

#define GU_NSG 4

kernel void k_topk(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], constant MoeArgs & a [[buffer(5)]],
                   ushort lane [[thread_index_in_simdgroup]]) {
    device const float * lg = A + a.logits_off;
    device const float * bias = (device const float *) (PB + a.bias_off);
    float pr[4], sel[4];
    for (int i = 0; i < 4; i++) { const int e = lane * 4 + i; pr[i] = sigm(lg[e]); sel[i] = pr[i] + bias[e]; }
    float wsum = 0, myw = 0; int myid = 0;
    for (int k = 0; k < TOPK; k++) {
        float bv = sel[0]; int bi = 0;
        for (int i = 1; i < 4; i++) if (sel[i] > bv) { bv = sel[i]; bi = i; }
        const float mv = simd_max(bv);
        const int win = simd_min((bv == mv) ? (int) (lane * 4 + bi) : 1 << 20);
        const float pw = simd_shuffle(pr[win % 4], (ushort) (win / 4));
        if ((int) (lane * 4 + bi) == win) sel[bi] = NEG_INF_F;
        if (lane == (ushort) k) { myid = win; myw = pw; }
        wsum += pw;
    }
    if (lane < TOPK) {
        A[a.sel_off + lane] = (float) myid;
        A[a.sel_off + TOPK + lane] = myw / (wsum + 1e-20f) * a.wscale;
    }
}

// grid: (FF / GU_NSG, TOPK); each simdgroup computes gate row i and up row i of one expert -> SiTU
kernel void k_moe_gu(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                     constant MoeArgs & a [[buffer(5)]],
                     uint2 tgp [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint k = tgp.y;
    const int e = (int) A[a.sel_off + k];
    const uint i = tgp.x * GU_NSG + sg;
    const ulong base = (ulong(a.gu_hi) << 32 | a.gu_lo) + (ulong) e * a.gu_expert_bytes;
    device const uchar * wp[2] = { W + base + (ulong) i * a.gu_row_bytes, W + base + (ulong) (FF + i) * a.gu_row_bytes };
    float sumf[2] = {0, 0};
    dot_q4k<2>(wp, E / 256, A + a.x_off, lane, sumf);
    const float gt = simd_sum(sumf[0]), up = simd_sum(sumf[1]);
    if (lane == 0) {
        const float act = a.beta * tanh(gt / a.beta) * sigm(gt) * (a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up);
        A[a.act_off + k * FF + i] = act;
    }
}

// down projection for all TOPK experts at once + weighted sum + post-conv + residual. Q8_0 or Q5_0 rows (K = FF).
// lane handles (expert, block) items f = lane + 32*s (TOPK*36 = 288 = 9*32); packed (unaligned) loads.
#define DN_NSG 2
#define DN_NR  2

template <int NR>
inline void down_rows(device const uchar * eb, uint row0, uint row_bytes, uint type, int b, thread const float4 * xv, float wk, thread float * acc) {
    for (int rr = 0; rr < NR; rr++) {
        device const uchar * rowp = eb + (ulong) (row0 + rr) * row_bytes;
        float sum = 0;
        if (type == 8) {
            device const block_q8_0 * bq = (device const block_q8_0 *) rowp + b;
            device const packed_char4 * q = (device const packed_char4 *) bq->qs;
            for (int i = 0; i < 8; i++) sum += dot(xv[i], float4(char4(q[i])));
            sum *= float(bq->d);
        } else {
            device const block_q5_0 * bq = (device const block_q5_0 *) rowp + b;
            const uint qh = as_type<uint>(uchar4(*(device const packed_uchar4 *) bq->qh));
            device const packed_uchar4 * qs4 = (device const packed_uchar4 *) bq->qs;
            for (int i = 0; i < 4; i++) {
                const uchar4 q = uchar4(qs4[i]);
                const uint hl = qh >> (4 * i), hh = qh >> (4 * i + 16);
                const float4 lo = float4(int4(uint4(q & 0xF) | ((uint4(hl, hl >> 1, hl >> 2, hl >> 3) & 1) << 4))) - 16.0f;
                const float4 hi = float4(int4(uint4(q >> 4) | ((uint4(hh, hh >> 1, hh >> 2, hh >> 3) & 1) << 4))) - 16.0f;
                sum += dot(xv[i], lo) + dot(xv[4 + i], hi);
            }
            sum *= float(bq->d);
        }
        acc[rr] += wk * sum;
    }
}

kernel void k_moe_down(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                       device float * CS [[buffer(4)]], constant MoeArgs & a [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * DN_NSG + sg) * DN_NR;
    const ulong base = ulong(a.dn_hi) << 32 | a.dn_lo;
    float acc[DN_NR] = {0, 0};
    for (int s = 0; s < 9; s++) {
        const int f = lane + 32 * s, k = f / 36, b = f % 36;
        device const float4 * x4 = (device const float4 *) (A + a.act_off + k * FF + b * 32);
        float4 xv[8];
        for (int i = 0; i < 8; i++) xv[i] = x4[i];
        const int e = (int) A[a.sel_off + k];
        down_rows<DN_NR>(W + base + (ulong) e * a.dn_expert_bytes, row0, a.dn_row_bytes, a.dn_type, b, xv, A[a.sel_off + TOPK + k], acc);
    }
    for (int rr = 0; rr < DN_NR; rr++) {
        const float y = simd_sum(acc[rr]);
        if (lane == 0) {
            const uint row = row0 + rr;
            device float * h = CS + a.cs_off + row * 3;
            device const float * w = (device const float *) (PB + a.convw_off) + row * 4;
            const float h0 = h[0], h1 = h[1], h2 = h[2];
            const float conv = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * y;
            h[0] = h1; h[1] = h2; h[2] = y;
            A[a.out_off + row] = A[a.res_off + row] + (y + conv) * 0.4472135954999579f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// argmax over logits (two stages). stage 1: grid G TGs x 1024 threads -> partial (val, idx). stage 2: 1 TG.
struct ArgmaxArgs { uint logits_off, n, part_off, nparts, out_index; uint pad0, pad1, pad2; };

kernel void k_argmax1(device float * A [[buffer(2)]], constant ArgmaxArgs & a [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32]; threadgroup int bi[32];
    float best = NEG_INF_F; int idx = 0;
    for (uint i = tg * 1024 + tid; i < a.n; i += ntg * 1024) { const float v = A[a.logits_off + i]; if (v > best) { best = v; idx = i; } }
    const float m = simd_max(best);
    int c = best == m ? idx : 0x7fffffff; c = simd_min(c);
    if (lane == 0) { bv[sg] = m; bi[sg] = c; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = bv[lane]; int ii = bi[lane];
        const float m2 = simd_max(v);
        int c2 = v == m2 ? ii : 0x7fffffff; c2 = simd_min(c2);
        if (lane == 0) { A[a.part_off + 2 * tg] = m2; A[a.part_off + 2 * tg + 1] = as_type<float>(c2); }
    }
}

kernel void k_argmax2(device float * A [[buffer(2)]], constant ArgmaxArgs & a [[buffer(5)]], device int * toks [[buffer(6)]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    float best = NEG_INF_F; int idx = 0x7fffffff;
    for (uint i = lane; i < a.nparts; i += 32) {
        const float v = A[a.part_off + 2 * i]; const int ii = as_type<int>(A[a.part_off + 2 * i + 1]);
        if (v > best || (v == best && ii < idx)) { best = v; idx = ii; }
    }
    const float m = simd_max(best);
    int c = best == m ? idx : 0x7fffffff; c = simd_min(c);
    if (lane == 0) toks[a.out_index] = c;
}
