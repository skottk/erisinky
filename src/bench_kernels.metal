// Matvec variants for micro-benchmarking (Q4_K, K = 2304). Included after kernels.metal.
struct BenchArgs { uint w_lo, w_hi, row_bytes, nrows, K, x_off, out_off, pad; };

// B: x read directly from device memory (no staging, no barrier)
template <int NR>
inline void dot_q4k_dev(device const uchar * const * w, int nb, device const float * x, ushort lane, thread float * sumf) {
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
template <int NR, int NSG>
kernel void kb_dev(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant BenchArgs & a [[buffer(5)]],
                   uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * NSG + sg) * NR;
    const ulong base = ulong(a.w_hi) << 32 | a.w_lo;
    device const uchar * wp[NR];
    for (int r = 0; r < NR; r++) wp[r] = W + base + (ulong) (row0 + r) * a.row_bytes;
    float sumf[NR] = {0};
    dot_q4k_dev<NR>(wp, a.K / 256, A + a.x_off, lane, sumf);
    for (int r = 0; r < NR; r++) { const float y = simd_sum(sumf[r]); if (lane == 0) A[a.out_off + row0 + r] = y; }
}

// C: one superblock per simdgroup step (32 lanes x 4 bytes of qs = 8 weights/lane), NR rows, x from device.
// lane L: qs bytes [4L, 4L+4): pair j = L/8, l0 = 4*(L%8): low nibbles -> sub-block 2j, high -> 2j+1
template <int NR, int NSG>
kernel void kb_blk(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant BenchArgs & a [[buffer(5)]],
                   uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * NSG + sg) * NR;
    const ulong base = ulong(a.w_hi) << 32 | a.w_lo;
    const int j = lane / 8, l0 = 4 * (lane % 8);
    const int nb = a.K / 256;
    device const float * x = A + a.x_off;
    float sumf[NR] = {0};
    for (int ib = 0; ib < nb; ib++) {
        const float4 xl = *(device const float4 *) (x + ib * 256 + 64 * j + l0);
        const float4 xh = *(device const float4 *) (x + ib * 256 + 64 * j + 32 + l0);
        const float sl = xl.x + xl.y + xl.z + xl.w, sh = xh.x + xh.y + xh.z + xh.w;
        for (int r = 0; r < NR; r++) {
            device const block_q4_K * b = (device const block_q4_K *) (W + base + (ulong) (row0 + r) * a.row_bytes) + ib;
            const uint q = *(device const uint *) (b->qs + 4 * lane);
            const float4 ql = float4(q & 0xF, (q >> 8) & 0xF, (q >> 16) & 0xF, (q >> 24) & 0xF);
            const float4 qh = float4((q >> 4) & 0xF, (q >> 12) & 0xF, (q >> 20) & 0xF, (q >> 28) & 0xF);
            float sc0, m0, sc1, m1;
            scale_min_k4(2 * j, b->scales, sc0, m0);
            scale_min_k4(2 * j + 1, b->scales, sc1, m1);
            const float d = b->d, dm = b->dmin;
            sumf[r] += d * (sc0 * dot(ql, xl) + sc1 * dot(qh, xh)) - dm * (m0 * sl + m1 * sh);
        }
    }
    for (int r = 0; r < NR; r++) { const float y = simd_sum(sumf[r]); if (lane == 0) A[a.out_off + row0 + r] = y; }
}

template [[host_name("kb_dev_2_2")]] kernel void kb_dev<2, 2>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_dev_2_4")]] kernel void kb_dev<2, 4>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_dev_4_2")]] kernel void kb_dev<4, 2>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_dev_4_4")]] kernel void kb_dev<4, 4>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_dev_8_2")]] kernel void kb_dev<8, 2>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_dev_1_4")]] kernel void kb_dev<1, 4>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_blk_2_4")]] kernel void kb_blk<2, 4>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_blk_4_4")]] kernel void kb_blk<4, 4>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_blk_8_2")]] kernel void kb_blk<8, 2>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);
template [[host_name("kb_blk_4_8")]] kernel void kb_blk<4, 8>(device const uchar *, device float *, constant BenchArgs &, uint, ushort, ushort);

// ---- MoE down variants. Each computes out[row] = sum_k w_k * dot(Wd[e_k][row], act_k) for E rows.
template <int NR>
inline void down_item_b(device const uchar * eb, uint row0, uint row_bytes, uint type, int b, thread const float4 * xv, float wk, thread float * acc) {
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
            const uint qh = uint(as_type<uint>(*(device const packed_uchar4 *) bq->qh));
            device const packed_uchar4 * qs4 = (device const packed_uchar4 *) bq->qs;
            for (int i = 0; i < 4; i++) {
                const uchar4 q = uchar4(qs4[i]);
                const uint hl = qh >> (4 * i), hh = qh >> (4 * i + 16);
                const float4 lo = float4(int4(uint4(q & 0xF) | (uint4(hl, hl >> 1, hl >> 2, hl >> 3) & 1) << 4)) - 16.0f;
                const float4 hi = float4(int4(uint4(q >> 4) | (uint4(hh, hh >> 1, hh >> 2, hh >> 3) & 1) << 4)) - 16.0f;
                sum += dot(xv[i], lo) + dot(xv[4 + i], hi);
            }
            sum *= float(bq->d);
        }
        acc[rr] += wk * sum;
    }
}

// V1: like k_moe_down but packed (unaligned) loads, NR rows per simdgroup, act from device
template <int NR, int NSG>
kernel void kd_dev(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeArgs & a [[buffer(5)]],
                   uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * NSG + sg) * NR;
    const ulong base = ulong(a.dn_hi) << 32 | a.dn_lo;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0;
    for (int s = 0; s < 9; s++) {
        const int f = lane + 32 * s, k = f / 36, b = f % 36;
        device const float4 * x4 = (device const float4 *) (A + a.act_off + k * FF + b * 32);
        float4 xv[8];
        for (int i = 0; i < 8; i++) xv[i] = x4[i];
        const int e = (int) A[a.sel_off + k];
        down_item_b<NR>(W + base + (ulong) e * a.dn_expert_bytes, row0, a.dn_row_bytes, a.dn_type, b, xv, A[a.sel_off + TOPK + k], acc);
    }
    for (int r = 0; r < NR; r++) { const float y = simd_sum(acc[r]); if (lane == 0) A[a.out_off + row0 + r] = y; }
}

// V2: act staged in threadgroup memory 4 experts at a time (f32, 18 KB)
template <int NR, int NSG>
kernel void kd_tg(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeArgs & a [[buffer(5)]],
                  uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 xs[4 * FF / 4];
    const uint row0 = (tg * NSG + sg) * NR;
    const ulong base = ulong(a.dn_hi) << 32 | a.dn_lo;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0;
    for (int half_ = 0; half_ < 2; half_++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 4 * FF / 4; i += NSG * 32) xs[i] = ((device const float4 *) (A + a.act_off + half_ * 4 * FF))[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 4 experts x 36 blocks = 144 items; 16 lanes per row-pair? keep 32 lanes: items f = lane + 32 s, s < 4.5 -> use 5 iters
        for (int s = 0; s < 5; s++) {
            const int f = lane + 32 * s;
            if (f >= 144) break;
            const int kl = f / 36, b = f % 36, k = half_ * 4 + kl;
            float4 xv[8];
            for (int i = 0; i < 8; i++) xv[i] = xs[(kl * FF + b * 32) / 4 + i];
            const int e = (int) A[a.sel_off + k];
            down_item_b<NR>(W + base + (ulong) e * a.dn_expert_bytes, row0, a.dn_row_bytes, a.dn_type, b, xv, A[a.sel_off + TOPK + k], acc);
        }
    }
    for (int r = 0; r < NR; r++) { const float y = simd_sum(acc[r]); if (lane == 0) A[a.out_off + row0 + r] = y; }
}

template [[host_name("kd_dev_4_4")]] kernel void kd_dev<4, 4>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_dev_2_4")]] kernel void kd_dev<2, 4>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_dev_8_2")]] kernel void kd_dev<8, 2>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_dev_8_4")]] kernel void kd_dev<8, 4>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_tg_4_8")]] kernel void kd_tg<4, 8>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort, ushort);
template [[host_name("kd_tg_2_8")]] kernel void kd_tg<2, 8>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort, ushort);
template [[host_name("kd_tg_4_16")]] kernel void kd_tg<4, 16>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort, ushort);
template [[host_name("kd_tg_8_8")]] kernel void kd_tg<8, 8>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort, ushort);
template [[host_name("kd_dev_1_4")]] kernel void kd_dev<1, 4>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_dev_2_2")]] kernel void kd_dev<2, 2>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_dev_1_8")]] kernel void kd_dev<1, 8>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_dev_2_8")]] kernel void kd_dev<2, 8>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);
template [[host_name("kd_dev_1_2")]] kernel void kd_dev<1, 2>(device const uchar *, device float *, constant MoeArgs &, uint, ushort, ushort);

// touch a byte range (prefetch into the system-level cache); result discarded
struct TouchArgs { uint lo, hi, n16, pad; };
kernel void k_touch(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant TouchArgs & a [[buffer(5)]],
                    uint gid [[thread_position_in_grid]], uint nth [[threads_per_grid]]) {
    device const uint4 * p = (device const uint4 *) (W + (ulong(a.hi) << 32 | a.lo));
    uint4 acc = 0;
    for (uint i = gid; i < a.n16; i += nth) acc ^= p[i];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) A[0] = 1.0f;
}

// ---- dense GEMM tiling variants (fp16 tensor cores). Y[t][r] = sum_k X[t][k] W[r][k]; weights dequantized to TG memory.
template <int BM, int BK>
inline void stage_w(device const uchar * wb, constant GSeg & S, uint r0, uint k0, ushort tid, thread half * wv) {
    constexpr int WPT = BM * BK / 128;
    for (int c = 0; c < WPT / 16; c++) {
        const uint idx = tid * WPT + c * 16, row = idx / BK, kk = idx % BK;
        deq16(wb + (ulong) min(r0 + row, S.nrows - 1) * S.row_bytes, S.wtype, k0 + kk, wv + c * 16);
    }
}
// staged activations (half in TG memory)
template <int BM, int BN, int BK>
kernel void kg_s(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & T [[buffer(9)]],
                 uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[BM * BK];
    threadgroup half sb[BN * BK];
    const uint r0 = tgp.x * BM, t0 = tgp.y * BN, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    constexpr int WPT = BM * BK / 128, XPT = BN * BK / 128;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(BK, BM));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(BK, BN));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(BN, BM, BK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint k0 = 0; k0 < K; k0 += BK) {
        half wv[WPT];
        stage_w<BM, BK>(wb, S, r0, k0, tid, wv);
        float xv[XPT];
        for (int i = 0; i < XPT; i++) { const uint idx = tid * XPT + i, tt = idx / BK, kk = idx % BK; xv[i] = A[S.in_off + min(t0 + tt, T - 1) * K + k0 + kk]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < WPT; i++) sa[tid * WPT + i] = wv[i];
        for (int i = 0; i < XPT; i++) sb[tid * XPT + i] = half(xv[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(tB, tA, cT);
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(BM, (int) (S.nrows - r0)), min(BN, (int) (T - t0))), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
// activations consumed straight from device memory (f32), relaxed precision (as llama.cpp's dense path)
template <int BM, int BN, int BK>
kernel void kg_d(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & T [[buffer(9)]],
                 uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[BM * BK];
    const uint r0 = tgp.x * BM, t0 = tgp.y * BN, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    constexpr int WPT = BM * BK / 128;
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(BK, BM));
    auto tX = tensor(A + S.in_off + (ulong) t0 * K, dextents<int32_t, 2>((int) K, min(BN, (int) (T - t0))), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(BN, BM, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA), float>();
    for (uint k0 = 0; k0 < K; k0 += BK) {
        half wv[WPT];
        stage_w<BM, BK>(wb, S, r0, k0, tid, wv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < WPT; i++) sa[tid * WPT + i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sx = tX.slice(k0, 0);
        mm.run(sx, tA, cT);
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(BM, (int) (S.nrows - r0)), min(BN, (int) (T - t0))), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
#define KGS(BM, BN, BK) template [[host_name("kg_" #BM "_" #BN "_" #BK "_0")]] kernel void kg_s<BM, BN, BK>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
#define KGD(BM, BN, BK) template [[host_name("kg_" #BM "_" #BN "_" #BK "_1")]] kernel void kg_d<BM, BN, BK>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
KGS(64, 32, 32) KGS(64, 64, 32) KGS(64, 32, 64) KGS(64, 64, 64) KGS(128, 32, 32) KGS(128, 64, 32)
KGD(64, 32, 32) KGD(64, 64, 32) KGD(64, 64, 64) KGD(128, 64, 32) KGD(128, 64, 64) KGD(128, 128, 32)
// double-buffered: dequantize tile k+1 (shader ALUs) while the tensor units multiply tile k; one barrier per step
template <int BM, int BN, int BK>
kernel void kg_db(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & T [[buffer(9)]],
                  uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[2][BM * BK];
    const uint r0 = tgp.x * BM, t0 = tgp.y * BN, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    constexpr int WPT = BM * BK / 128;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, BM));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(BK, BM));
    auto tX = tensor(A + S.in_off + (ulong) t0 * K, dextents<int32_t, 2>((int) K, min(BN, (int) (T - t0))), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(BN, BM, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    half wv[WPT];
    stage_w<BM, BK>(wb, S, r0, 0, tid, wv);
    for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[0] + tid * WPT))[i] = ((thread half4 *) wv)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) stage_w<BM, BK>(wb, S, r0, k0 + BK, tid, wv);
        auto sx = tX.slice(k0, 0);
        if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        if (more) for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + tid * WPT))[i] = ((thread half4 *) wv)[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(BM, (int) (S.nrows - r0)), min(BN, (int) (T - t0))), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
#define KGB(BM, BN, BK) template [[host_name("kg_" #BM "_" #BN "_" #BK "_2")]] kernel void kg_db<BM, BN, BK>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
KGB(64, 64, 32) KGB(64, 32, 32) KGB(128, 64, 32) KGB(64, 128, 32) KGB(64, 64, 64)
// activations pre-converted to half in device memory (halves the L2 activation stream); double-buffered weights
template <int BM, int BN, int BK>
kernel void kg_h(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & T [[buffer(9)]],
                 uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[2][BM * BK];
    const uint r0 = tgp.x * BM, t0 = tgp.y * BN, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    constexpr int WPT = BM * BK / 128;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, BM));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(BK, BM));
    device half * xh = (device half *) (A + S.in_off) + (ulong) t0 * K;
    auto tX = tensor(xh, dextents<int32_t, 2>((int) K, min(BN, (int) (T - t0))), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(BN, BM, BK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    half wv[WPT];
    stage_w<BM, BK>(wb, S, r0, 0, tid, wv);
    for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[0] + tid * WPT))[i] = ((thread half4 *) wv)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) stage_w<BM, BK>(wb, S, r0, k0 + BK, tid, wv);
        auto sx = tX.slice(k0, 0);
        if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        if (more) for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + tid * WPT))[i] = ((thread half4 *) wv)[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(BM, (int) (S.nrows - r0)), min(BN, (int) (T - t0))), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
#define KGH(BM, BN, BK) template [[host_name("kg_" #BM "_" #BN "_" #BK "_3")]] kernel void kg_h<BM, BN, BK>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
KGH(64, 64, 32) KGH(64, 128, 32) KGH(64, 64, 64) KGH(128, 128, 32) KGH(128, 64, 32) KGH(64, 128, 64)
// bottleneck isolation: MODE 0 = no dequant, X half from device; 1 = no dequant, X staged in TG; 2 = dequant, X staged in TG (from half)
template <int MODE>
kernel void kg_iso(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & T [[buffer(9)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[2][64 * 32];
    threadgroup half sb[2][64 * 32];
    const uint r0 = tgp.x * 64, t0 = tgp.y * 64, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    for (uint i = tid; i < 64 * 32; i += 128) { sa[0][i] = half(0.01f * (i % 13)); sa[1][i] = sa[0][i]; }
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(32, 64));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(32, 64));
    auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb[0], dextents<int32_t, 2>(32, 64));
    auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb[1], dextents<int32_t, 2>(32, 64));
    device half * xh = (device half *) (A + S.in_off) + (ulong) t0 * K;
    auto tX = tensor(xh, dextents<int32_t, 2>((int) K, min(64, (int) (T - t0))), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(64, 64, 32, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB0), decltype(tA0), float>();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += 32) {
        if (MODE == 0) { auto sx = tX.slice(k0, 0); if (buf) mm.run(sx, tA1, cT); else mm.run(sx, tA0, cT); }
        else {
            // stage X (and maybe W) for this step into buffer buf
            const uint tt = tid / 2, kk = (tid % 2) * 16;
            device const half4 * src = (device const half4 *) (xh + (ulong) min(tt, T - t0 - 1) * K + k0 + kk);
            threadgroup half4 * dst = (threadgroup half4 *) (sb[buf] + tt * 32 + kk);
            for (int i = 0; i < 4; i++) dst[i] = src[i];
            if (MODE == 2) {
                half wv[16];
                deq16(wb + (ulong) min(r0 + tt, S.nrows - 1) * S.row_bytes, S.wtype, k0 + kk, wv);
                for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf] + tt * 32 + kk))[i] = ((thread half4 *) wv)[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (buf) mm.run(tB1, tA1, cT); else mm.run(tB0, tA0, cT);
        }
        buf ^= 1;
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(64, (int) (S.nrows - r0)), min(64, (int) (T - t0))), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
template [[host_name("kg_iso0_3")]] kernel void kg_iso<0>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
template [[host_name("kg_iso1_3")]] kernel void kg_iso<1>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
template [[host_name("kg_iso2_3")]] kernel void kg_iso<2>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
// big tiles: BM x BN with NSG simdgroups, half X from device, double-buffered dequant
template <int BM, int BN, int NSG>
kernel void kg_big(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & T [[buffer(9)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int BK = 32, NT = NSG * 32, WPT = BM * BK / NT < 16 ? 16 : BM * BK / NT;
    threadgroup half sa[2][BM * BK];
    const uint r0 = tgp.x * BM, t0 = tgp.y * BN, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, BM));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(BK, BM));
    device half * xh = (device half *) (A + S.in_off) + (ulong) t0 * K;
    auto tX = tensor(xh, dextents<int32_t, 2>((int) K, min(BN, (int) (T - t0))), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(BN, BM, BK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    half wv[WPT];
    const bool act = tid * WPT < BM * BK;
    auto stage = [&](uint k0) {
        if (act) for (int c = 0; c < WPT / 16; c++) {
            const uint idx = tid * WPT + c * 16, row = idx / BK, kk = idx % BK;
            deq16(wb + (ulong) min(r0 + row, S.nrows - 1) * S.row_bytes, S.wtype, k0 + kk, wv + c * 16);
        }
    };
    stage(0);
    if (act) for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[0] + tid * WPT))[i] = ((thread half4 *) wv)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) stage(k0 + BK);
        auto sx = tX.slice(k0, 0);
        if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        if (more && act) for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + tid * WPT))[i] = ((thread half4 *) wv)[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(BM, (int) (S.nrows - r0)), min(BN, (int) (T - t0))), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
#define KGBIG(BM, BN, NSG) template [[host_name("kgb_" #BM "_" #BN "_" #NSG)]] kernel void kg_big<BM, BN, NSG>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
KGBIG(128, 128, 8) KGBIG(128, 64, 8) KGBIG(64, 128, 8) KGBIG(64, 64, 4) KGBIG(64, 256, 16) KGBIG(64, 512, 32) KGBIG(128, 128, 16) KGBIG(128, 256, 32) KGBIG(256, 64, 16) KGBIG(64, 32, 2) KGBIG(64, 32, 4) KGBIG(64, 16, 2) KGBIG(64, 64, 8) KGBIG(64, 128, 16) KGBIG(64, 64, 2) KGBIG(64, 128, 4)

// same with swapped grid axes (token tiles on x, row tiles on y)
// big tiles: BM x BN with NSG simdgroups, half X from device, double-buffered dequant
template <int BM, int BN, int NSG>
kernel void kg_bigs(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & T [[buffer(9)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int BK = 32, NT = NSG * 32, WPT = BM * BK / NT < 16 ? 16 : BM * BK / NT;
    threadgroup half sa[2][BM * BK];
    const uint r0 = tgp.y * BM, t0 = tgp.x * BN, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, BM));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(BK, BM));
    device half * xh = (device half *) (A + S.in_off) + (ulong) t0 * K;
    auto tX = tensor(xh, dextents<int32_t, 2>((int) K, min(BN, (int) (T - t0))), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(BN, BM, BK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    half wv[WPT];
    const bool act = tid * WPT < BM * BK;
    auto stage = [&](uint k0) {
        if (act) for (int c = 0; c < WPT / 16; c++) {
            const uint idx = tid * WPT + c * 16, row = idx / BK, kk = idx % BK;
            deq16(wb + (ulong) min(r0 + row, S.nrows - 1) * S.row_bytes, S.wtype, k0 + kk, wv + c * 16);
        }
    };
    stage(0);
    if (act) for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[0] + tid * WPT))[i] = ((thread half4 *) wv)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) stage(k0 + BK);
        auto sx = tX.slice(k0, 0);
        if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        if (more && act) for (int i = 0; i < WPT / 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + tid * WPT))[i] = ((thread half4 *) wv)[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(min(BM, (int) (S.nrows - r0)), min(BN, (int) (T - t0))), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
#define KGBIGS(BM, BN, NSG) template [[host_name("kgs_" #BM "_" #BN "_" #NSG)]] kernel void kg_bigs<BM, BN, NSG>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort);
KGBIGS(64, 128, 8) KGBIGS(64, 64, 4) KGBIGS(64, 32, 4) KGBIGS(64, 96, 6) KGBIGS(64, 112, 7) KGBIGS(64, 192, 12) KGBIGS(64, 256, 16)

// per-simdgroup tensor ops sharing one staged weight tile: simdgroup s owns tokens [16 s, 16 s + 16) of the tile.
// T (buffer 9) low 16 bits = tokens, high bits = number of ACTIVE simdgroups per tile (the others skip the matmul): tests whether
// simdgroups idling at the barrier cost anything.
template <int NSG>
kernel void kg_sg(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & TT [[buffer(9)]],
                  uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int BM = 64, BK = 32, BN = 16 * NSG;
    threadgroup half sa[2][BM * BK];
    const uint T = TT & 0xFFFF, nact = TT >> 16;
    const uint r0 = tgp.y * BM, t0 = tgp.x * BN + sg * 16, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, BM));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(BK, BM));
    device half * xh = (device half *) (A + S.in_off) + (ulong) t0 * K;
    auto tX = tensor(xh, dextents<int32_t, 2>((int) K, 16), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, BM, BK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroup> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool on = sg < nact;
    half wv[16];
    const bool act = tid < 128;
    const uint idx = tid * 16, row = idx / BK, kk = idx % BK;
    device const uchar * wr = wb + (ulong) min(r0 + row, S.nrows - 1) * S.row_bytes;
    if (act) { deq16(wr, S.wtype, kk, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + idx))[i] = ((thread half4 *) wv)[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more && act) deq16(wr, S.wtype, k0 + BK + kk, wv);
        if (on) {
            auto sx = tX.slice(k0, 0);
            if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        }
        if (more && act) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + idx))[i] = ((thread half4 *) wv)[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    if (on) {
        auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(BM, 16), array<int, 2>({1, (int) S.ldo}));
        cT.store(tC);
    }
}
template [[host_name("kgsg_8")]] kernel void kg_sg<8>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgsg_16")]] kernel void kg_sg<16>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgsg_4")]] kernel void kg_sg<4>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);

// same as kg_sg<8> with a short-latency Q4_K staging: the block header (d, dmin, 12 scale bytes) is loaded once per super-block
// and kept in registers, so a k-step is one 16-byte load + unpack with no dependent loads in front of it.
template <int NSG, int MODE>
kernel void kg_sgf(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & TT [[buffer(9)]],
                   uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int BM = 64, BK = 32, BN = 16 * NSG;
    threadgroup half sa[2][BM * BK];
    const uint T = TT & 0xFFFF, nact = TT >> 16;
    const uint r0 = tgp.y * BM, t0 = tgp.x * BN + sg * 16, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, BM));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(BK, BM));
    device half * xh = (device half *) (A + S.in_off) + (ulong) t0 * K;
    auto tX = tensor(xh, dextents<int32_t, 2>((int) K, 16), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, BM, BK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroup> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool on = sg < nact;
    const bool act = tid < 128;
    const uint idx = tid * 16, row = idx / BK, hp = (idx % BK) / 16;
    device const uchar * wr = wb + (ulong) min(r0 + row, S.nrows - 1) * S.row_bytes;
    float hd = 0, hm = 0; uint s0 = 0, s1 = 0, s2 = 0;
    half4 w0, w1, w2, w3;
    auto stage = [&](uint k0) {
        device const block_q4_K * b = (device const block_q4_K *) wr + k0 / 256;
        const uint j = (k0 % 256) / 32;
        if (j == 0) { hd = b->d; hm = b->dmin; const uint3 sc3 = *(device const packed_uint3 *) b->scales; s0 = sc3.x; s1 = sc3.y; s2 = sc3.z; }
        // 6-bit scale / min of sub-block j from the 12 header bytes held in registers
        uint sc, mn;
        if (j < 4) { sc = (s0 >> (8 * j)) & 63; mn = (s1 >> (8 * j)) & 63; }
        else { const uint hi = (s2 >> (8 * (j - 4))) & 0xFF; sc = (hi & 0xF) | (((s0 >> (8 * (j - 4) + 6)) & 3) << 4); mn = (hi >> 4) | (((s1 >> (8 * (j - 4) + 6)) & 3) << 4); }
        const float df = hd * float(sc), mf = hm * float(mn);
        const uint4 v = (*(device const uint4 *) (b->qs + (j / 2) * 32 + hp * 16) >> ((j & 1) * 4)) & 0x0F0F0F0Fu;
        if (MODE == 0) {
            w0 = half4(fma(float4(as_type<uchar4>(v.x)), df, -mf)); w1 = half4(fma(float4(as_type<uchar4>(v.y)), df, -mf));
            w2 = half4(fma(float4(as_type<uchar4>(v.z)), df, -mf)); w3 = half4(fma(float4(as_type<uchar4>(v.w)), df, -mf));
        } else {
            w0 = half4(fma(float4(nib4(v.x)), df, -mf)); w1 = half4(fma(float4(nib4(v.y)), df, -mf));
            w2 = half4(fma(float4(nib4(v.z)), df, -mf)); w3 = half4(fma(float4(nib4(v.w)), df, -mf));
        }
    };
    auto put = [&](uint buf) { threadgroup half4 * d = (threadgroup half4 *) (sa[buf] + idx); d[0] = w0; d[1] = w1; d[2] = w2; d[3] = w3; };
    if (act) { stage(0); put(0); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more && act) stage(k0 + BK);
        if (on) {
            auto sx = tX.slice(k0, 0);
            if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        }
        if (more && act) put(buf ^ 1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    if (on) {
        auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + r0, dextents<int32_t, 2>(BM, 16), array<int, 2>({1, (int) S.ldo}));
        cT.store(tC);
    }
}
template [[host_name("kgsgf0_8")]] kernel void kg_sgf<8, 0>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgsgf1_8")]] kernel void kg_sgf<8, 1>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgsgf0_4")]] kernel void kg_sgf<4, 0>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgsgf0_16")]] kernel void kg_sgf<16, 0>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);

// precision-mode probe for the tile op: RELAX = matmul2d relaxed_precision, HDST = half destination (fp16 accumulators),
// BF = bfloat weights. Same structure as kg_sgf<8, 0>.
template <int RELAX, int HDST>
kernel void kg_mode(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & TT [[buffer(9)]],
                    uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int BM = 64, BK = 32, NSG = 8, BN = 16 * NSG;
    threadgroup half sa[2][BM * BK];
    const uint r0 = tgp.y * BM, t0 = tgp.x * BN + sg * 16, K = S.K;
    device const uchar * wb = W + (ulong(S.w_hi) << 32 | S.w_lo);
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(BK, BM));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(BK, BM));
    device half * xh = (device half *) (A + S.in_off) + (ulong) t0 * K;
    auto tX = tensor(xh, dextents<int32_t, 2>((int) K, 16), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, BM, BK, false, true, RELAX != 0, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroup> mm;
#define DT typename metal::conditional<HDST != 0, half, float>::type
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), DT>();
    const bool act = tid < 128;
    const uint idx = tid * 16, row = idx / BK, hp = (idx % BK) / 16;
    device const uchar * wr = wb + (ulong) min(r0 + row, S.nrows - 1) * S.row_bytes;
    WHead wh; half4 wv[4];
    if (act) { wstage(wr, S.wtype, 0, hp, wh, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + idx))[i] = wv[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more && act) wstage(wr, S.wtype, k0 + BK, hp, wh, wv);
        auto sx = tX.slice(k0, 0);
        if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        if (more && act) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + idx))[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    for (short i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            const auto ix = cT.get_multidimensional_index(i);
            A[S.out_off + (ulong) (t0 + ix[1]) * S.ldo + r0 + ix[0]] = float(cT[i]);
        }
    }
}
template [[host_name("kgmode_0_0")]] kernel void kg_mode<0, 0>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgmode_1_0")]] kernel void kg_mode<1, 0>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgmode_0_1")]] kernel void kg_mode<0, 1>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);
template [[host_name("kgmode_1_1")]] kernel void kg_mode<1, 1>(device const uchar *, device float *, constant GSeg &, constant uint &, uint2, ushort, ushort);

// hardware ceiling probe: fp16 x fp16 straight from device memory, no staging, no barriers. KS = k per op call (KS = 0: one
// call over the whole K). Left = 16 tokens of X per simdgroup, right = 64 "rows" taken from the same half array.
template <int NSG, int KS>
kernel void kg_peak(device float * A [[buffer(2)]], constant GSeg & S [[buffer(5)]], constant uint & TT [[buffer(9)]],
                    uint2 tgp [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint K = S.K, t0 = (tgp.x * NSG + sg) * 16, r0 = (tgp.y * 64) % (TT - 64);
    device half * xh = (device half *) (A + S.in_off);
    auto tX = tensor(xh + (ulong) t0 * K, dextents<int32_t, 2>((int) K, 16), array<int, 2>({1, (int) K}));
    auto tW = tensor(xh + (ulong) r0 * K, dextents<int32_t, 2>((int) K, 64), array<int, 2>({1, (int) K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(16, 64, KS == 0 ? 2304 : KS, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroup> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
    if (KS == 0) mm.run(tX, tW, cT);
    else for (uint k0 = 0; k0 < K; k0 += KS) { auto sx = tX.slice(k0, 0); auto sw = tW.slice(k0, 0); mm.run(sx, sw, cT); }
    auto tC = tensor(A + S.out_off + (ulong) t0 * S.ldo + tgp.y * 64, dextents<int32_t, 2>(64, 16), array<int, 2>({1, (int) S.ldo}));
    cT.store(tC);
}
template [[host_name("kgpeak_8_0")]] kernel void kg_peak<8, 0>(device float *, constant GSeg &, constant uint &, uint2, ushort);
template [[host_name("kgpeak_8_32")]] kernel void kg_peak<8, 32>(device float *, constant GSeg &, constant uint &, uint2, ushort);
template [[host_name("kgpeak_8_128")]] kernel void kg_peak<8, 128>(device float *, constant GSeg &, constant uint &, uint2, ushort);
template [[host_name("kgpeak_8_576")]] kernel void kg_peak<8, 576>(device float *, constant GSeg &, constant uint &, uint2, ushort);
template [[host_name("kgpeak_1_0")]] kernel void kg_peak<1, 0>(device float *, constant GSeg &, constant uint &, uint2, ushort);
template [[host_name("kgpeak_4_0")]] kernel void kg_peak<4, 0>(device float *, constant GSeg &, constant uint &, uint2, ushort);

