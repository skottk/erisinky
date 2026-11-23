mod common;

use common::gguf_builder::{GgufBuilder, KvVal, TensorSpec};
use inkling::engine::{
    BLOCK, E, EXT, FF, HD, LOOP_END, LOOP_START, NEXP, NH, NKV, NREL, N_PASS, N_PHYS, TOPK,
};
use inkling::gguf::{
    try_type_block, Gguf, WeightAuditReport, WeightOffender, T_BF16, T_F16, T_F32, T_Q4_0, T_Q4_K,
    T_Q5_0, T_Q5_K, T_Q6_K, T_Q8_0,
};
use std::io::{ErrorKind, Write};

fn assert_invalid_data<T: std::fmt::Debug>(res: std::io::Result<T>, ctx: &str) {
    match res {
        Err(e) => assert_eq!(
            e.kind(),
            ErrorKind::InvalidData,
            "{ctx}: expected InvalidData, got {e:?}"
        ),
        Ok(v) => panic!("{ctx}: expected Err(InvalidData), got Ok({v:?})"),
    }
}

fn valid_inkling_model_builder(n_vocab: u64, n_vocab_pad: u64) -> GgufBuilder {
    let swa_pat: Vec<bool> = (0..N_PHYS).map(|i| i % BLOCK != BLOCK - 1).collect();
    let mut b = GgufBuilder::new()
        .kv("general.architecture", KvVal::Str("inkling2_loop".into()))
        .kv("inkling2_loop.embedding_length", KvVal::U32(E as u32))
        .kv("inkling2_loop.block_count", KvVal::U32(N_PHYS as u32))
        .kv("inkling2_loop.expert_count", KvVal::U32(NEXP as u32))
        .kv("inkling2_loop.expert_used_count", KvVal::U32(TOPK as u32))
        .kv("inkling2_loop.loop.start", KvVal::U32(LOOP_START as u32))
        .kv("inkling2_loop.loop.end", KvVal::U32(LOOP_END as u32))
        .kv("inkling2_loop.loop.count", KvVal::U32(N_PASS as u32))
        .kv("inkling2_loop.attn_res.block_size", KvVal::U32(BLOCK as u32))
        .kv("inkling2_loop.attention.sliding_window", KvVal::U32(EXT as u32))
        .kv("inkling2_loop.attention.relative_extent", KvVal::U32(EXT as u32))
        .kv(
            "inkling2_loop.attention.sliding_window_pattern",
            KvVal::ArrBool(swa_pat.clone()),
        )
        .kv("inkling2_loop.vocab_size_unpadded", KvVal::U32(n_vocab as u32))
        .kv("inkling2_loop.attention.layer_norm_rms_epsilon", KvVal::F32(1e-6))
        .kv("inkling2_loop.attention.log_scale_alpha", KvVal::F32(0.1))
        .kv("inkling2_loop.attention.log_scale_floor", KvVal::F32(1024.0))
        .kv("inkling2_loop.activation.situ_beta", KvVal::F32(3.0))
        .kv("inkling2_loop.activation.situ_linear_beta", KvVal::F32(0.0))
        .kv("inkling2_loop.expert_weights_scale", KvVal::F32(2.5))
        .kv("inkling2_loop.final_logit_softcapping", KvVal::F32(30.0));

    let mut add = |name: String, ne: Vec<u64>, ty: u32| {
        b.tensors.push(TensorSpec {
            name,
            ne,
            ty,
            offset_override: Some(0),
            data: Vec::new(),
        });
    };

    for (i, &swa) in swa_pat.iter().enumerate() {
        let n = |s: &str| format!("blk.{i}.{s}");
        if swa {
            add(n("attn_norm.weight"), vec![E as u64], T_F32);
            if i > 0 {
                add(n("attn_res_query.weight"), vec![E as u64], T_F32);
            }
        } else {
            add(n("attn_q_inp_norm.weight"), vec![E as u64], T_F32);
            add(n("attn_k_inp_norm.weight"), vec![E as u64], T_F32);
            add(n("attn_v_inp_norm.weight"), vec![E as u64], T_F32);
            add(n("attn_res_query.weight"), vec![E as u64, 3], T_F32);
            add(n("attn_k_conv_proj.weight"), vec![E as u64, (NKV * 4) as u64], T_Q4_K);
            add(n("attn_v_conv_proj.weight"), vec![E as u64, (NKV * 4) as u64], T_Q6_K);
        }
        add(n("attn_q.weight"), vec![E as u64, (NH * HD) as u64], T_Q4_K);
        add(n("attn_gate.weight"), vec![E as u64, (NH * HD) as u64], T_Q4_K);
        add(n("attn_k.weight"), vec![E as u64, (NKV * HD) as u64], T_Q4_K);
        add(
            n("attn_v.weight"),
            vec![E as u64, (NKV * HD) as u64],
            if swa { T_Q4_K } else { T_Q6_K },
        );
        add(n("attn_output.weight"), vec![(NH * HD) as u64, E as u64], T_Q4_K);
        add(n("attn_rel.weight"), vec![E as u64, (NH * NREL) as u64], T_Q4_K);
        add(n("attn_sink_proj.weight"), vec![E as u64, NH as u64], T_Q4_K);
        add(n("ffn_res_query.weight"), vec![E as u64], T_F32);
        add(n("ffn_norm.weight"), vec![E as u64], T_F32);
        add(n("attn_q_clip.weight"), vec![HD as u64, NH as u64], T_F32);
        add(n("attn_rel_clip.weight"), vec![NREL as u64, NH as u64], T_F32);
        add(n("attn_rel_bias.weight"), vec![NREL as u64, EXT as u64], T_F32);
        add(n("attn_k_conv.weight"), vec![4, (NKV * HD) as u64], T_F32);
        add(n("attn_v_conv.weight"), vec![4, (NKV * HD) as u64], T_F32);
        add(n("attn_post_conv.weight"), vec![4, E as u64], T_F32);
        add(n("ffn_post_conv.weight"), vec![4, E as u64], T_F32);
        add(n("ffn_gate_inp.weight"), vec![E as u64, NEXP as u64], T_F32);
        add(n("exp_probs_b.bias"), vec![NEXP as u64], T_F32);
        add(
            n("ffn_gate_up_exps.weight"),
            vec![E as u64, (2 * FF) as u64, NEXP as u64],
            T_Q4_K,
        );
        add(
            n("ffn_down_exps.weight"),
            vec![FF as u64, E as u64, NEXP as u64],
            if i % 2 == 0 { T_Q8_0 } else { T_Q5_0 },
        );
    }

    for i in LOOP_START..LOOP_END {
        let n = |s: &str| format!("blk.{i}.{s}");
        add(n("loop_msg_query.weight"), vec![E as u64, 4], T_F32);
        for c in ["q", "k", "v", "ffn"] {
            add(n(&format!("loop_{c}_ctrl_down.weight")), vec![(2 * E) as u64, 64], T_F32);
            add(n(&format!("loop_{c}_ctrl_gain.weight")), vec![64, E as u64], T_F32);
            add(n(&format!("loop_{c}_ctrl_forget.weight")), vec![64, E as u64], T_F32);
        }
    }

    add("token_embd_norm.weight".into(), vec![E as u64], T_F32);
    add("output_norm.weight".into(), vec![E as u64], T_F32);
    add("output_res_query.weight".into(), vec![E as u64], T_F32);
    add("token_embd.weight".into(), vec![E as u64, n_vocab_pad], T_Q4_K);
    add("output.weight".into(), vec![E as u64, n_vocab_pad], T_Q6_K);
    b
}

fn open_sparse_gguf(builder: &GgufBuilder) -> Gguf {
    let header = builder.build();
    let mut max_nbytes = 0u64;
    for t in &builder.tensors {
        if let Ok((bs, be)) = try_type_block(t.ty) {
            let n: u64 = t.ne.iter().copied().product();
            let nb = (n / be as u64) * bs as u64;
            max_nbytes = max_nbytes.max(nb);
        }
    }
    let mut f = tempfile::NamedTempFile::new().expect("create sparse temp gguf");
    f.write_all(&header).expect("write header");
    f.flush().expect("flush header");
    let total_len = (header.len() as u64) + max_nbytes + 4096;
    f.as_file().set_len(total_len).expect("set sparse len");
    Gguf::open(f.path().to_str().unwrap()).expect("open sparse gguf")
}

fn set_kv(b: &mut GgufBuilder, key: &str, val: KvVal) {
    for (k, v) in &mut b.kvs {
        if k == key {
            *v = val;
            return;
        }
    }
    panic!("missing kv {key}");
}

// -----------------------------------------------------------------------------
// Part 1: Load-time hyperparameter floors (fast-math safety)
// -----------------------------------------------------------------------------

#[test]
fn validate_inkling_model_rejects_unsafe_rms_epsilon_under_fast_math() {
    let bad_values: &[(&str, KvVal)] = &[
        ("eps == 0.0 (f32)", KvVal::F32(0.0)),
        ("eps == 0.0 (f64)", KvVal::F64(0.0)),
        ("eps == -1e-6 (f32)", KvVal::F32(-1e-6)),
        ("eps == 1e-20 (f32)", KvVal::F32(1e-20)),
        ("eps == 1e-13 (f64)", KvVal::F64(1e-13)),
        ("eps == 9.9e-13 (f32)", KvVal::F32(9.9e-13)),
        ("eps == f32::MIN_POSITIVE", KvVal::F32(f32::MIN_POSITIVE)),
        ("eps == NaN", KvVal::F32(f32::NAN)),
        ("eps == +Inf", KvVal::F32(f32::INFINITY)),
        ("eps == -Inf", KvVal::F32(f32::NEG_INFINITY)),
        ("eps == f64::MAX (overflows f32)", KvVal::F64(f64::MAX)),
    ];

    for (label, val) in bad_values {
        let mut b = valid_inkling_model_builder(1000, 1024);
        set_kv(&mut b, "inkling2_loop.attention.layer_norm_rms_epsilon", val.clone());
        let g = open_sparse_gguf(&b);
        assert_invalid_data(g.validate_inkling_model(), label);
    }
}

#[test]
fn validate_inkling_model_accepts_safe_rms_epsilon_floor() {
    for (label, val) in [
        ("eps == 1e-12 (f32)", KvVal::F32(1e-12)),
        ("eps == 1e-12 (f64)", KvVal::F64(1e-12)),
        ("eps == 1e-6 (f32)", KvVal::F32(1e-6)),
    ] {
        let mut b = valid_inkling_model_builder(1000, 1024);
        set_kv(&mut b, "inkling2_loop.attention.layer_norm_rms_epsilon", val);
        let g = open_sparse_gguf(&b);
        g.validate_inkling_model()
            .unwrap_or_else(|e| panic!("{label} should be accepted, got {e:?}"));
    }
}

#[test]
fn validate_inkling_model_rejects_negative_and_non_finite_softcap() {
    let bad_values: &[(&str, KvVal)] = &[
        ("softcap == -1.0 (f32)", KvVal::F32(-1.0)),
        ("softcap == -1e-6 (f32)", KvVal::F32(-1e-6)),
        ("softcap == -30.0 (f64)", KvVal::F64(-30.0)),
        ("softcap == NaN", KvVal::F32(f32::NAN)),
        ("softcap == +Inf", KvVal::F32(f32::INFINITY)),
        ("softcap == -Inf", KvVal::F32(f32::NEG_INFINITY)),
        ("softcap == f64::MAX (overflows f32)", KvVal::F64(f64::MAX)),
    ];

    for (label, val) in bad_values {
        let mut b = valid_inkling_model_builder(1000, 1024);
        set_kv(&mut b, "inkling2_loop.final_logit_softcapping", val.clone());
        let g = open_sparse_gguf(&b);
        assert_invalid_data(g.validate_inkling_model(), label);
    }
}

#[test]
fn validate_inkling_model_accepts_non_negative_softcap() {
    for (label, val) in [
        ("softcap == 0.0 (disabled)", KvVal::F32(0.0)),
        ("softcap == 30.0", KvVal::F32(30.0)),
    ] {
        let mut b = valid_inkling_model_builder(1000, 1024);
        set_kv(&mut b, "inkling2_loop.final_logit_softcapping", val);
        let g = open_sparse_gguf(&b);
        g.validate_inkling_model()
            .unwrap_or_else(|e| panic!("{label} should be accepted, got {e:?}"));
    }
}

// -----------------------------------------------------------------------------
// Part 2: Parallel GGUF weight/scale finiteness audit
// -----------------------------------------------------------------------------

const F16_FINITE_SAMPLES: &[u16] = &[
    0x0000, // +0.0
    0x8000, // -0.0
    0x0001, // smallest +subnormal
    0x8001, // smallest -subnormal
    0x03FF, // largest +subnormal
    0x83FF, // largest -subnormal
    0x3C00, // +1.0
    0xBC00, // -1.0
    0x7BFF, // +65504.0 (max finite fp16)
    0xFBFF, // -65504.0 (min finite fp16)
];

const F16_NON_FINITE_SAMPLES: &[(&str, u16)] = &[
    ("qNaN", 0x7E00),
    ("sNaN", 0x7C01),
    ("-NaN", 0xFE00),
    ("+Inf", 0x7C00),
    ("-Inf", 0xFC00),
];

const BF16_FINITE_SAMPLES: &[u16] = &[
    0x0000, // +0.0
    0x8000, // -0.0
    0x0001, // smallest +subnormal
    0x8001, // smallest -subnormal
    0x007F, // largest +subnormal
    0x3F80, // +1.0
    0xBF80, // -1.0
    0x7F7F, // max finite bf16
    0xFF7F, // min finite bf16
];

const BF16_NON_FINITE_SAMPLES: &[(&str, u16)] = &[
    ("qNaN", 0x7FC0),
    ("sNaN", 0x7F81),
    ("-NaN", 0xFFC0),
    ("+Inf", 0x7F80),
    ("-Inf", 0xFF80),
];

const F32_NON_FINITE_SAMPLES: &[(&str, u32)] = &[
    ("qNaN", 0x7FC0_0000),
    ("sNaN", 0x7F80_0001),
    ("-NaN", 0xFFC0_0000),
    ("+Inf", 0x7F80_0000),
    ("-Inf", 0xFF80_0000),
];

fn make_quant_blocks(ty: u32, n_blocks: usize, fill_payload: u8) -> Vec<u8> {
    let (bs, _be) = try_type_block(ty).unwrap();
    let mut data = vec![fill_payload; bs * n_blocks];
    for b in 0..n_blocks {
        let base = b * bs;
        let d = F16_FINITE_SAMPLES[b % F16_FINITE_SAMPLES.len()].to_le_bytes();
        let dmin = F16_FINITE_SAMPLES[(b + 3) % F16_FINITE_SAMPLES.len()].to_le_bytes();
        match ty {
            T_Q4_0 | T_Q5_0 | T_Q8_0 => {
                data[base..base + 2].copy_from_slice(&d);
            }
            T_Q4_K | T_Q5_K => {
                data[base..base + 2].copy_from_slice(&d);
                data[base + 2..base + 4].copy_from_slice(&dmin);
            }
            T_Q6_K => {
                data[base + 208..base + 210].copy_from_slice(&d);
            }
            _ => panic!("not a quantized block type: {ty}"),
        }
    }
    data
}

#[test]
fn audit_weights_finiteness_passes_on_all_clean_types_and_edge_finite_values() {
    let f32_vals: [f32; 10] = [
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007F_FFFF),
        1.0,
        -1.0,
        65504.0,
        f32::MAX,
        f32::MIN,
    ];
    let mut f32_bytes = Vec::with_capacity(f32_vals.len() * 4);
    for v in f32_vals {
        f32_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let mut f16_bytes = Vec::with_capacity(F16_FINITE_SAMPLES.len() * 2);
    for &u in F16_FINITE_SAMPLES {
        f16_bytes.extend_from_slice(&u.to_le_bytes());
    }

    let mut bf16_bytes = Vec::with_capacity(BF16_FINITE_SAMPLES.len() * 2);
    for &u in BF16_FINITE_SAMPLES {
        bf16_bytes.extend_from_slice(&u.to_le_bytes());
    }

    // Fill quantized payloads with 0xFF (which looks like fp16 NaN if wrongly scanned!)
    // to prove only the true scale header offsets are inspected.
    let q4_0_bytes = make_quant_blocks(T_Q4_0, 4, 0xFF);
    let q5_0_bytes = make_quant_blocks(T_Q5_0, 4, 0xFF);
    let q8_0_bytes = make_quant_blocks(T_Q8_0, 4, 0xFF);
    let q4_k_bytes = make_quant_blocks(T_Q4_K, 4, 0xFF);
    let q5_k_bytes = make_quant_blocks(T_Q5_K, 4, 0xFF);
    let q6_k_bytes = make_quant_blocks(T_Q6_K, 4, 0xFF);

    let raw = GgufBuilder::new()
        .tensor("t_f32", vec![f32_vals.len() as u64], T_F32, f32_bytes)
        .tensor("t_f16", vec![F16_FINITE_SAMPLES.len() as u64], T_F16, f16_bytes)
        .tensor("t_bf16", vec![BF16_FINITE_SAMPLES.len() as u64], T_BF16, bf16_bytes)
        .tensor("t_q4_0", vec![4 * 32], T_Q4_0, q4_0_bytes)
        .tensor("t_q5_0", vec![4 * 32], T_Q5_0, q5_0_bytes)
        .tensor("t_q8_0", vec![4 * 32], T_Q8_0, q8_0_bytes)
        .tensor("t_q4_k", vec![4 * 256], T_Q4_K, q4_k_bytes)
        .tensor("t_q5_k", vec![4 * 256], T_Q5_K, q5_k_bytes)
        .tensor("t_q6_k", vec![4 * 256], T_Q6_K, q6_k_bytes)
        .build();

    let g = Gguf::from_bytes(&raw).expect("parse synthetic clean gguf");
    let report: WeightAuditReport = g.audit_weights_finiteness();

    assert!(report.is_clean(), "expected clean report, got {report:?}");
    assert_eq!(report.tensors_checked, 9);
    assert_eq!(
        report.blocks_checked,
        (f32_vals.len() + F16_FINITE_SAMPLES.len() + BF16_FINITE_SAMPLES.len() + 6 * 4) as u64
    );
    assert_eq!(report.non_finite_count, 0);
    assert!(report.offenders.is_empty());
}

#[test]
fn audit_weights_finiteness_detects_non_finite_f32() {
    for &(label, bad_bits) in F32_NON_FINITE_SAMPLES {
        let mut data = vec![0u8; 4 * 4];
        data[8..12].copy_from_slice(&bad_bits.to_le_bytes());
        let raw = GgufBuilder::new()
            .tensor("w_f32", vec![4], T_F32, data)
            .build();
        let g = Gguf::from_bytes(&raw).unwrap();
        let report = g.audit_weights_finiteness();
        assert!(!report.is_clean(), "F32 {label} must fail audit");
        assert_eq!(report.tensors_checked, 1);
        assert_eq!(report.blocks_checked, 4);
        assert_eq!(report.non_finite_count, 1, "F32 {label}");
        assert_eq!(
            report.offenders,
            vec![WeightOffender {
                tensor: "w_f32".into(),
                ty: T_F32,
                non_finite: 1,
            }]
        );
    }
}

#[test]
fn audit_weights_finiteness_detects_non_finite_f16() {
    for &(label, bad_bits) in F16_NON_FINITE_SAMPLES {
        let mut data = vec![0u8; 6 * 2];
        data[6..8].copy_from_slice(&bad_bits.to_le_bytes());
        let raw = GgufBuilder::new()
            .tensor("w_f16", vec![6], T_F16, data)
            .build();
        let g = Gguf::from_bytes(&raw).unwrap();
        let report = g.audit_weights_finiteness();
        assert!(!report.is_clean(), "F16 {label} must fail audit");
        assert_eq!(report.blocks_checked, 6);
        assert_eq!(report.non_finite_count, 1, "F16 {label}");
        assert_eq!(report.offenders.len(), 1);
        assert_eq!(report.offenders[0].tensor, "w_f16");
        assert_eq!(report.offenders[0].ty, T_F16);
        assert_eq!(report.offenders[0].non_finite, 1);
    }
}

#[test]
fn audit_weights_finiteness_detects_non_finite_bf16() {
    for &(label, bad_bits) in BF16_NON_FINITE_SAMPLES {
        let mut data = vec![0u8; 6 * 2];
        data[10..12].copy_from_slice(&bad_bits.to_le_bytes());
        let raw = GgufBuilder::new()
            .tensor("w_bf16", vec![6], T_BF16, data)
            .build();
        let g = Gguf::from_bytes(&raw).unwrap();
        let report = g.audit_weights_finiteness();
        assert!(!report.is_clean(), "BF16 {label} must fail audit");
        assert_eq!(report.blocks_checked, 6);
        assert_eq!(report.non_finite_count, 1, "BF16 {label}");
        assert_eq!(report.offenders[0].ty, T_BF16);
    }
}

#[test]
fn audit_weights_finiteness_detects_non_finite_q4_0_q5_0_q8_0_scales() {
    for (name, ty) in [
        ("w_q4_0", T_Q4_0),
        ("w_q5_0", T_Q5_0),
        ("w_q8_0", T_Q8_0),
    ] {
        let (bs, be) = try_type_block(ty).unwrap();
        for &(label, bad_bits) in F16_NON_FINITE_SAMPLES {
            let mut data = make_quant_blocks(ty, 3, 0xFF);
            // Corrupt block 2's fp16 `d` at offset 0..2
            let off = 2 * bs;
            data[off..off + 2].copy_from_slice(&bad_bits.to_le_bytes());

            let raw = GgufBuilder::new()
                .tensor(name, vec![(3 * be) as u64], ty, data)
                .build();
            let g = Gguf::from_bytes(&raw).unwrap();
            let report = g.audit_weights_finiteness();
            assert!(!report.is_clean(), "{name} {label} must fail audit");
            assert_eq!(report.blocks_checked, 3, "{name} {label}");
            assert_eq!(report.non_finite_count, 1, "{name} {label}");
            assert_eq!(
                report.offenders,
                vec![WeightOffender {
                    tensor: name.into(),
                    ty,
                    non_finite: 1,
                }]
            );
        }
    }
}

#[test]
fn audit_weights_finiteness_detects_non_finite_q4_k_and_q5_k_d_and_dmin() {
    for (name, ty) in [("w_q4_k", T_Q4_K), ("w_q5_k", T_Q5_K)] {
        let (bs, be) = try_type_block(ty).unwrap();
        for &(label, bad_bits) in F16_NON_FINITE_SAMPLES {
            // Corrupt `d` (0..2) in block 1
            let mut data_d = make_quant_blocks(ty, 2, 0xFF);
            data_d[bs..bs + 2].copy_from_slice(&bad_bits.to_le_bytes());
            let g_d = Gguf::from_bytes(
                &GgufBuilder::new()
                    .tensor(name, vec![(2 * be) as u64], ty, data_d)
                    .build(),
            )
            .unwrap();
            let r_d = g_d.audit_weights_finiteness();
            assert!(!r_d.is_clean(), "{name} d={label} must fail");
            assert_eq!(r_d.blocks_checked, 2);
            assert_eq!(r_d.non_finite_count, 1, "{name} d={label}");

            // Corrupt `dmin` (2..4) in block 1
            let mut data_dmin = make_quant_blocks(ty, 2, 0xFF);
            data_dmin[bs + 2..bs + 4].copy_from_slice(&bad_bits.to_le_bytes());
            let g_dmin = Gguf::from_bytes(
                &GgufBuilder::new()
                    .tensor(name, vec![(2 * be) as u64], ty, data_dmin)
                    .build(),
            )
            .unwrap();
            let r_dmin = g_dmin.audit_weights_finiteness();
            assert!(!r_dmin.is_clean(), "{name} dmin={label} must fail");
            assert_eq!(r_dmin.blocks_checked, 2);
            assert_eq!(r_dmin.non_finite_count, 1, "{name} dmin={label}");
        }

        // Corrupt BOTH `d` and `dmin` in block 0 -> counts 2 non-finite scales
        let mut data_both = make_quant_blocks(ty, 2, 0xFF);
        data_both[0..2].copy_from_slice(&0x7E00u16.to_le_bytes());
        data_both[2..4].copy_from_slice(&0xFC00u16.to_le_bytes());
        let g_both = Gguf::from_bytes(
            &GgufBuilder::new()
                .tensor(name, vec![(2 * be) as u64], ty, data_both)
                .build(),
        )
        .unwrap();
        let r_both = g_both.audit_weights_finiteness();
        assert_eq!(r_both.non_finite_count, 2, "{name} both d and dmin bad");
        assert_eq!(r_both.offenders[0].non_finite, 2);
    }
}

#[test]
fn audit_weights_finiteness_detects_non_finite_q6_k_scale_at_offset_208() {
    let (bs, be) = try_type_block(T_Q6_K).unwrap();
    assert_eq!(bs, 210);
    for &(label, bad_bits) in F16_NON_FINITE_SAMPLES {
        let mut data = make_quant_blocks(T_Q6_K, 3, 0xFF);
        // Corrupt block 1's fp16 `d` at offset 208..210
        let off = bs + 208;
        data[off..off + 2].copy_from_slice(&bad_bits.to_le_bytes());
        let raw = GgufBuilder::new()
            .tensor("w_q6_k", vec![(3 * be) as u64], T_Q6_K, data)
            .build();
        let g = Gguf::from_bytes(&raw).unwrap();
        let report = g.audit_weights_finiteness();
        assert!(!report.is_clean(), "Q6_K {label} must fail audit");
        assert_eq!(report.blocks_checked, 3);
        assert_eq!(report.non_finite_count, 1, "Q6_K {label}");
        assert_eq!(
            report.offenders,
            vec![WeightOffender {
                tensor: "w_q6_k".into(),
                ty: T_Q6_K,
                non_finite: 1,
            }]
        );
    }
}

#[test]
fn audit_weights_finiteness_parallel_multi_tensor_and_deterministic_order() {
    let mut b = GgufBuilder::new();
    // 16 clean tensors across multiple types
    for i in 0..16 {
        b = b.tensor(
            format!("clean_{i:02}"),
            vec![256],
            T_Q4_K,
            make_quant_blocks(T_Q4_K, 1, 0xFF),
        );
    }
    // 3 bad tensors inserted out of alphabetical order
    let mut bad_z = make_quant_blocks(T_Q6_K, 2, 0x00);
    bad_z[208..210].copy_from_slice(&0x7C00u16.to_le_bytes()); // +Inf
    b = b.tensor("z_bad_q6k", vec![512], T_Q6_K, bad_z);

    let mut bad_a = vec![0u8; 16];
    bad_a[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    bad_a[12..16].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
    b = b.tensor("a_bad_f32", vec![4], T_F32, bad_a);

    let mut bad_m = make_quant_blocks(T_Q4_K, 2, 0x00);
    bad_m[144 + 2..144 + 4].copy_from_slice(&0x7E00u16.to_le_bytes()); // dmin NaN in block 1
    b = b.tensor("m_bad_q4k", vec![512], T_Q4_K, bad_m);

    let g = Gguf::from_bytes(&b.build()).unwrap();
    let report = g.audit_weights_finiteness();

    assert!(!report.is_clean());
    assert_eq!(report.tensors_checked, 19);
    assert_eq!(report.blocks_checked, 16 + 2 + 4 + 2);
    assert_eq!(report.non_finite_count, 4);
    assert_eq!(
        report.offenders,
        vec![
            WeightOffender {
                tensor: "a_bad_f32".into(),
                ty: T_F32,
                non_finite: 2,
            },
            WeightOffender {
                tensor: "m_bad_q4k".into(),
                ty: T_Q4_K,
                non_finite: 1,
            },
            WeightOffender {
                tensor: "z_bad_q6k".into(),
                ty: T_Q6_K,
                non_finite: 1,
            },
        ]
    );
}

#[test]
fn audit_weights_finiteness_empty_gguf_is_clean() {
    let raw = GgufBuilder::new().build();
    let g = Gguf::from_bytes(&raw).unwrap();
    let report = g.audit_weights_finiteness();
    assert!(report.is_clean());
    assert_eq!(report.tensors_checked, 0);
    assert_eq!(report.blocks_checked, 0);
    assert_eq!(report.non_finite_count, 0);
    assert!(report.offenders.is_empty());
}
