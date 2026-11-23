mod common;

use common::gguf_builder::{
    GgufBuilder, KvVal, TensorSpec, GGUF_TYPE_ARRAY, GGUF_TYPE_UINT32,
};
use inkling::engine::{
    BLOCK, E, EXT, FF, HD, LOOP_END, LOOP_START, NEXP, NH, NKV, NREL, N_PASS, N_PHYS, TOPK,
};
use inkling::gguf::{
    try_type_block, Gguf, Tensor, T_BF16, T_F32, T_Q4_K, T_Q5_0, T_Q6_K, T_Q8_0,
};
use proptest::prelude::*;
use std::io::{ErrorKind, Write};

fn write_raw_header(nt: u64, nkv: u64) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&nt.to_le_bytes());
    out.extend_from_slice(&nkv.to_le_bytes());
    out
}

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

/// Build a valid minimal `inkling2_loop` GGUF descriptor table where all tensors overlap at offset 0.
/// Use `open_sparse_gguf` to back the data region with a sparse zero file (instant, tiny RAM/disk).
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

/// Write a `GgufBuilder` with empty overlapping tensor payloads into a sparse temp file
/// sized to hold the largest tensor, then open via `Gguf::open`.
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

// ----------------------------------------------------------------------------- parser hardening tests

#[test]
fn rejects_empty_bytes() {
    assert_invalid_data(Gguf::from_bytes(&[]).map(|_| ()), "empty bytes");
}

#[test]
fn rejects_empty_file_in_open() {
    let tmp = tempfile::NamedTempFile::new().expect("create temp file");
    assert_invalid_data(
        Gguf::open(tmp.path().to_str().unwrap()).map(|_| ()),
        "empty file in Gguf::open",
    );
}

#[test]
fn rejects_bad_magic() {
    let mut b = GgufBuilder::new();
    b.magic = *b"NOPE";
    let raw = b.build();
    assert_invalid_data(Gguf::from_bytes(&raw).map(|_| ()), "bad magic");
}

#[test]
fn rejects_truncated_headers() {
    let full = GgufBuilder::new()
        .kv("k", KvVal::Str("hello".into()))
        .tensor("t", vec![4], T_F32, vec![0u8; 16])
        .build();
    for cut in [1usize, 3, 4, 8, 12, 23, 28, full.len() - 1] {
        assert_invalid_data(
            Gguf::from_bytes(&full[..cut]).map(|_| ()),
            &format!("truncated to {cut} bytes"),
        );
    }
}

#[test]
fn rejects_zero_and_overflow_alignment() {
    for val in [
        KvVal::U32(0),
        KvVal::U64(0),
        KvVal::I32(0),
        KvVal::I32(-32),
        KvVal::U64(u64::MAX),
    ] {
        let raw = GgufBuilder::new()
            .kv("general.alignment", val.clone())
            .build();
        assert_invalid_data(
            Gguf::from_bytes(&raw).map(|_| ()),
            &format!("alignment {val:?}"),
        );
    }
}

#[test]
fn rejects_nested_arrays_without_stack_overflow() {
    let mut shallow = Vec::new();
    shallow.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
    shallow.extend_from_slice(&0u64.to_le_bytes());
    let raw_shallow = GgufBuilder::new()
        .kv(
            "nested.shallow",
            KvVal::Raw {
                ty: GGUF_TYPE_ARRAY,
                payload: shallow,
            },
        )
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_shallow).map(|_| ()),
        "shallow nested array",
    );

    let depth = 5000usize;
    let mut payload = Vec::with_capacity(depth * 12 + 12);
    for _ in 0..depth {
        payload.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        payload.extend_from_slice(&1u64.to_le_bytes());
    }
    payload.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
    payload.extend_from_slice(&0u64.to_le_bytes());
    let raw_deep = GgufBuilder::new()
        .kv(
            "nested.deep",
            KvVal::Raw {
                ty: GGUF_TYPE_ARRAY,
                payload,
            },
        )
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_deep).map(|_| ()),
        "depth-5000 nested array",
    );
}

#[test]
fn rejects_tensor_ne_product_overflow() {
    let raw = GgufBuilder::new()
        .tensor_with_offset("overflow", vec![1u64 << 33, 1u64 << 33], T_F32, 0, vec![0u8; 64])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw).map(|_| ()),
        "ne.product() u64 overflow",
    );
}

#[test]
fn rejects_tensor_nbytes_overflow() {
    let raw = GgufBuilder::new()
        .tensor_with_offset("nbytes_ovf", vec![(u64::MAX / 4) + 1], T_F32, 0, vec![0u8; 64])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw).map(|_| ()),
        "nbytes u64 overflow",
    );
}

#[test]
fn rejects_tensor_file_off_wraparound() {
    let raw = GgufBuilder::new()
        .tensor_with_offset("wrap_off", vec![4], T_F32, u64::MAX - 8, vec![0u8; 64])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw).map(|_| ()),
        "file_off + data_off wraparound",
    );

    let probe = GgufBuilder::new()
        .tensor_with_offset("w", vec![4], T_F32, 0, vec![0u8; 128])
        .build();
    let data_off = (probe.len() - 128) as u64;
    let crafted_off = u64::MAX - data_off - 4;
    let raw2 = GgufBuilder::new()
        .tensor_with_offset("w", vec![4], T_F32, crafted_off, vec![0u8; 128])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw2).map(|_| ()),
        "file_off + nbytes wraparound",
    );
}

#[test]
fn rejects_rd_take_overflow_and_huge_string_len() {
    for str_len in [u64::MAX, u64::MAX - 16, 1u64 << 40] {
        let mut raw = write_raw_header(0, 1);
        raw.extend_from_slice(&str_len.to_le_bytes());
        raw.extend_from_slice(&[0u8; 16]);
        assert_invalid_data(
            Gguf::from_bytes(&raw).map(|_| ()),
            &format!("huge string len {str_len}"),
        );
    }
}

#[test]
fn rejects_huge_counts_dos() {
    for nkv in [u64::MAX, 1_000_000_000] {
        let raw = write_raw_header(0, nkv);
        assert_invalid_data(
            Gguf::from_bytes(&raw).map(|_| ()),
            &format!("huge nkv {nkv}"),
        );
    }
    for nt in [u64::MAX, 1_000_000_000] {
        let raw = write_raw_header(nt, 0);
        assert_invalid_data(
            Gguf::from_bytes(&raw).map(|_| ()),
            &format!("huge nt {nt}"),
        );
    }
    for nd in [u32::MAX, 10_000_000] {
        let mut raw = write_raw_header(1, 0);
        raw.extend_from_slice(&1u64.to_le_bytes());
        raw.push(b'x');
        raw.extend_from_slice(&nd.to_le_bytes());
        assert_invalid_data(
            Gguf::from_bytes(&raw).map(|_| ()),
            &format!("huge nd {nd}"),
        );
    }
    for arr_len in [u64::MAX, 1_000_000_000] {
        let mut payload = Vec::new();
        payload.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        payload.extend_from_slice(&arr_len.to_le_bytes());
        let raw = GgufBuilder::new()
            .kv(
                "huge_arr",
                KvVal::Raw {
                    ty: GGUF_TYPE_ARRAY,
                    payload,
                },
            )
            .build();
        assert_invalid_data(
            Gguf::from_bytes(&raw).map(|_| ()),
            &format!("huge array len {arr_len}"),
        );
    }
}

#[test]
fn rejects_unknown_types() {
    assert_invalid_data(try_type_block(999), "try_type_block(999)");

    let raw_ty = GgufBuilder::new()
        .tensor("bad_ty", vec![4], 999, vec![0u8; 16])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_ty).map(|_| ()),
        "unknown ggml tensor type",
    );

    let raw_vt = GgufBuilder::new()
        .kv(
            "bad_vt",
            KvVal::Raw {
                ty: 99,
                payload: vec![0u8; 8],
            },
        )
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_vt).map(|_| ()),
        "unknown GGUF value type",
    );

    let mut arr_payload = Vec::new();
    arr_payload.extend_from_slice(&99u32.to_le_bytes());
    arr_payload.extend_from_slice(&0u64.to_le_bytes());
    let raw_et = GgufBuilder::new()
        .kv(
            "bad_et",
            KvVal::Raw {
                ty: GGUF_TYPE_ARRAY,
                payload: arr_payload,
            },
        )
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_et).map(|_| ()),
        "unknown GGUF array element type",
    );
}

#[test]
fn rejects_block_indivisible_and_zero_rank_tensors() {
    let raw_indiv = GgufBuilder::new()
        .tensor_with_offset("indiv", vec![10], T_Q4_K, 0, vec![0u8; 256])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_indiv).map(|_| ()),
        "ne=[10] not divisible by Q4_K be=256",
    );

    let raw_row_indiv = GgufBuilder::new()
        .tensor_with_offset("row_indiv", vec![2, 128], T_Q4_K, 0, vec![0u8; 256])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_row_indiv).map(|_| ()),
        "ne=[2, 128] row dim not divisible by Q4_K be=256",
    );

    let raw_zero_rank = GgufBuilder::new()
        .tensor_with_offset("zero_rank", vec![], T_F32, 0, vec![0u8; 16])
        .build();
    assert_invalid_data(
        Gguf::from_bytes(&raw_zero_rank).map(|_| ()),
        "zero-rank tensor ne=[]",
    );

    let z = Tensor {
        name: "zero".into(),
        ne: vec![],
        ty: T_F32,
        file_off: 0,
        nbytes: 0,
    };
    assert_eq!(z.row_bytes(), 0);
    assert_invalid_data(z.try_row_bytes(), "try_row_bytes on zero-rank tensor");

    let bad = Tensor {
        name: "bad".into(),
        ne: vec![10],
        ty: T_Q4_K,
        file_off: 0,
        nbytes: 0,
    };
    assert_invalid_data(bad.try_row_bytes(), "try_row_bytes on indivisible Q4_K");
}

// ----------------------------------------------------------------------------- model validation tests

#[test]
fn validate_inkling_model_accepts_valid_fixture() {
    let b = valid_inkling_model_builder(1000, 1024).kv(
        "tokenizer.ggml.tokens",
        KvVal::ArrStr((0..1000).map(|i| format!("t{i}")).collect()),
    );
    let g = open_sparse_gguf(&b);
    g.validate_inkling_model().expect("valid inkling2_loop model");
}

#[test]
fn validate_inkling_model_rejects_missing_required_tensor() {
    for missing in [
        "blk.0.attn_q.weight",
        "blk.5.attn_res_query.weight",
        "blk.5.attn_k_conv_proj.weight",
        "blk.6.loop_q_ctrl_down.weight",
        "output.weight",
    ] {
        let mut b = valid_inkling_model_builder(1000, 1024);
        b.tensors.retain(|t| t.name != missing);
        let g = open_sparse_gguf(&b);
        assert_invalid_data(
            g.validate_inkling_model(),
            &format!("missing tensor {missing}"),
        );
    }
}

#[test]
fn validate_inkling_model_rejects_wrong_tensor_shape_or_rank() {
    // Undersized rows on attn_q
    let mut b1 = valid_inkling_model_builder(1000, 1024);
    let t = b1.tensors.iter_mut().find(|t| t.name == "blk.0.attn_q.weight").unwrap();
    t.ne = vec![E as u64, 1024];
    let g1 = open_sparse_gguf(&b1);
    assert_invalid_data(g1.validate_inkling_model(), "undersized blk.0.attn_q.weight");

    // Wrong rank (2D instead of 3D) on ffn_gate_up_exps
    let mut b2 = valid_inkling_model_builder(1000, 1024);
    let t = b2
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ffn_gate_up_exps.weight")
        .unwrap();
    t.ne = vec![E as u64, (2 * FF) as u64];
    let g2 = open_sparse_gguf(&b2);
    assert_invalid_data(g2.validate_inkling_model(), "2D ffn_gate_up_exps.weight");

    // Wrong attn_res_query count on full-attention layer (1 instead of 3)
    let mut b3 = valid_inkling_model_builder(1000, 1024);
    let t = b3
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.5.attn_res_query.weight")
        .unwrap();
    t.ne = vec![E as u64, 1];
    let g3 = open_sparse_gguf(&b3);
    assert_invalid_data(g3.validate_inkling_model(), "wrong blk.5.attn_res_query.weight shape");
}

#[test]
fn validate_inkling_model_rejects_wrong_tensor_type() {
    for (name, bad_ty) in [
        ("blk.0.attn_q.weight", T_Q6_K),
        ("blk.0.ffn_down_exps.weight", T_F32),
        ("blk.0.ffn_norm.weight", T_BF16),
        ("blk.5.attn_k_conv_proj.weight", T_F32),
        ("output.weight", T_Q4_K),
        ("token_embd.weight", T_Q6_K),
    ] {
        let mut b = valid_inkling_model_builder(1000, 1024);
        let t = b.tensors.iter_mut().find(|t| t.name == name).unwrap();
        t.ty = bad_ty;
        let g = open_sparse_gguf(&b);
        assert_invalid_data(
            g.validate_inkling_model(),
            &format!("wrong ty {bad_ty} on {name}"),
        );
    }
}

#[test]
fn validate_inkling_model_rejects_short_swa_pattern() {
    let mut b = valid_inkling_model_builder(1000, 1024);
    for (k, v) in &mut b.kvs {
        if k == "inkling2_loop.attention.sliding_window_pattern" {
            *v = KvVal::ArrBool(vec![true; N_PHYS - 1]);
        }
    }
    let g = open_sparse_gguf(&b);
    assert_invalid_data(g.validate_inkling_model(), "swa_pat.len() < N_PHYS");
}

#[test]
fn validate_inkling_model_rejects_n_vocab_exceeding_n_vocab_pad() {
    let b = valid_inkling_model_builder(1025, 1024);
    let g = open_sparse_gguf(&b);
    assert_invalid_data(g.validate_inkling_model(), "n_vocab > n_vocab_pad");
}

#[test]
fn validate_inkling_model_rejects_tokenizer_exceeding_tok_embd_rows() {
    let b = valid_inkling_model_builder(1000, 1024).kv(
        "tokenizer.ggml.tokens",
        KvVal::ArrStr((0..1025).map(|i| format!("t{i}")).collect()),
    );
    let g = open_sparse_gguf(&b);
    assert_invalid_data(
        g.validate_inkling_model(),
        "id_to_tok.len() > tok_embd.rows",
    );
}

#[test]
fn validate_inkling_model_rejects_bad_architecture_and_non_finite_hyperparam() {
    let mut b1 = valid_inkling_model_builder(1000, 1024);
    for (k, v) in &mut b1.kvs {
        if k == "general.architecture" {
            *v = KvVal::Str("llama".into());
        }
    }
    let g1 = open_sparse_gguf(&b1);
    assert_invalid_data(g1.validate_inkling_model(), "wrong architecture");

    let mut b2 = valid_inkling_model_builder(1000, 1024);
    for (k, v) in &mut b2.kvs {
        if k == "inkling2_loop.attention.layer_norm_rms_epsilon" {
            *v = KvVal::F32(f32::NAN);
        }
    }
    let g2 = open_sparse_gguf(&b2);
    assert_invalid_data(g2.validate_inkling_model(), "NaN rms_epsilon");
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn proptest_from_bytes_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        if let Ok(g) = Gguf::from_bytes(&bytes) {
            for t in g.tensors.values() {
                prop_assert_eq!(g.bytes(t).len() as u64, t.nbytes);
                prop_assert!(t.try_row_bytes().is_ok());
            }
        }
    }

    #[test]
    fn proptest_gguf_prefixed_bytes_never_panics(tail in prop::collection::vec(any::<u8>(), 0..512)) {
        let mut bytes = Vec::with_capacity(8 + tail.len());
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&tail);
        if let Ok(g) = Gguf::from_bytes(&bytes) {
            for t in g.tensors.values() {
                prop_assert_eq!(g.bytes(t).len() as u64, t.nbytes);
                prop_assert!(t.try_row_bytes().is_ok());
            }
        }
    }
}
