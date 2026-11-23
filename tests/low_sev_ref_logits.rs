use std::collections::HashMap;
use std::fs;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;
use tempfile::TempDir;

const MOCK_LLAMA_H: &str = r#"#ifndef LLAMA_H
#define LLAMA_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t llama_token;
typedef int32_t llama_pos;

struct llama_model;
struct llama_vocab;
struct llama_context;
struct llama_batch_ext;

enum llama_flash_attn_type {
    LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
    LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
};

enum llama_process_type {
    LLAMA_PROCESS_TYPE_DECODE = 0,
};

struct llama_model_params {
    int32_t n_gpu_layers;
};

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    enum llama_flash_attn_type flash_attn_type;
};

void llama_backend_init(void);
void llama_backend_free(void);

struct llama_model_params llama_model_default_params(void);
struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);
void llama_model_free(struct llama_model * model);

const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);

int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);

int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);

struct llama_context_params llama_context_default_params(void);
struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);
void llama_free(struct llama_context * ctx);

struct llama_batch_ext * llama_batch_ext_init(struct llama_context * ctx);
void llama_batch_ext_free(struct llama_batch_ext * batch);
void llama_batch_ext_clear(struct llama_batch_ext * batch);
int32_t llama_batch_ext_add_token(struct llama_batch_ext * batch, int32_t seq_id, llama_token token);
void llama_batch_ext_set_pos(struct llama_batch_ext * batch, int32_t idx, const llama_pos * pos);
void llama_batch_ext_set_output_logits(struct llama_batch_ext * batch, int32_t idx, bool output);

int32_t llama_process(struct llama_context * ctx, enum llama_process_type type, struct llama_batch_ext * batch);
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_H
"#;

const MOCK_LLAMA_STUB_CPP: &str = r#"#include "llama.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct llama_vocab {
    int32_t n_vocab;
};

struct llama_model {
    uint32_t magic;
    bool freed;
    llama_vocab vocab;
    bool null_vocab;
};

struct llama_context {
    uint32_t magic;
    bool freed;
    llama_model * model;
    int32_t n_vocab;
    std::vector<float> logits;
    int32_t last_batch_len;
};

struct llama_batch_ext {
    uint32_t magic;
    bool freed;
    llama_context * ctx;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<bool> outputs;
};

namespace {

int g_backend_inits = 0;
int g_backend_frees = 0;
int g_models_alloc = 0;
int g_models_free = 0;
int g_ctxs_alloc = 0;
int g_ctxs_free = 0;
int g_batches_alloc = 0;
int g_batches_free = 0;
int g_double_frees = 0;
int g_order_errors = 0;
int g_tokenize_calls = 0;
bool g_atexit_registered = false;

bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

std::string env_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

void write_lifecycle_log() {
    const char * path = std::getenv("MOCK_LIFECYCLE_FILE");
    if (!path || !path[0]) return;
    FILE * f = std::fopen(path, "w");
    if (!f) return;
    std::fprintf(f, "backend_inits=%d\n", g_backend_inits);
    std::fprintf(f, "backend_frees=%d\n", g_backend_frees);
    std::fprintf(f, "models_alloc=%d\n", g_models_alloc);
    std::fprintf(f, "models_free=%d\n", g_models_free);
    std::fprintf(f, "ctxs_alloc=%d\n", g_ctxs_alloc);
    std::fprintf(f, "ctxs_free=%d\n", g_ctxs_free);
    std::fprintf(f, "batches_alloc=%d\n", g_batches_alloc);
    std::fprintf(f, "batches_free=%d\n", g_batches_free);
    std::fprintf(f, "double_frees=%d\n", g_double_frees);
    std::fprintf(f, "order_errors=%d\n", g_order_errors);
    std::fclose(f);
}

void ensure_atexit() {
    if (!g_atexit_registered) {
        std::atexit(write_lifecycle_log);
        g_atexit_registered = true;
    }
}

} // namespace

extern "C" {

void llama_backend_init(void) {
    ensure_atexit();
    g_backend_inits++;
}

void llama_backend_free(void) {
    ensure_atexit();
    if (g_models_alloc != g_models_free || g_ctxs_alloc != g_ctxs_free || g_batches_alloc != g_batches_free) {
        g_order_errors++;
    }
    g_backend_frees++;
}

struct llama_model_params llama_model_default_params(void) {
    llama_model_params p{};
    p.n_gpu_layers = 0;
    return p;
}

struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params /*params*/) {
    ensure_atexit();
    if (env_flag("MOCK_FAIL_MODEL") || !path_model || !path_model[0]) {
        return nullptr;
    }
    FILE * probe = std::fopen(path_model, "rb");
    if (!probe) {
        return nullptr;
    }
    std::fclose(probe);

    auto * m = new llama_model{};
    m->magic = 0x11A0A111u;
    m->freed = false;
    int32_t nv = 8;
    if (const char * vs = std::getenv("MOCK_VOCAB_SIZE")) {
        nv = std::atoi(vs);
    }
    m->vocab.n_vocab = nv;
    m->null_vocab = env_flag("MOCK_FAIL_VOCAB");
    g_models_alloc++;
    return m;
}

void llama_model_free(struct llama_model * model) {
    ensure_atexit();
    if (!model) return;
    if (model->freed || model->magic != 0x11A0A111u) {
        g_double_frees++;
        return;
    }
    if (g_ctxs_alloc != g_ctxs_free || g_batches_alloc != g_batches_free) {
        g_order_errors++;
    }
    model->freed = true;
    g_models_free++;
    delete model;
}

const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model) {
    // Intentionally dereference model like real llama.cpp so unchecked NULL crashes.
    if (model->null_vocab) {
        return nullptr;
    }
    return &model->vocab;
}

int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab) {
    // Intentionally dereference vocab like real llama.cpp.
    return vocab->n_vocab;
}

int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool /*add_special*/,
    bool /*parse_special*/) {
    // Dereference vocab to catch unchecked NULL.
    volatile int32_t nv = vocab->n_vocab;
    (void)nv;
    (void)text;
    (void)text_len;

    g_tokenize_calls++;
    const std::string mode = env_str("MOCK_TOKENIZE_MODE");

    if (mode == "always_negative") {
        return -1;
    }
    if (mode == "always_negative_big") {
        return -5;
    }
    if (mode == "int_min") {
        return INT32_MIN;
    }
    if (mode == "zero") {
        return 0;
    }
    if (mode == "positive_on_probe") {
        return 5;
    }
    if (mode == "two_pass_grow") {
        // Call 1 (probe): under-reports -2; Call 2 (cap=2): reports -4; Call 3 (cap>=4): succeeds with 4 tokens.
        if (tokens == nullptr || n_tokens_max == 0) {
            return -2;
        }
        if (n_tokens_max < 4) {
            return -4;
        }
        tokens[0] = 1;
        tokens[1] = 2;
        tokens[2] = 3;
        tokens[3] = 4;
        return 4;
    }

    // Default happy path: 3 prompt tokens [1, 2, 3].
    if (tokens == nullptr || n_tokens_max < 3) {
        return -3;
    }
    tokens[0] = 1;
    tokens[1] = 2;
    tokens[2] = 3;
    return 3;
}

int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token /*token*/,
    char * buf,
    int32_t length,
    int32_t /*lstrip*/,
    bool /*special*/) {
    volatile int32_t nv = vocab->n_vocab;
    (void)nv;
    const char piece[] = "tok";
    if (!buf || length < 3) return -3;
    std::memcpy(buf, piece, 3);
    return 3;
}

struct llama_context_params llama_context_default_params(void) {
    llama_context_params p{};
    p.n_ctx = 512;
    p.n_batch = 512;
    p.n_ubatch = 512;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    return p;
}

struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params /*params*/) {
    ensure_atexit();
    // Dereference model like real llama.cpp.
    volatile uint32_t m = model->magic;
    (void)m;
    if (env_flag("MOCK_FAIL_CTX")) {
        return nullptr;
    }
    auto * ctx = new llama_context{};
    ctx->magic = 0xC07E7111u;
    ctx->freed = false;
    ctx->model = model;
    ctx->n_vocab = model->vocab.n_vocab;
    ctx->last_batch_len = 0;
    g_ctxs_alloc++;
    return ctx;
}

void llama_free(struct llama_context * ctx) {
    ensure_atexit();
    if (!ctx) return;
    if (ctx->freed || ctx->magic != 0xC07E7111u) {
        g_double_frees++;
        return;
    }
    if (g_batches_alloc != g_batches_free) {
        g_order_errors++;
    }
    ctx->freed = true;
    g_ctxs_free++;
    delete ctx;
}

struct llama_batch_ext * llama_batch_ext_init(struct llama_context * ctx) {
    ensure_atexit();
    // Dereference ctx like real llama.cpp.
    volatile uint32_t m = ctx->magic;
    (void)m;
    if (env_flag("MOCK_FAIL_BATCH")) {
        return nullptr;
    }
    auto * b = new llama_batch_ext{};
    b->magic = 0xBA7C4111u;
    b->freed = false;
    b->ctx = ctx;
    g_batches_alloc++;
    return b;
}

void llama_batch_ext_free(struct llama_batch_ext * batch) {
    ensure_atexit();
    if (!batch) return;
    if (batch->freed || batch->magic != 0xBA7C4111u) {
        g_double_frees++;
        return;
    }
    batch->freed = true;
    g_batches_free++;
    delete batch;
}

void llama_batch_ext_clear(struct llama_batch_ext * batch) {
    batch->tokens.clear();
    batch->positions.clear();
    batch->outputs.clear();
}

int32_t llama_batch_ext_add_token(struct llama_batch_ext * batch, int32_t /*seq_id*/, llama_token token) {
    int32_t idx = static_cast<int32_t>(batch->tokens.size());
    batch->tokens.push_back(token);
    batch->positions.push_back(0);
    batch->outputs.push_back(false);
    return idx;
}

void llama_batch_ext_set_pos(struct llama_batch_ext * batch, int32_t idx, const llama_pos * pos) {
    if (idx >= 0 && static_cast<size_t>(idx) < batch->positions.size() && pos) {
        batch->positions[static_cast<size_t>(idx)] = *pos;
    }
}

void llama_batch_ext_set_output_logits(struct llama_batch_ext * batch, int32_t idx, bool output) {
    if (idx >= 0 && static_cast<size_t>(idx) < batch->outputs.size()) {
        batch->outputs[static_cast<size_t>(idx)] = output;
    }
}

int32_t llama_process(struct llama_context * ctx, enum llama_process_type /*type*/, struct llama_batch_ext * batch) {
    volatile uint32_t cm = ctx->magic;
    volatile uint32_t bm = batch->magic;
    (void)cm;
    (void)bm;

    if (env_flag("MOCK_FAIL_DECODE")) {
        return -1;
    }

    const int32_t cnt = static_cast<int32_t>(batch->tokens.size());
    const int32_t nv = ctx->n_vocab;
    ctx->last_batch_len = cnt;
    ctx->logits.assign(static_cast<size_t>(cnt) * static_cast<size_t>(nv), 0.0f);

    for (int32_t i = 0; i < cnt; ++i) {
        const int32_t tok = batch->tokens[static_cast<size_t>(i)];
        const int32_t pos = batch->positions[static_cast<size_t>(i)];
        const int32_t next_tok = (tok + 1) % nv;
        float * row = &ctx->logits[static_cast<size_t>(i) * static_cast<size_t>(nv)];
        for (int32_t v = 0; v < nv; ++v) {
            row[v] = static_cast<float>(v) * 0.1f;
        }
        row[next_tok] = 10.0f + static_cast<float>(pos);
    }
    return 0;
}

float * llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    volatile uint32_t cm = ctx->magic;
    (void)cm;
    if (env_flag("MOCK_NULL_LOGITS")) {
        return nullptr;
    }
    if (env_flag("MOCK_NULL_LOGITS_LAST") && i == -1) {
        return nullptr;
    }
    if (ctx->last_batch_len <= 0 || ctx->n_vocab <= 0) {
        return nullptr;
    }
    int32_t idx = (i == -1) ? (ctx->last_batch_len - 1) : i;
    if (idx < 0 || idx >= ctx->last_batch_len) {
        return nullptr;
    }
    return &ctx->logits[static_cast<size_t>(idx) * static_cast<size_t>(ctx->n_vocab)];
}

} // extern "C"
"#;

struct CompiledBinary {
    _dir: TempDir,
    bin_path: PathBuf,
    dummy_model: PathBuf,
}

fn ref_logits_cpp_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("ref_logits.cpp")
}

fn compiled_binary() -> &'static CompiledBinary {
    static BIN: OnceLock<CompiledBinary> = OnceLock::new();
    BIN.get_or_init(|| {
        let dir = TempDir::new().expect("create temp dir for ref_logits compilation");
        let inc_dir = dir.path().join("inc");
        fs::create_dir_all(&inc_dir).expect("create inc dir");

        let header = inc_dir.join("llama.h");
        let stub = dir.path().join("llama_stub.cpp");
        let bin_path = dir.path().join("ref_logits");
        let dummy_model = dir.path().join("dummy.gguf");

        fs::write(&header, MOCK_LLAMA_H).expect("write mock llama.h");
        fs::write(&stub, MOCK_LLAMA_STUB_CPP).expect("write mock llama_stub.cpp");
        fs::write(&dummy_model, b"GGUF").expect("write dummy model file");

        let out = Command::new("c++")
            .arg("-std=c++17")
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Werror")
            .arg("-O0")
            .arg("-g")
            .arg("-I")
            .arg(&inc_dir)
            .arg(ref_logits_cpp_path())
            .arg(&stub)
            .arg("-o")
            .arg(&bin_path)
            .output()
            .expect("invoke c++ compiler");

        assert!(
            out.status.success(),
            "failed to compile tools/ref_logits.cpp against mock llama.h:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );

        CompiledBinary {
            _dir: dir,
            bin_path,
            dummy_model,
        }
    })
}

fn run_ref(args: &[&str], envs: &[(&str, &str)]) -> (Output, HashMap<String, i32>) {
    let bin = compiled_binary();
    let log_dir = TempDir::new().expect("temp dir for lifecycle log");
    let log_path = log_dir.path().join("lifecycle.txt");

    let mut cmd = Command::new(&bin.bin_path);
    cmd.args(args);
    cmd.env("MOCK_LIFECYCLE_FILE", &log_path);
    for (k, v) in envs {
        cmd.env(k, v);
    }

    let output = cmd.output().expect("execute ref_logits binary");
    let mut map = HashMap::new();
    if let Ok(text) = fs::read_to_string(&log_path) {
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                if let Ok(n) = v.trim().parse::<i32>() {
                    map.insert(k.trim().to_string(), n);
                }
            }
        }
    }
    (output, map)
}

fn assert_clean_failure(output: &Output, lifecycle: &HashMap<String, i32>, context: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.signal().is_none(),
        "{context}: binary crashed with signal {:?} instead of clean exit! stderr:\n{stderr}",
        output.status.signal()
    );
    let code = output
        .status
        .code()
        .expect("process exited without signal must have exit code");
    assert_ne!(
        code, 0,
        "{context}: expected non-zero exit code, got 0! stderr:\n{stderr}"
    );
    assert!(
        !stderr.trim().is_empty(),
        "{context}: expected informative error message on stderr, got empty stderr"
    );
    assert_lifecycle_clean(lifecycle, context);
}

fn assert_lifecycle_clean(lifecycle: &HashMap<String, i32>, context: &str) {
    if lifecycle.is_empty() {
        return;
    }
    let get = |k: &str| *lifecycle.get(k).unwrap_or(&0);
    assert_eq!(
        get("backend_inits"),
        get("backend_frees"),
        "{context}: llama_backend_init/free mismatch: {lifecycle:?}"
    );
    assert_eq!(
        get("models_alloc"),
        get("models_free"),
        "{context}: llama_model alloc/free mismatch (leak): {lifecycle:?}"
    );
    assert_eq!(
        get("ctxs_alloc"),
        get("ctxs_free"),
        "{context}: llama_context alloc/free mismatch (leak): {lifecycle:?}"
    );
    assert_eq!(
        get("batches_alloc"),
        get("batches_free"),
        "{context}: llama_batch_ext alloc/free mismatch (leak): {lifecycle:?}"
    );
    assert_eq!(
        get("double_frees"),
        0,
        "{context}: detected double-free: {lifecycle:?}"
    );
    assert_eq!(
        get("order_errors"),
        0,
        "{context}: detected out-of-order teardown: {lifecycle:?}"
    );
}

fn read_f32_le_vec(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).expect("read logits file");
    let (chunks, remainder) = bytes.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "logits byte length must be multiple of 4"
    );
    chunks.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

#[test]
fn compiles_with_strict_warnings_zero_errors() {
    let dir = TempDir::new().expect("temp dir");
    let inc_dir = dir.path().join("inc");
    fs::create_dir_all(&inc_dir).expect("create inc");
    fs::write(inc_dir.join("llama.h"), MOCK_LLAMA_H).expect("write llama.h");
    let obj = dir.path().join("ref_logits.o");

    let out = Command::new("c++")
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-c")
        .arg("-I")
        .arg(&inc_dir)
        .arg(ref_logits_cpp_path())
        .arg("-o")
        .arg(&obj)
        .output()
        .expect("run c++ -Wall -Wextra -Werror");

    assert!(
        out.status.success(),
        "tools/ref_logits.cpp failed -Wall -Wextra -Werror:\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn missing_and_invalid_cli_args_fail_cleanly() {
    let bin = compiled_binary();
    let model = bin.dummy_model.to_str().unwrap();
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    let prefix_s = prefix.to_str().unwrap();

    // Missing arguments (argc = 1, 3, 4)
    let (out, lc) = run_ref(&[], &[]);
    assert_clean_failure(&out, &lc, "no args");

    let (out, lc) = run_ref(&[model, "hello"], &[]);
    assert_clean_failure(&out, &lc, "argc=3");

    let (out, lc) = run_ref(&[model, "hello", "2"], &[]);
    assert_clean_failure(&out, &lc, "argc=4");

    // Invalid n_gen (negative or non-numeric)
    let (out, lc) = run_ref(&[model, "hello", "-1", prefix_s], &[]);
    assert_clean_failure(&out, &lc, "negative n_gen");

    let (out, lc) = run_ref(&[model, "hello", "not_a_number", prefix_s], &[]);
    assert_clean_failure(&out, &lc, "non-numeric n_gen");
}

#[test]
fn missing_model_file_returns_null_and_exits_cleanly() {
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    let (out, lc) = run_ref(
        &[
            "/nonexistent_ref_logits_model_dir/missing.gguf",
            "hello",
            "2",
            prefix.to_str().unwrap(),
        ],
        &[],
    );
    assert_clean_failure(&out, &lc, "missing model file");
}

#[test]
fn null_vocab_exits_cleanly_and_frees_model() {
    let bin = compiled_binary();
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    let (out, lc) = run_ref(
        &[
            bin.dummy_model.to_str().unwrap(),
            "hello",
            "2",
            prefix.to_str().unwrap(),
        ],
        &[("MOCK_FAIL_VOCAB", "1")],
    );
    assert_clean_failure(&out, &lc, "null vocab");
    assert_eq!(lc.get("models_alloc"), Some(&1));
    assert_eq!(lc.get("models_free"), Some(&1));
}

#[test]
fn null_context_exits_cleanly_and_frees_model() {
    let bin = compiled_binary();
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    let (out, lc) = run_ref(
        &[
            bin.dummy_model.to_str().unwrap(),
            "hello",
            "2",
            prefix.to_str().unwrap(),
        ],
        &[("MOCK_FAIL_CTX", "1")],
    );
    assert_clean_failure(&out, &lc, "llama_init_from_model returns nullptr");
    assert_eq!(lc.get("models_alloc"), Some(&1));
    assert_eq!(lc.get("models_free"), Some(&1));
}

#[test]
fn null_batch_exits_cleanly_and_frees_ctx_and_model() {
    let bin = compiled_binary();
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    let (out, lc) = run_ref(
        &[
            bin.dummy_model.to_str().unwrap(),
            "hello",
            "2",
            prefix.to_str().unwrap(),
        ],
        &[("MOCK_FAIL_BATCH", "1")],
    );
    assert_clean_failure(&out, &lc, "llama_batch_ext_init returns nullptr");
    assert_eq!(lc.get("ctxs_alloc"), Some(&1));
    assert_eq!(lc.get("ctxs_free"), Some(&1));
    assert_eq!(lc.get("models_free"), Some(&1));
}

#[test]
fn tokenize_negative_error_and_overflow_fail_cleanly() {
    let bin = compiled_binary();
    let model = bin.dummy_model.to_str().unwrap();

    for mode in [
        "always_negative",
        "always_negative_big",
        "int_min",
        "zero",
        "positive_on_probe",
    ] {
        let tmp = TempDir::new().unwrap();
        let prefix = tmp.path().join("out");
        let (out, lc) = run_ref(
            &[model, "hello", "2", prefix.to_str().unwrap()],
            &[("MOCK_TOKENIZE_MODE", mode)],
        );
        assert_clean_failure(&out, &lc, &format!("tokenize mode={mode}"));
        assert_eq!(lc.get("models_alloc"), Some(&1));
        assert_eq!(lc.get("models_free"), Some(&1));
    }
}

#[test]
fn tokenize_dynamic_two_pass_resize_succeeds() {
    let bin = compiled_binary();
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    let (out, lc) = run_ref(
        &[
            bin.dummy_model.to_str().unwrap(),
            "hello",
            "1",
            prefix.to_str().unwrap(),
        ],
        &[("MOCK_TOKENIZE_MODE", "two_pass_grow")],
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "two_pass_grow should succeed, got status {:?}, stderr:\n{stderr}",
        out.status
    );
    assert_lifecycle_clean(&lc, "two_pass_grow");

    // Prompt tokens [1, 2, 3, 4], last prompt token 4 predicts (4+1)%8 = 5.
    let toks = fs::read_to_string(format!("{}.tokens", prefix.display())).expect("read .tokens");
    assert_eq!(toks, "1\n2\n3\n4\n5\n");
    let logits = read_f32_le_vec(&PathBuf::from(format!("{}.logits", prefix.display())));
    assert_eq!(logits.len(), 5 * 8);
}

#[test]
fn non_positive_vocab_size_fails_cleanly() {
    let bin = compiled_binary();
    let model = bin.dummy_model.to_str().unwrap();

    for vs in ["0", "-4"] {
        let tmp = TempDir::new().unwrap();
        let prefix = tmp.path().join("out");
        let (out, lc) = run_ref(
            &[model, "hello", "1", prefix.to_str().unwrap()],
            &[("MOCK_VOCAB_SIZE", vs)],
        );
        assert_clean_failure(&out, &lc, &format!("vocab size={vs}"));
    }
}

#[test]
fn unwritable_output_paths_fail_cleanly_and_free_resources() {
    let bin = compiled_binary();
    let model = bin.dummy_model.to_str().unwrap();

    // Case 1: Parent directory does not exist (both .logits and .tokens fopen fail).
    let (out, lc) = run_ref(
        &[model, "hello", "2", "/nonexistent_ref_logits_out_dir/out"],
        &[],
    );
    assert_clean_failure(&out, &lc, "unwritable output directory");
    assert_eq!(lc.get("ctxs_alloc"), Some(&1));
    assert_eq!(lc.get("ctxs_free"), Some(&1));
    assert_eq!(lc.get("models_free"), Some(&1));

    // Case 2: .logits can be opened, but .tokens is a directory so second fopen fails.
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    fs::create_dir(format!("{}.tokens", prefix.display())).expect("create blocking dir");
    let (out, lc) = run_ref(&[model, "hello", "2", prefix.to_str().unwrap()], &[]);
    assert_clean_failure(&out, &lc, "second fopen (.tokens) fails");
    assert_eq!(lc.get("ctxs_free"), Some(&1));
    assert_eq!(lc.get("models_free"), Some(&1));
}

#[test]
fn decode_failure_exits_cleanly_and_frees_all_resources() {
    let bin = compiled_binary();
    let tmp = TempDir::new().unwrap();
    let prefix = tmp.path().join("out");
    let (out, lc) = run_ref(
        &[
            bin.dummy_model.to_str().unwrap(),
            "hello",
            "2",
            prefix.to_str().unwrap(),
        ],
        &[("MOCK_FAIL_DECODE", "1")],
    );
    assert_clean_failure(&out, &lc, "llama_process returns non-zero");
    assert_eq!(lc.get("batches_alloc"), Some(&1));
    assert_eq!(lc.get("batches_free"), Some(&1));
    assert_eq!(lc.get("ctxs_free"), Some(&1));
    assert_eq!(lc.get("models_free"), Some(&1));
}

#[test]
fn null_logits_exits_cleanly_during_dump_and_generation() {
    let bin = compiled_binary();
    let model = bin.dummy_model.to_str().unwrap();

    // Case 1: llama_get_logits_ith returns nullptr on every call (during prompt dump).
    let tmp1 = TempDir::new().unwrap();
    let p1 = tmp1.path().join("out");
    let (out, lc) = run_ref(
        &[model, "hello", "2", p1.to_str().unwrap()],
        &[("MOCK_NULL_LOGITS", "1")],
    );
    assert_clean_failure(&out, &lc, "null logits during dump");

    // Case 2: llama_get_logits_ith(ctx, -1) returns nullptr during greedy generation loop.
    let tmp2 = TempDir::new().unwrap();
    let p2 = tmp2.path().join("out");
    let (out, lc) = run_ref(
        &[model, "hello", "2", p2.to_str().unwrap()],
        &[("MOCK_NULL_LOGITS_LAST", "1")],
    );
    assert_clean_failure(&out, &lc, "null logits on i=-1 during generation");
}

#[test]
fn happy_path_sequential_and_batch_modes_produce_exact_outputs() {
    let bin = compiled_binary();
    let model = bin.dummy_model.to_str().unwrap();

    for batch_arg in ["0", "1"] {
        let tmp = TempDir::new().unwrap();
        let prefix = tmp.path().join("out");
        let (out, lc) = run_ref(
            &[model, "hello world", "2", prefix.to_str().unwrap(), batch_arg],
            &[],
        );

        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "happy path (batch={batch_arg}) failed: {:?}, stderr:\n{stderr}",
            out.status
        );
        assert_lifecycle_clean(&lc, &format!("happy path batch={batch_arg}"));
        assert_eq!(lc.get("batches_alloc"), Some(&1));
        assert_eq!(lc.get("batches_free"), Some(&1));

        // Prompt tokens [1, 2, 3], then generated [4, 5].
        let toks = fs::read_to_string(format!("{}.tokens", prefix.display())).expect("read .tokens");
        assert_eq!(toks, "1\n2\n3\n4\n5\n");

        // 5 rows * 8 vocab floats.
        let logits = read_f32_le_vec(&PathBuf::from(format!("{}.logits", prefix.display())));
        assert_eq!(logits.len(), 5 * 8);

        let expected_tokens = [1i32, 2, 3, 4, 5];
        for (pos, &tok) in expected_tokens.iter().enumerate() {
            let next_tok = ((tok + 1) % 8) as usize;
            for v in 0..8usize {
                let got = logits[pos * 8 + v];
                let want = if v == next_tok {
                    10.0 + pos as f32
                } else {
                    v as f32 * 0.1
                };
                assert!(
                    (got - want).abs() < 1e-6,
                    "batch={batch_arg} pos={pos} v={v}: got {got}, want {want}"
                );
            }
        }
    }

    // Also verify n_gen == 0 dumps only prompt tokens and logits.
    let tmp0 = TempDir::new().unwrap();
    let prefix0 = tmp0.path().join("out0");
    let (out, lc) = run_ref(&[model, "hello", "0", prefix0.to_str().unwrap()], &[]);
    assert!(out.status.success(), "n_gen=0 failed: {:?}", out.status);
    assert_lifecycle_clean(&lc, "n_gen=0");
    let toks = fs::read_to_string(format!("{}.tokens", prefix0.display())).expect("read .tokens");
    assert_eq!(toks, "1\n2\n3\n");
    let logits = read_f32_le_vec(&PathBuf::from(format!("{}.logits", prefix0.display())));
    assert_eq!(logits.len(), 3 * 8);
}
