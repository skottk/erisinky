// Reference oracle: run llama.cpp on a prompt one token at a time (greedy), dump token ids + full logits.
// usage: ref_logits model.gguf "prompt" n_gen out_prefix [batch=0|1]
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool parse_int_arg(const char * s, int & out) {
    if (!s || s[0] == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

struct Resources {
    bool backend_inited = false;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch_ext * batch = nullptr;
    FILE * fl = nullptr;
    FILE * ft = nullptr;

    Resources() = default;
    Resources(const Resources &) = delete;
    Resources & operator=(const Resources &) = delete;

    ~Resources() {
        if (ft) std::fclose(ft);
        if (fl) std::fclose(fl);
        if (batch) llama_batch_ext_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        if (backend_inited) llama_backend_free();
    }
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s model prompt n_gen out_prefix [batch]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    std::string prompt = argv[2];
    int n_gen = 0;
    if (!parse_int_arg(argv[3], n_gen) || n_gen < 0) {
        std::fprintf(stderr, "error: invalid n_gen '%s' (expected non-negative integer)\n", argv[3]);
        return 1;
    }
    std::string out = argv[4];
    if (out.empty()) {
        std::fprintf(stderr, "error: out_prefix must not be empty\n");
        return 1;
    }

    bool batch_mode = false;
    if (argc > 5) {
        int batch_flag = 0;
        if (!parse_int_arg(argv[5], batch_flag)) {
            std::fprintf(stderr, "error: invalid batch flag '%s' (expected 0 or 1)\n", argv[5]);
            return 1;
        }
        batch_mode = (batch_flag != 0);
    }

    if (prompt.size() > static_cast<size_t>(INT32_MAX)) {
        std::fprintf(stderr, "error: prompt exceeds maximum supported length\n");
        return 1;
    }

    Resources res;
    llama_backend_init();
    res.backend_inited = true;

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    res.model = llama_model_load_from_file(model_path, mp);
    if (!res.model) {
        std::fprintf(stderr, "error: failed to load model from '%s'\n", model_path);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(res.model);
    if (!vocab) {
        std::fprintf(stderr, "error: llama_model_get_vocab returned NULL\n");
        return 1;
    }

    const int32_t nv = llama_vocab_n_tokens(vocab);
    if (nv <= 0) {
        std::fprintf(stderr, "error: invalid vocab size %d\n", nv);
        return 1;
    }

    const int32_t prompt_len = static_cast<int32_t>(prompt.size());
    const int32_t n_probe = llama_tokenize(vocab, prompt.c_str(), prompt_len, nullptr, 0, true, true);
    if (n_probe == INT32_MIN || n_probe >= 0) {
        std::fprintf(stderr, "error: llama_tokenize probe failed (returned %d)\n", n_probe);
        return 1;
    }

    int32_t cap = -n_probe;
    std::vector<llama_token> toks(static_cast<size_t>(cap));
    int32_t n = llama_tokenize(vocab, prompt.c_str(), prompt_len, toks.data(), cap, true, true);
    if (n < 0 && n != INT32_MIN && -n > cap) {
        cap = -n;
        toks.resize(static_cast<size_t>(cap));
        n = llama_tokenize(vocab, prompt.c_str(), prompt_len, toks.data(), cap, true, true);
    }
    if (n <= 0 || n > cap) {
        std::fprintf(stderr, "error: llama_tokenize failed (returned %d, capacity %d)\n", n, cap);
        return 1;
    }
    toks.resize(static_cast<size_t>(n));

    if (n_gen > INT32_MAX - n) {
        std::fprintf(stderr, "error: total token count overflows int32\n");
        return 1;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    if (std::getenv("REF_NO_FA")) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    res.ctx = llama_init_from_model(res.model, cp);
    if (!res.ctx) {
        std::fprintf(stderr, "error: llama_init_from_model failed\n");
        return 1;
    }

    const std::string logits_path = out + ".logits";
    const std::string tokens_path = out + ".tokens";
    res.fl = std::fopen(logits_path.c_str(), "wb");
    if (!res.fl) {
        std::fprintf(stderr, "error: failed to open '%s': %s\n", logits_path.c_str(), std::strerror(errno));
        return 1;
    }
    res.ft = std::fopen(tokens_path.c_str(), "w");
    if (!res.ft) {
        std::fprintf(stderr, "error: failed to open '%s': %s\n", tokens_path.c_str(), std::strerror(errno));
        return 1;
    }

    res.batch = llama_batch_ext_init(res.ctx);
    if (!res.batch) {
        std::fprintf(stderr, "error: llama_batch_ext_init failed\n");
        return 1;
    }

    int32_t pos = 0;
    auto run = [&](const llama_token * t, int32_t cnt) -> bool {
        llama_batch_ext_clear(res.batch);
        for (int32_t i = 0; i < cnt; ++i) {
            const int32_t idx = llama_batch_ext_add_token(res.batch, 0, t[i]);
            if (idx < 0) {
                std::fprintf(stderr, "error: llama_batch_ext_add_token failed (%d)\n", idx);
                return false;
            }
            const llama_pos p = pos + i;
            llama_batch_ext_set_pos(res.batch, idx, &p);
            llama_batch_ext_set_output_logits(res.batch, idx, true);
        }
        if (llama_process(res.ctx, LLAMA_PROCESS_TYPE_DECODE, res.batch) != 0) {
            std::fprintf(stderr, "error: decode failed\n");
            return false;
        }
        for (int32_t i = 0; i < cnt; ++i) {
            const float * logits = llama_get_logits_ith(res.ctx, i);
            if (!logits) {
                std::fprintf(stderr, "error: llama_get_logits_ith(%d) returned NULL\n", i);
                return false;
            }
            if (std::fwrite(logits, sizeof(float), static_cast<size_t>(nv), res.fl) != static_cast<size_t>(nv)) {
                std::fprintf(stderr, "error: failed writing logits to '%s'\n", logits_path.c_str());
                return false;
            }
            if (std::fprintf(res.ft, "%d\n", t[i]) < 0) {
                std::fprintf(stderr, "error: failed writing token to '%s'\n", tokens_path.c_str());
                return false;
            }
        }
        pos += cnt;
        return true;
    };

    if (batch_mode) {
        for (int32_t i = 0; i < n; i += 512) {
            if (!run(&toks[static_cast<size_t>(i)], std::min<int32_t>(512, n - i))) {
                return 1;
            }
        }
    } else {
        for (int32_t i = 0; i < n; ++i) {
            if (!run(&toks[static_cast<size_t>(i)], 1)) {
                return 1;
            }
        }
    }

    for (int g = 0; g < n_gen; ++g) {
        const float * lg = llama_get_logits_ith(res.ctx, -1);
        if (!lg) {
            std::fprintf(stderr, "error: llama_get_logits_ith(-1) returned NULL during generation\n");
            return 1;
        }
        int32_t best = 0;
        for (int32_t v = 1; v < nv; ++v) {
            if (lg[v] > lg[best]) {
                best = v;
            }
        }
        llama_token cur = best;
        char buf[256];
        const int32_t k = llama_token_to_piece(vocab, cur, buf, static_cast<int32_t>(sizeof(buf)), 0, true);
        if (k > 0) {
            std::fwrite(buf, 1, std::min(static_cast<size_t>(k), sizeof(buf)), stderr);
        }
        if (!run(&cur, 1)) {
            return 1;
        }
    }

    std::fprintf(stderr, "\n[ref] prompt tokens %d, generated %d, vocab %d\n", n, n_gen, nv);

    bool close_ok = true;
    if (std::fclose(res.fl) != 0) {
        std::fprintf(stderr, "error: failed to close '%s': %s\n", logits_path.c_str(), std::strerror(errno));
        close_ok = false;
    }
    res.fl = nullptr;

    if (std::fclose(res.ft) != 0) {
        std::fprintf(stderr, "error: failed to close '%s': %s\n", tokens_path.c_str(), std::strerror(errno));
        close_ok = false;
    }
    res.ft = nullptr;

    return close_ok ? 0 : 1;
}
