#include "setup/context.h"
#include "common/dtype.h"
#include "common/shape.h"
#include "loader/model_load.h"

#include <cstring>
#include <fstream>
#include <vector>

namespace {

template<typename T>
bool expect_eq(const char* field, T got, T want) {
    if (got != want) {
        std::cerr << field << " mismatch: got " << got << ", want " << want << "\n";
        return false;
    }
    return true;
}

bool expect_near(const char* field, float got, float want, float atol) {
    if (std::fabs(got - want) >= atol) {
        std::cerr << field << " mismatch: got " << got << ", want " << want << "\n";
        return false;
    }
    return true;
}

bool check_shape(const std::string& name, const Tensor& t,
                 const std::vector<size_t>& want) {
    if (!shape_equals(t, want)) {
        std::cerr << name << " shape mismatch: got [";
        for (uint8_t i = 0; i < t.ndim; ++i) {
            if (i) std::cerr << ", ";
            std::cerr << t.shape[i];
        }
        std::cerr << "], want [";
        for (size_t i = 0; i < want.size(); ++i) {
            if (i) std::cerr << ", ";
            std::cerr << want[i];
        }
        std::cerr << "]\n";
        return false;
    }
    return true;
}

Tensor& resolve_tensor(ModelLoad& params, const std::string& key, int layer) {
    if (layer == -1) {
        return params.global_weights.at(key);
    }
    return params.layer_weights[static_cast<size_t>(layer)].at(key);
}

#include "expected/model_load.inl"

} // namespace

// Validates the raw file prefix: MOG magic, format version 2, and a sane header_size
// that fits within the file. Catches truncated or non-MOG inputs before load.
int test_header() {
    const std::string path = model_path();
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        std::cerr << "model file not found: " << path << "\n";
        return 1;
    }

    f.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    if (file_size < model_format::FILE_PREFIX_SIZE) {
        std::cerr << "file too small for MOG prefix: " << file_size << " bytes\n";
        return 1;
    }

    char prefix[model_format::FILE_PREFIX_SIZE];
    f.read(prefix, sizeof(prefix));
    if (!f) {
        std::cerr << "failed to read MOG prefix\n";
        return 1;
    }

    if (std::memcmp(prefix, model_format::MAGIC, 4) != 0) {
        std::cerr << "magic mismatch: expected MOG\\0\n";
        return 1;
    }

    uint32_t version;
    std::memcpy(&version, prefix + 4, sizeof(version));
    if (version != model_format::FORMAT_VERSION) {
        std::cerr << "format version mismatch: got " << version
                  << ", want " << model_format::FORMAT_VERSION << "\n";
        return 1;
    }

    uint64_t header_size;
    std::memcpy(&header_size, prefix + 8, sizeof(header_size));
    if (header_size == 0) {
        std::cerr << "header_size must be non-zero\n";
        return 1;
    }
    if (header_size >= file_size) {
        std::cerr << "header_size out of bounds: " << header_size
                  << " >= file_size " << file_size << "\n";
        return 1;
    }

    return 0;
}

// Checks that ModelLoad parsed the config KV for Qwen3-0.6B: architecture, dims,
// rope/norm settings, f16 quant, tied embeddings, and special token ids.
int test_config() {
    const Config& c = get_model()->config;

    if (c.architecture != "qwen3") {
        std::cerr << "architecture mismatch: got " << c.architecture << ", want qwen3\n";
        return 1;
    }
    if (!expect_eq("hidden_size", c.hidden_size, size_t{1024})) return 1;
    if (!expect_eq("intermediate_size", c.intermediate_size, size_t{3072})) return 1;
    if (!expect_eq("n_layers", c.n_layers, size_t{28})) return 1;
    if (!expect_eq("n_heads", c.n_heads, size_t{16})) return 1;
    if (!expect_eq("n_kv_heads", c.n_kv_heads, size_t{8})) return 1;
    if (!expect_eq("head_dim", c.head_dim, size_t{128})) return 1;
    if (!expect_eq("vocab_size", c.vocab_size, size_t{151936})) return 1;
    if (!expect_eq("sliding_window", c.sliding_window, size_t{0})) return 1;
    if (!expect_eq("max_position_embeddings", c.max_position_embeddings, size_t{40960})) return 1;
    if (!expect_near("rope_theta", c.rope_theta, 1000000.0f, 1.0f)) return 1;
    if (!expect_near("norm_eps", c.norm_eps, 1e-6f, 1e-9f)) return 1;

    if (c.quant != "f16") {
        std::cerr << "quant mismatch: got " << c.quant << ", want f16\n";
        return 1;
    }
    if (!c.tie_word_embeddings) {
        std::cerr << "tie_word_embeddings should be true\n";
        return 1;
    }
    if (!expect_eq("bos_token_id", c.bos_token_id, uint32_t{151643})) return 1;
    if (!expect_eq("eos_token_id", c.eos_token_id, uint32_t{151645})) return 1;

    return 0;
}

// MOG v2 stores a pre-tokenize regex that must match QwenPreTokenizer's hardcoded pattern.
int test_tokenizer_metadata() {
    if (std::string(QwenPreTokenizer::pattern).find("(?i:'s|'t|'re|'ve|'m|'ll|'d)") == std::string::npos) {
        std::cerr << "unexpected Qwen pre_tokenize regex\n";
        return 1;
    }

    return 0;
}

// Verifies the tensor table: 28 layers x 11 weights (incl. q_norm/k_norm), two
// global f16 tensors (embed + norm, no lm_head when tied), and expected shapes/dtypes.
int test_tensor_inventory() {
    const std::shared_ptr<ModelLoad> params = get_model();

    if (!expect_eq("global tensor count", params->global_weights.size(), size_t{2})) {
        return 1;
    }
    if (!expect_eq("layer count", params->layer_weights.size(), params->config.n_layers)) {
        return 1;
    }

    for (size_t layer = 0; layer < params->layer_weights.size(); ++layer) {
        const auto& weights = params->layer_weights[layer];
        if (!expect_eq("layer tensor count", weights.size(), size_t{11})) {
            std::cerr << "  at layer " << layer << "\n";
            return 1;
        }

        for (const char* name : kLayerTensorNames) {
            if (weights.find(name) == weights.end()) {
                std::cerr << "missing layer tensor: " << name << " at layer " << layer << "\n";
                return 1;
            }
        }

        for (const auto& spec : kLayerShapeSpecs) {
            const Tensor& t = weights.at(spec.first);
            if (!check_shape(spec.first, t, spec.second)) {
                std::cerr << "  at layer " << layer << "\n";
                return 1;
            }
            if (t.dtype != DType::F16) {
                std::cerr << spec.first << " should be f16 at layer " << layer << "\n";
                return 1;
            }
        }
    }

    for (const char* name : kGlobalTensorNames) {
        if (params->global_weights.find(name) == params->global_weights.end()) {
            std::cerr << "missing global tensor: " << name << "\n";
            return 1;
        }
    }

    for (const auto& spec : kGlobalShapeSpecs) {
        const Tensor& t = params->global_weights.at(spec.first);
        if (!check_shape(spec.first, t, spec.second)) {
            return 1;
        }
        if (t.dtype != DType::F16) {
            std::cerr << spec.first << " should be f16\n";
            return 1;
        }
    }

    if (params->global_weights.find("lm_head.weight") != params->global_weights.end()) {
        std::cerr << "lm_head.weight should be absent when tie_word_embeddings is true\n";
        return 1;
    }

    (void)params->get_tensor(0, "self_attn.q_proj.weight");
    (void)params->get_tensor(-1, "model.embed_tokens.weight");

    return 0;
}

// Spot-checks first/last f16 values at known payload offsets on a few tensors
// (layers 0 and 27). Confirms mmap offsets and f16 decode, not just header layout.
int test_weight_spotcheck() {
    const std::shared_ptr<ModelLoad> params = get_model();
    constexpr float atol = 1e-3f;

    for (const auto& entry : kWeightSpotChecks) {
        Tensor& t = resolve_tensor(*params, entry.key, entry.layer);
        const size_t n = t.numel;

        const float first_val = static_cast<float>(t.f16()[0]);
        const float last_val = static_cast<float>(t.f16()[n - 1]);

        if (!expect_near("first val", first_val, entry.first, atol)) {
            std::cerr << entry.key;
            if (entry.layer >= 0) std::cerr << " (layer " << entry.layer << ")";
            std::cerr << "\n";
            return 1;
        }
        if (!expect_near("last val", last_val, entry.last, atol)) {
            std::cerr << entry.key;
            if (entry.layer >= 0) std::cerr << " (layer " << entry.layer << ")";
            std::cerr << "\n";
            return 1;
        }
    }

    return 0;
}

static RegisterTest header_f16("header", "f16", &test_header);
static RegisterTest config_f16("config", "f16", &test_config);
static RegisterTest tokenizer_metadata_f16("tokenizer metadata", "f16", &test_tokenizer_metadata);
static RegisterTest inventory_f16("tensor inventory", "f16", &test_tensor_inventory);
static RegisterTest spotcheck_f16("weight spotcheck", "f16", &test_weight_spotcheck);
