#include "setup/context.h"
#include "model_format.h"
#include "fp16.h"
#include "parameters.h"

#include <cstring>
#include <fstream>
#include <variant>
#include <vector>

namespace {

using TensorVariant = std::variant<Tensor<float>, Tensor<int8_t>, Tensor<fp16_t>>;

std::string resolve_model_path() {
    std::string model_path = "mistral-7B-Q8F16.mog";
    if (!std::ifstream(model_path).good()) {
        model_path = "../mistral-7B-Q8F16.mog";
    }
    return model_path;
}

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

std::vector<size_t> tensor_shape(const TensorVariant& v) {
    if (std::holds_alternative<Tensor<float>>(v)) {
        return std::get<Tensor<float>>(v).shape;
    }
    if (std::holds_alternative<Tensor<fp16_t>>(v)) {
        return std::get<Tensor<fp16_t>>(v).shape;
    }
    return std::get<Tensor<int8_t>>(v).shape;
}

size_t tensor_numel(const TensorVariant& v) {
    if (std::holds_alternative<Tensor<float>>(v)) {
        return std::get<Tensor<float>>(v).get_numel();
    }
    if (std::holds_alternative<Tensor<fp16_t>>(v)) {
        return std::get<Tensor<fp16_t>>(v).get_numel();
    }
    return std::get<Tensor<int8_t>>(v).get_numel();
}

bool is_int8_tensor(const TensorVariant& v) {
    return std::holds_alternative<Tensor<int8_t>>(v);
}

bool is_fp16_tensor(const TensorVariant& v) {
    return std::holds_alternative<Tensor<fp16_t>>(v);
}

float tensor_get(const TensorVariant& v, size_t i) {
    if (std::holds_alternative<Tensor<float>>(v)) {
        return std::get<Tensor<float>>(v).get(i);
    }
    if (std::holds_alternative<Tensor<fp16_t>>(v)) {
        return std::get<Tensor<fp16_t>>(v).get(i);
    }
    return std::get<Tensor<int8_t>>(v).get(i);
}

size_t shape_product(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) {
        n *= d;
    }
    return n;
}

bool check_shape(const std::string& name, const std::vector<size_t>& got,
                 const std::vector<size_t>& want) {
    if (got != want) {
        std::cerr << name << " shape mismatch: got [";
        for (size_t i = 0; i < got.size(); ++i) {
            if (i) std::cerr << ", ";
            std::cerr << got[i];
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

bool layer_tensor_wants_int8(const std::string& name, const std::string& quant) {
    if (!is_q8f16(quant)) {
        return false;
    }
    return name == "mlp.gate_proj.weight" || name == "mlp.up_proj.weight";
}

bool layer_tensor_wants_f16(const std::string& name, const std::string& quant, bool f16_linear) {
    if (!is_q8f16(quant) || !f16_linear) {
        return false;
    }
    return !layer_tensor_wants_int8(name, quant);
}

TensorVariant& resolve_tensor(Parameters& params, const std::string& key, int layer) {
    if (layer == -1) {
        return params.global_weights.at(key);
    }
    return params.layer_weights[static_cast<size_t>(layer)].at(key);
}

struct ShapeSpec {
    const char* name;
    std::vector<size_t> shape;
};

const std::vector<const char*> kLayerTensorNames = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
};

const std::vector<const char*> kGlobalTensorNames = {
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
};

const std::vector<ShapeSpec> kLayerShapeSpecs = {
    {"input_layernorm.weight", {4096}},
    {"post_attention_layernorm.weight", {4096}},
    {"self_attn.q_proj.weight", {4096, 4096}},
    {"self_attn.k_proj.weight", {1024, 4096}},
    {"self_attn.v_proj.weight", {1024, 4096}},
    {"self_attn.o_proj.weight", {4096, 4096}},
    {"mlp.gate_proj.weight", {14336, 4096}},
    {"mlp.up_proj.weight", {14336, 4096}},
    {"mlp.down_proj.weight", {4096, 14336}},
};

const std::vector<ShapeSpec> kGlobalShapeSpecs = {
    {"model.embed_tokens.weight", {32000, 4096}},
    {"model.norm.weight", {4096}},
    {"lm_head.weight", {32000, 4096}},
};

// Mistral-7B-v0.1 checkpoint goldens; validates mmap offsets and dequant, not MOG layout.
struct WeightSpotCheck {
    const char* key;
    int layer;
    float first;
    float last;
};

const std::vector<WeightSpotCheck> kWeightSpotChecks = {
    {"lm_head.weight", -1, -0.002593994140625f, -7.82012939453125e-05f},
    {"model.embed_tokens.weight", -1, -2.1864194925294548e-36f, -0.00250244140625f},
    {"model.norm.weight", -1, 5.34375f, 5.28125f},
    {"input_layernorm.weight", 0, -7.4803829193115234e-06f, 0.006591796875f},
    {"post_attention_layernorm.weight", 0, 0.41796875f, 0.400390625f},
    {"self_attn.q_proj.weight", 0, 5.3882598876953125e-05f, -0.000492095947265625f},
    {"self_attn.k_proj.weight", 0, -1.564621925354004e-06f, 0.000873565673828125f},
    {"self_attn.v_proj.weight", 0, -0.00041961669921875f, 0.00151824951171875f},
    {"self_attn.o_proj.weight", 0, 0.000675201416015625f, -0.00090789794921875f},
    {"mlp.gate_proj.weight", 0, -0.00421142578125f, -0.003753662109375f},
    {"mlp.up_proj.weight", 0, -0.0001773834228515625f, 7.43865966796875e-05f},
    {"mlp.down_proj.weight", 0, -0.0026397705078125f, -0.001953125f},
    {"input_layernorm.weight", 31, 2.53125f, 2.671875f},
    {"post_attention_layernorm.weight", 31, 3.703125f, 3.71875f},
    {"self_attn.q_proj.weight", 31, 0.000484466552734375f, -0.00168609619140625f},
    {"self_attn.k_proj.weight", 31, 0.001739501953125f, 0.0006866455078125f},
    {"self_attn.v_proj.weight", 31, 0.00177001953125f, -0.00049591064453125f},
    {"self_attn.o_proj.weight", 31, -0.0022430419921875f, -0.001678466796875f},
    {"mlp.gate_proj.weight", 31, 0.000270843505859375f, 0.00116729736328125f},
    {"mlp.up_proj.weight", 31, 0.001495361328125f, 0.0013580322265625f},
    {"mlp.down_proj.weight", 31, 0.00180816650390625f, 0.0024566650390625f},
};

} // namespace

int test_mog_header() {
    const std::string path = resolve_model_path();
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
    if (!expect_eq("format version", version, model_format::FORMAT_VERSION)) {
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

int test_mog_config() {
    const Config& c = get_params()->config;

    if (!expect_eq("hidden_size", c.hidden_size, size_t{4096})) return 1;
    if (!expect_eq("intermediate_size", c.intermediate_size, size_t{14336})) return 1;
    if (!expect_eq("n_layers", c.n_layers, size_t{32})) return 1;
    if (!expect_eq("n_heads", c.n_heads, size_t{32})) return 1;
    if (!expect_eq("n_kv_heads", c.n_kv_heads, size_t{8})) return 1;
    if (!expect_eq("head_dim", c.head_dim, size_t{128})) return 1;
    if (!expect_eq("vocab_size", c.vocab_size, size_t{32000})) return 1;
    if (!expect_eq("max_position_embeddings", c.max_position_embeddings, size_t{32768})) return 1;
    if (!expect_eq("sliding_window", c.sliding_window, size_t{4096})) return 1;
    if (!expect_near("rope_theta", c.rope_theta, 10000.0f, 1e-3f)) return 1;
    if (!expect_near("norm_eps", c.norm_eps, 1e-5f, 1e-8f)) return 1;

    if (c.quant != "Q8F16" && c.quant != "f32" && c.quant != "int8") {
        std::cerr << "quant mismatch: got " << c.quant << ", want Q8F16 or f32\n";
        return 1;
    }

    if (c.head_dim != c.hidden_size / c.n_heads) {
        std::cerr << "head_dim not derived from hidden_size / n_heads\n";
        return 1;
    }

    return 0;
}

int test_mog_tensor_inventory() {
    const std::shared_ptr<Parameters> params = get_params();
    const std::string& quant = params->config.quant;
    const bool f16_linear = params->uses_f16_linear_weights();

    if (!expect_eq("global tensor count", params->global_weights.size(), size_t{3})) {
        return 1;
    }
    if (!expect_eq("layer count", params->layer_weights.size(), params->config.n_layers)) {
        return 1;
    }

    for (size_t layer = 0; layer < params->layer_weights.size(); ++layer) {
        const auto& weights = params->layer_weights[layer];
        if (!expect_eq("layer tensor count", weights.size(), size_t{9})) {
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
            const TensorVariant& v = weights.at(spec.name);
            if (!check_shape(spec.name, tensor_shape(v), spec.shape)) {
                std::cerr << "  at layer " << layer << "\n";
                return 1;
            }
            if (tensor_numel(v) != shape_product(spec.shape)) {
                std::cerr << spec.name << " numel mismatch at layer " << layer << "\n";
                return 1;
            }

            const bool want_int8 = layer_tensor_wants_int8(spec.name, quant);
            const bool want_f16 = layer_tensor_wants_f16(spec.name, quant, f16_linear);
            if (is_int8_tensor(v) != want_int8) {
                std::cerr << spec.name << " dtype mismatch at layer " << layer
                          << ": expected " << (want_int8 ? "int8" : (want_f16 ? "f16" : "f32")) << "\n";
                return 1;
            }
            if (want_f16 && !is_fp16_tensor(v)) {
                std::cerr << spec.name << " dtype mismatch at layer " << layer << ": expected f16\n";
                return 1;
            }
            if (!want_int8 && !want_f16 && is_fp16_tensor(v)) {
                std::cerr << spec.name << " dtype mismatch at layer " << layer << ": expected f32\n";
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
        const TensorVariant& v = params->global_weights.at(spec.name);
        if (!check_shape(spec.name, tensor_shape(v), spec.shape)) {
            return 1;
        }
        if (tensor_numel(v) != shape_product(spec.shape)) {
            std::cerr << spec.name << " numel mismatch\n";
            return 1;
        }
        if (is_int8_tensor(v)) {
            std::cerr << spec.name << " should not be int8\n";
            return 1;
        }
        if (is_q8f16(quant) && f16_linear && !is_fp16_tensor(v)) {
            std::cerr << spec.name << " should be f16\n";
            return 1;
        }
        if (quant == "f32" && !std::holds_alternative<Tensor<float>>(v)) {
            std::cerr << spec.name << " should be f32\n";
            return 1;
        }
    }

    if (f16_linear) {
        (void)params->get_tensor<fp16_t>(0, "self_attn.q_proj.weight");
        (void)params->get_tensor<fp16_t>(-1, "lm_head.weight");
    } else {
        (void)params->get_tensor<float>(0, "self_attn.q_proj.weight");
        (void)params->get_tensor<float>(-1, "lm_head.weight");
    }

    return 0;
}

int test_mog_weight_spotcheck() {
    const std::shared_ptr<Parameters> params = get_params();

    for (const auto& entry : kWeightSpotChecks) {
        TensorVariant& v = resolve_tensor(*params, entry.key, entry.layer);
        const size_t n = tensor_numel(v);

        const float first_val = tensor_get(v, 0);
        const float last_val = tensor_get(v, n - 1);
        const float atol = is_fp16_tensor(v) ? 1e-3f : 0.0f;

        if (atol > 0.0f) {
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
            continue;
        }

        if (!equals(first_val, entry.first)) {
            std::cerr << entry.key << " first val mismatch " << first_val
                      << " vs " << entry.first;
            if (entry.layer >= 0) {
                std::cerr << " (layer " << entry.layer << ")";
            }
            std::cerr << "\n";
            return 1;
        }
        if (!equals(last_val, entry.last)) {
            std::cerr << entry.key << " last val mismatch " << last_val
                      << " vs " << entry.last;
            if (entry.layer >= 0) {
                std::cerr << " (layer " << entry.layer << ")";
            }
            std::cerr << "\n";
            return 1;
        }
    }

    return 0;
}

static RegisterTest mog_header_q8f16("mog header", "Q8F16", &test_mog_header);
static RegisterTest mog_header_int8("mog header", "int8", &test_mog_header);
static RegisterTest mog_config_f32("mog config", "f32", &test_mog_config);
static RegisterTest mog_config_q8f16("mog config", "Q8F16", &test_mog_config);
static RegisterTest mog_config_int8("mog config", "int8", &test_mog_config);
static RegisterTest mog_inventory_f32("mog tensor inventory", "f32", &test_mog_tensor_inventory);
static RegisterTest mog_inventory_q8f16("mog tensor inventory", "Q8F16", &test_mog_tensor_inventory);
static RegisterTest mog_inventory_int8("mog tensor inventory", "int8", &test_mog_tensor_inventory);
static RegisterTest mog_spotcheck_f32("mog weight spotcheck", "f32", &test_mog_weight_spotcheck);
static RegisterTest mog_spotcheck_q8f16("mog weight spotcheck", "Q8F16", &test_mog_weight_spotcheck);
static RegisterTest mog_spotcheck_int8("mog weight spotcheck", "int8", &test_mog_weight_spotcheck);
