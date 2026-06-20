// Diagnostic D2: per-layer divergence of the int8 engine vs HF goldens.
// Runs the "sky" prompt, and after each decoder layer compares the engine's
// hidden state against the Hugging Face golden for that layer, printing max
// abs error and cosine similarity. A jump localizes a bug; gradual growth is
// accumulating quant noise.
//
// Pass "--rope" to call RotaryEmbedding::init_freq (correct RoPE); omit it to
// replicate the engine's actual behavior (inv_freq never initialized).
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <string>
#include "../../include/parameters.h"
#include "../../include/modules.h"
#include "../../include/inference_state.h"

static std::unordered_map<std::string, std::vector<float>> load_goldens(const std::string& path) {
    std::unordered_map<std::string, std::vector<float>> g;
    std::ifstream f(path);
    std::string name, line;
    while (std::getline(f, name)) {
        if (name.empty() || name[0] == '#') continue;
        if (!std::getline(f, line)) break;
        std::stringstream ss(line);
        std::vector<float> v;
        float x;
        while (ss >> x) v.push_back(x);
        g[name] = std::move(v);
    }
    return g;
}

static void compare(const std::string& label, const Tensor<float>& h,
                    const std::vector<float>& golden) {
    double max_abs = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    size_t n = std::min((size_t)h.numel, golden.size());
    for (size_t i = 0; i < n; i++) {
        double a = h.data[i], b = golden[i];
        max_abs = std::max(max_abs, std::fabs(a - b));
        dot += a * b; na += a * a; nb += b * b;
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    std::cout << "  " << label
              << "  max_abs=" << max_abs
              << "  cos=" << cos << "\n";
}

int main(int argc, char** argv) {
    std::string model_path = "mistral.mog";
    bool use_rope = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--rope") use_rope = true;
        else model_path = a;
    }

    auto params = std::make_shared<Parameters>();
    params->load_parameters(model_path);

    std::string golden_path = "test/mistral/logits_expected.txt";
    if (!std::ifstream(golden_path).good()) golden_path = "../test/mistral/logits_expected.txt";
    auto g = load_goldens(golden_path);
    if (g.find("layer_stack_sky_L0") == g.end()) {
        std::cerr << "Missing layer_stack goldens. Run: DUMP_LAYER_STACK=1 python scripts/test/mistral/logits.py\n";
        return 1;
    }

    InferenceState infer(params->config);
    if (use_rope) RotaryEmbedding::init_freq(infer, params->config);

    Model<int8_t> model(params);
    std::vector<uint32_t> tokens = params->tokenizer.encode("The color of the sky is");

    infer.pos = 0;
    for (size_t i = 0; i + 1 < tokens.size(); i++) model.forward(infer, tokens[i]);

    model.embedding.forward(infer, tokens.back());
    std::cout << "init_freq called: " << (use_rope ? "yes" : "no")
              << "   (inv_freq[0..2] = " << infer.inv_freq.data[0] << ", "
              << infer.inv_freq.data[1] << ", " << infer.inv_freq.data[2] << ")\n";

    for (size_t layer = 0; layer < model.layers.size(); layer++) {
        model.layers[layer].forward(infer);
        std::string key = "layer_stack_sky_L" + std::to_string(layer);
        if (g.count(key)) compare("L" + std::to_string(layer), infer.hidden_state, g[key]);
    }
    model.norm.forward(infer);
    if (g.count("layer_stack_sky_norm")) compare("norm", infer.hidden_state, g["layer_stack_sky_norm"]);

    return 0;
}
