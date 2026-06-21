#include "context.h"
#include <fstream>
#include <sstream>
#include <limits>

std::vector<TestCase> tests;

std::shared_ptr<Parameters> get_params(){
    static bool init = false;

    static std::shared_ptr<Parameters> params = std::make_shared<Parameters>();

    if (init){
        return params;
    }

    std::string model_path = "mistral-7B-Q8F16.mog";
    if (!std::ifstream(model_path).good()) {
        model_path = "../mistral-7B-Q8F16.mog";
    }

    params->load_parameters(model_path);
    init = true;

    return params;
}

void load_expected_values(){
    auto load_file = [](const std::string& path) {
        if (!std::ifstream(path).good()) {
            return;
        }

        std::ifstream f(path);
        std::string line, name;

        while (std::getline(f, name)){
            if (name == "" || name[0] == '#') continue;
            std::getline(f, line);
            std::stringstream ss(line);
            float x;
            size_t i = 0;
            std::vector<float> data;
            while (ss >> x){
                data.push_back(x);
                i++;
            }
            expected.emplace(name, Tensor<float>(arena, data, {i}));
        }
    };

    load_file("test/mistral/expected.txt");
    load_file("test/mistral/logits_expected.txt");
}

bool has_logits_golden(){
    return expected.find("logits_sky_step0_top10_ids") != expected.end();
}

TopK get_topk(const Tensor<float>& logits, size_t k){
    TopK result;
    std::vector<bool> used(logits.numel, false);

    for (size_t n = 0; n < k; n++){
        float best = -std::numeric_limits<float>::infinity();
        size_t best_i = 0;

        for (size_t i = 0; i < logits.numel; i++){
            if (!used[i] && logits.data[i] > best){
                best = logits.data[i];
                best_i = i;
            }
        }

        used[best_i] = true;
        result.ids.push_back(static_cast<uint32_t>(best_i));
        result.vals.push_back(best);
    }

    return result;
}

static float equals_atol(){
    if (get_params()->config.quant == "f32") {
        return 5e-2f;
    }
    if (is_q8f16(get_params()->config.quant)) {
        return 1.1e-1f; // worst pre-L31 layer ~0.106 (sky L29)
    }
    return 5e-2f;
}

bool equals(float x, float y){
    return std::fabs(x - y) < equals_atol();
}

bool equals(const Tensor<float>& x, const Tensor<float>& y){
    return equals(x, y, equals_atol());
}

bool equals(const Tensor<float>& x, const Tensor<float>& y, float atol){
    if (x.numel != y.numel){
        return false;
    }

    for (int i=0; i<x.numel; i++){
        if (std::fabs(x.data[i] - y.data[i]) >= atol){
            std::cout << "Mismatch at pos " << i
                      << ": expected " << y.data[i]
                      << ", got " << x.data[i] << std::endl;
            return false;
        }
    }

    return true;
}
