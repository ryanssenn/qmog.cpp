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

    std::string model_path = "mistral.mog";
    if (!std::ifstream(model_path).good()) {
        model_path = "../mistral.mog";
    }

    params->load_parameters(model_path);
    init = true;

    return params;
}

void load_expected_values(){
    auto load_file = [](const std::string& primary, const std::string& fallback) {
        std::string path = primary;
        if (!std::ifstream(path).good()) {
            path = fallback;
        }
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

    load_file("test/mistral/expected.txt", "../test/mistral/expected.txt");
    load_file("test/mistral/logits_expected.txt", "../test/mistral/logits_expected.txt");
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

bool compare_topk(const TopK& got, const std::string& ids_key, const std::string& vals_key, float atol){
    if (expected.find(ids_key) == expected.end() || expected.find(vals_key) == expected.end()){
        std::cout << "Missing golden keys: " << ids_key << " / " << vals_key << std::endl;
        return false;
    }

    const Tensor<float>& exp_ids = expected.at(ids_key);
    const Tensor<float>& exp_vals = expected.at(vals_key);

    if (got.ids.size() != exp_ids.numel || got.vals.size() != exp_vals.numel){
        std::cout << "TopK size mismatch for " << ids_key << std::endl;
        return false;
    }

    for (size_t i = 0; i < got.ids.size(); i++){
        uint32_t exp_id = static_cast<uint32_t>(exp_ids.data[i]);
        if (got.ids[i] != exp_id){
            std::cout << "TopK id mismatch at rank " << i
                      << " for " << ids_key
                      << ": expected token " << exp_id
                      << ", got " << got.ids[i] << std::endl;
            return false;
        }
        if (std::fabs(got.vals[i] - exp_vals.data[i]) >= atol){
            std::cout << "TopK value mismatch at rank " << i
                      << " for " << vals_key
                      << ": expected " << exp_vals.data[i]
                      << ", got " << got.vals[i] << std::endl;
            return false;
        }
    }

    return true;
}

bool compare_topk_greedy_argmax(const TopK& got, const std::string& ids_key){
    if (expected.find(ids_key) == expected.end()){
        std::cout << "Missing golden key: " << ids_key << std::endl;
        return false;
    }

    uint32_t exp_id = static_cast<uint32_t>(expected.at(ids_key).data[0]);
    if (got.ids.empty() || got.ids[0] != exp_id){
        std::cout << "Greedy argmax mismatch for " << ids_key
                  << ": expected token " << exp_id;
        if (!got.ids.empty()){
            std::cout << ", got " << got.ids[0];
        }
        std::cout << std::endl;
        return false;
    }

    return true;
}

static float equals_atol(){
    // f32 tests compare engine f32 vs HF f32; int8 tests compare int8 vs HF f32 goldens.
    if (get_params()->config.quant == "int8") {
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
