#include <string>
#include <iostream>
#include "../../../include/parameters.h"
#include "../../../include/kernels.h"
#include "../../../include/modules.h"

struct TestCase {
    std::string name;
    std::string quant;
    int (*func)();
};

extern std::vector<TestCase> tests;

struct RegisterTest {
    RegisterTest(std::string name, std::string quant, int (*func)()){
        tests.push_back({name, quant, func});
    }
};

std::shared_ptr<Parameters> get_params();
inline InferenceState infer(get_params()->config);
inline Arena arena(4*1024*1024); // 4 MB

inline std::unordered_map<std::string, Tensor<float>> expected;
void load_expected_values();

struct TopK {
    std::vector<uint32_t> ids;
    std::vector<float> vals;
};

TopK get_topk(const Tensor<float>& logits, size_t k);
bool has_logits_golden();
bool compare_topk(const TopK& got, const std::string& ids_key, const std::string& vals_key, float atol);
bool compare_topk_greedy_argmax(const TopK& got, const std::string& ids_key);

bool equals(float x, float y);
bool equals(const Tensor<float>& x, const Tensor<float>& y);
bool equals(const Tensor<float>& x, const Tensor<float>& y, float atol);