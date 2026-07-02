#include <string>
#include <vector>

#include "common/dtype.h"
#include "common/tensor.h"
#include "loader/model_load.h"

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

void init_tests(const std::string& path);
std::shared_ptr<ModelLoad> get_model();
const std::string& model_path();

inline Tensor golden_tensor(const float* data, size_t n) {
    return Tensor::from_ptr(const_cast<float*>(data), DType::F32, Shape::from_dims({n}));
}

bool equals(const Tensor& x, const Tensor& y, float atol);
bool expect_tensor(const Tensor& got, const float* golden, size_t n, float atol, const char* name);
