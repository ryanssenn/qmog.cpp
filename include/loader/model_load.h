#pragma once

#include <cstdint>
#include <unordered_map>
#include <string>
#include "common/tensor.h"
#include "tokenizer/qwen_tokenizer.h"

namespace model_format {

constexpr char MAGIC[4] = {'M', 'O', 'G', '\0'};
constexpr uint32_t FORMAT_VERSION = 2;

enum class ConfigValueType : uint8_t {
    STRING = 0,
    UINT32 = 1,
    FLOAT32 = 2,
};

constexpr size_t FILE_PREFIX_SIZE = 16; // magic + version + header_size

} // namespace model_format

struct Config {
    std::string architecture;
    size_t hidden_size;
    size_t intermediate_size;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    size_t vocab_size;
    size_t head_dim;
    size_t sliding_window;
    size_t max_position_embeddings;
    float rope_theta;
    float norm_eps;
    std::string quant;
    bool tie_word_embeddings = false;
    uint32_t bos_token_id = 0;
    uint32_t eos_token_id = 0;
};

class BinaryReader;

struct ModelLoad {
    Config config;
    QwenTokenizer tokenizer;

    // One storage wrapping the whole mmap'd file; weights are views into it.
    std::shared_ptr<Storage> storage;

    std::unordered_map<std::string, Tensor> global_weights;
    std::vector<std::unordered_map<std::string, Tensor>> layer_weights;

    static void* map_file(int fd, size_t& size);
    void load_tensor(std::unordered_map<std::string, Tensor>& m, const std::string& key, uint8_t dtype, const std::vector<size_t>& shape, uint64_t offset);

    void load_config(BinaryReader& reader);
    void load_weights(size_t payload_base, BinaryReader& reader);
    void load(const std::string& path, Device device = Device::CPU);

    Tensor& get_tensor(int layer, const std::string& name);
};
