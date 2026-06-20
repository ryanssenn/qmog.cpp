#include <unordered_map>
#include <string>
#include "tensor.h"
#include "tokenizer.h"
#include <variant>

#pragma once

struct Config {
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
};

class BinaryReader;

struct Parameters {
    Config config;
    Tokenizer tokenizer;

    // Global weights (For Mistral, this is embeddings, final layernorm, lm_head)
    std::unordered_map<std::string, std::variant<Tensor<float>, Tensor<int8_t>>> global_weights;

    // Layer specific weights
    std::vector<std::unordered_map<std::string, std::variant<Tensor<float>, Tensor<int8_t>>>> layer_weights;

    static void* map_file(int fd, size_t& size);
    void load_tensor(std::unordered_map<std::string, std::variant<Tensor<float>, Tensor<int8_t>>>& m, char* p, const std::string& key, uint8_t dtype, const std::vector<size_t>& shape, uint64_t offset, uint64_t scale_offset, uint32_t scale_size);

    void load_config(BinaryReader& reader);
    void load_weights(char* p, BinaryReader& reader);
    void load_parameters(const std::string& path);

    template<typename T>
    Tensor<T> get_tensor(int layer, const std::string& name);
};
