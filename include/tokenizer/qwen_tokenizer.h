#pragma once

#include "tokenizer/bpe.h"
#include "tokenizer/qwen_pre_tokenize.h"

#include <cstdint>
#include <string>
#include <vector>

struct QwenTokenizer : Bpe {
    QwenPreTokenizer pre_tokenizer;

    mutable bool special_tokens_initialized = false;
    mutable std::vector<std::pair<std::string, uint32_t>> special_tokens;

    void load(BinaryReader& reader);

    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;

private:
    const std::vector<std::pair<std::string, uint32_t>>& special_tokens_list() const;
    std::vector<uint32_t> bpe(const std::string& text) const;
};
