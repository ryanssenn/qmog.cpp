#pragma once
#include "tensor.h"
#include "binary_reader.h"

struct Tokenizer {
    // Vocab lookup tables
    // Store the vocab to go back and forth Token <-> ID
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, uint32_t> token_to_id;

    // BPE Merge table
    // Maps a merged token pair to its merge rank (Vocab token ID)
    // Each key packs two 32-bit token IDs (left and right) into one 64-bit integer
    std::unordered_map<uint64_t, uint32_t> merge_to_rank;

    // Maps a packed (left,right) token pair to its merged token ID to avoid recomputing merges during encoding
    std::unordered_map<uint64_t, uint32_t> merge_to_id;

    std::vector<uint32_t> get_id(const std::string& token) const;

    uint64_t pack(uint32_t left, uint32_t right) const;
    uint64_t get_lowest_pair(std::vector<uint32_t>& tokens) const;
    std::vector<uint32_t> merge(std::vector<uint32_t>& tokens, uint32_t left, uint32_t right, uint32_t merged) const;

    void load(BinaryReader& reader);

    std::string pre_tokenize_mistral(const std::string& text) const;
    std::vector<uint32_t> encode(const std::string& text) const;

    std::string decode_mistral(std::string s) const;
    std::string decode(const std::vector<uint32_t>& token) const;
};
