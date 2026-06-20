#include "../../include/tokenizer.h"
#include <iostream>
#include <algorithm>


uint64_t Tokenizer::pack(uint32_t left, uint32_t right) const{
    // Pack left
    uint64_t packed = left;
    // Shift to left and zero out 32 right-most bits
    packed = packed << 32;
    // Insert right on the right with OR
    packed = packed | right;

    return packed;
}

std::vector<uint32_t> Tokenizer::get_id(const std::string& token) const{
    auto it = token_to_id.find(token);

    // This is the byte fallback
    if (it == token_to_id.end()){
        std::vector<uint32_t> res;

        for (int i=0; i<token.size(); i++) {
            unsigned char b = static_cast<unsigned char>(token[i]);
            char buf[8];
            snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto bit = token_to_id.find(buf);
            res.push_back(bit->second);
        }

        return res;
    }
    return {it->second};
}


void Tokenizer::load(BinaryReader& reader){
    uint32_t vocab_count = reader.read_u32();
    uint32_t max_id = 0;

    std::vector<std::pair<std::string, uint32_t>> vocab_entries;
    vocab_entries.reserve(vocab_count);

    for (uint32_t i = 0; i < vocab_count; i++){
        std::string token = reader.read_string();
        uint32_t id = reader.read_u32();
        vocab_entries.emplace_back(token, id);
        max_id = std::max(max_id, id);
    }

    id_to_token.assign(max_id + 1, "");
    token_to_id.clear();

    for (auto& [token, id] : vocab_entries){
        id_to_token[id] = token;
        token_to_id[token] = id;
    }

    uint32_t merge_count = reader.read_u32();
    for (uint32_t i = 0; i < merge_count; i++){
        std::string merge = reader.read_string();

        size_t s = merge.find(" ");
        std::string token1 = merge.substr(0, s);
        std::string token2 = merge.substr(s + 1);

        uint32_t id1 = token_to_id[token1];
        uint32_t id2 = token_to_id[token2];

        uint64_t packed = pack(id1, id2);

        merge_to_rank[packed] = i;
        merge_to_id[packed] = token_to_id[token1 + token2];
    }
}

// Get the merge pair with lowest rank in a list of tokens
// Returns UINT64_MAX when no merge is possible
uint64_t Tokenizer::get_lowest_pair(std::vector<uint32_t>& tokens) const{
    uint32_t lowest_rank = UINT32_MAX;
    uint64_t result = UINT64_MAX;

    for (int i=0;i<tokens.size()-1;i++){
        uint64_t packed = pack(tokens[i], tokens[i+1]);

        auto it = merge_to_rank.find(packed);
        if (it != merge_to_rank.end() && it->second < lowest_rank){
            lowest_rank = it->second;
            result = packed;
        }
    }

    return result;
}

// Merge all occurrences of the (left, right) token pair into merged token
std::vector<uint32_t> Tokenizer::merge(std::vector<uint32_t>& tokens, uint32_t left, uint32_t right, uint32_t merged) const{
    std::vector<uint32_t> merged_tokens;

    int i = 0;
    while (i < tokens.size()){
        if (tokens[i] == left && i + 1 < tokens.size() && tokens[i+1] == right){
            merged_tokens.push_back(merged);
            i += 2;
            continue;
        }

        merged_tokens.push_back(tokens[i]);
        i++;
    }

    return merged_tokens;
}

// For mistral, use Metaspace pre-tokenization, replace spaces with '▁' and prepend
std::string Tokenizer::pre_tokenize_mistral(const std::string& text) const{
    std::string out = "▁";

    for (auto c : text){
        if (c == ' '){
            out += "▁";
            continue;
        }
        out += c;
    }
    return out;
}

// Runs the BPE merge based tokenization
std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    // If want to support other tokenizers config we would support other stages that are not needed in Mistral

    // Normalization (not implemented for Mistral)

    // Pre-tokenization
    std::string text2 = pre_tokenize_mistral(text);

    // Now run the BPE Algorithm

    // Transform into a list of token IDs
    std::vector<uint32_t> tokens;

    size_t i = 0;
    while (i < text2.size()) {
        // If we have a valid multiple byte UTF-8 encoding, we want to merge them into one string
        // I should probably use a library to handle raw UTF-8 bytes but couldnt find
        // The first byte leading 1s indicate the number of byte to read

        unsigned char b0 = static_cast<unsigned char>(text2[i]);
        size_t len = 1;

        if      ((b0 & 0x80) == 0x00) len = 1;          // 0xxxxxxx
        else if ((b0 & 0xE0) == 0xC0) len = 2;          // 110xxxxx
        else if ((b0 & 0xF0) == 0xE0) len = 3;          // 1110xxxx
        else if ((b0 & 0xF8) == 0xF0) len = 4;          // 11110xxx

        std::string s = text2.substr(i, len);

        std::vector<uint32_t> ids = get_id(s);
        tokens.insert(tokens.end(), ids.begin(), ids.end());
        i += len;
    }

    uint64_t packed = get_lowest_pair(tokens);

    // While we still find a valid pair to merge
    while (packed != UINT64_MAX){
        uint32_t left  = static_cast<uint32_t>(packed >> 32);
        uint32_t right = static_cast<uint32_t>(packed & 0xFFFFFFFFu);

        // Perform the merge on tokens
        tokens = merge(tokens, left, right, merge_to_id.at(packed));
        packed = get_lowest_pair(tokens);
    }

    // Mistral post-processing, BOS token
    tokens.insert(tokens.begin(), token_to_id.at("<s>"));

    return tokens;
}

static bool try_decode_byte_token(const std::string& token, char& out) {
    // Vocab byte-fallback tokens use the form "<0x0A>" (see get_id).
    if (token.size() != 6 || token[0] != '<' || token[1] != '0' ||
        (token[2] != 'x' && token[2] != 'X') || token[5] != '>') {
        return false;
    }

    auto hex_digit = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };

    int hi = hex_digit(token[3]);
    int lo = hex_digit(token[4]);
    if (hi < 0 || lo < 0) {
        return false;
    }

    out = static_cast<char>((hi << 4) | lo);
    return true;
}

std::string Tokenizer::decode_mistral(std::string s) const{
    std::string t = "▁";
    std::string result;

    if (s.size() < t.size()) {
        return s;
    }

    int i = 0;

    while (i <= s.size() - t.size()){
        if (s.substr(i, t.size()) == t) {
            result += " ";
            i += t.size();
            continue;
        }
        result += s[i];
        i++;
    }

    if (i < s.size()){
        result += s.substr(i, s.size() - i);
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;

    for (auto& t : tokens) {
        const std::string& token = id_to_token[t];
        char byte;
        if (try_decode_byte_token(token, byte)) {
            result += byte;
        } else {
            result += decode_mistral(token);
        }
    }

    return result;
}