#include "tokenizer/utf8.h"
#include "tokenizer/qwen_tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <utility>

namespace {

struct ByteMaps {
    std::array<std::string, 256> byte_to_token;
    std::unordered_map<uint32_t, uint8_t> codepoint_to_byte;
};

const ByteMaps& byte_maps() {
    static const ByteMaps maps = [] {
        ByteMaps m;

        std::vector<uint32_t> bytes;
        std::vector<uint32_t> codepoints;

        for (uint32_t b = '!'; b <= '~'; ++b) bytes.push_back(b);
        for (uint32_t b = 0xA1; b <= 0xAC; ++b) bytes.push_back(b);
        for (uint32_t b = 0xAE; b <= 0xFF; ++b) bytes.push_back(b);

        codepoints = bytes;

        uint32_t next = 0;
        std::unordered_set<uint32_t> printable(bytes.begin(), bytes.end());
        for (uint32_t b = 0; b < 256; ++b) {
            if (printable.find(b) != printable.end()) {
                continue;
            }

            bytes.push_back(b);
            codepoints.push_back(256 + next);
            next++;
        }

        for (size_t i = 0; i < bytes.size(); ++i) {
            m.byte_to_token[bytes[i]] = utf8_encode(codepoints[i]);
            m.codepoint_to_byte[codepoints[i]] = static_cast<uint8_t>(bytes[i]);
        }

        return m;
    }();

    return maps;
}

bool is_special_token(const std::string& token) {
    return token.size() >= 4
        && token.compare(0, 2, "<|") == 0
        && token.compare(token.size() - 2, 2, "|>") == 0;
}

}

// Loads vocab and merge rules from the MOG header.
// MOG v2 stores a pre-tokenize regex that must match the hardcoded Qwen pattern.
void QwenTokenizer::load(BinaryReader& reader) {
    Bpe::load(reader);
    QwenPreTokenizer::expect_pattern(reader.read_string());
}

const std::vector<std::pair<std::string, uint32_t>>& QwenTokenizer::special_tokens_list() const {
    if (special_tokens_initialized) {
        return special_tokens;
    }

    special_tokens.clear();
    for (const auto& [tok, id] : token_to_id) {
        if (is_special_token(tok)) {
            special_tokens.push_back({tok, id});
        }
    }
    std::sort(special_tokens.begin(), special_tokens.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });
    special_tokens_initialized = true;
    return special_tokens;
}

std::vector<uint32_t> QwenTokenizer::bpe(const std::string& text) const {
    const auto& maps = byte_maps();
    std::vector<uint32_t> tokens;
    tokens.reserve(text.size());

    for (unsigned char b : text) {
        auto it = token_to_id.find(maps.byte_to_token[b]);
        if (it == token_to_id.end()) {
            std::cerr << "byte token missing from vocabulary" << std::endl;
            std::exit(1);
        }
        tokens.push_back(it->second);
    }

    return apply_merges(std::move(tokens));
}

std::vector<uint32_t> QwenTokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> result;

    size_t pos = 0;

    while (pos < text.size()) {
        size_t next_special = std::string::npos;
        std::string special;
        uint32_t special_id = 0;

        for (const auto& [tok, id] : special_tokens_list()) {
            size_t found = text.find(tok, pos);
            if (found != std::string::npos &&
                (next_special == std::string::npos || found < next_special ||
                 (found == next_special && tok.size() > special.size()))) {
                next_special = found;
                special = tok;
                special_id = id;
            }
        }

        std::string regular = next_special == std::string::npos
            ? text.substr(pos)
            : text.substr(pos, next_special - pos);

        for (const std::string& piece : pre_tokenizer.split(regular)) {
            std::vector<uint32_t> ids = bpe(piece);
            result.insert(result.end(), ids.begin(), ids.end());
        }

        if (next_special == std::string::npos) {
            break;
        }

        result.push_back(special_id);
        pos = next_special + special.size();
    }

    return result;
}

std::string QwenTokenizer::decode(const std::vector<uint32_t>& tokens) const {
    const auto& maps = byte_maps();
    std::string result;

    for (uint32_t id : tokens) {
        if (id >= id_to_token.size()) {
            std::cerr << "token id out of range: " << id << std::endl;
            std::exit(1);
        }

        const std::string& token = id_to_token[id];
        if (is_special_token(token)) {
            result += token;
            continue;
        }

        size_t i = 0;
        while (i < token.size()) {
            uint32_t cp = 0;
            utf8_next(token, i, cp);
            auto it = maps.codepoint_to_byte.find(cp);
            if (it == maps.codepoint_to_byte.end()) {
                std::cerr << "token cannot be decoded as byte-level text: "
                          << token << std::endl;
                std::exit(1);
            }
            result.push_back(static_cast<char>(it->second));
        }
    }

    return result;
}
