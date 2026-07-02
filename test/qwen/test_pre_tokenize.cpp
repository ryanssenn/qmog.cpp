#include "tokenizer/qwen_pre_tokenize.h"
#include "setup/context.h"

#include <iostream>

namespace {

const QwenPreTokenizer& qwen_pre_tokenizer() {
    return get_model()->tokenizer.pre_tokenizer;
}

// Compare split output to an expected list of pieces and print both on mismatch.
bool expect_pieces(const QwenPreTokenizer& pt,
                   const std::string& text,
                   const std::vector<std::string>& expected) {
    std::vector<std::string> got = pt.split(text);

    if (got == expected) {
        return true;
    }

    std::cout << "pre-tokenize split mismatch for: " << text << std::endl;
    std::cout << "got:";
    for (const std::string& piece : got) {
        std::cout << " \"" << piece << "\"";
    }
    std::cout << std::endl;
    std::cout << "expected:";
    for (const std::string& piece : expected) {
        std::cout << " \"" << piece << "\"";
    }
    std::cout << std::endl;
    return false;
}

// Verify that concatenating all split pieces reproduces the original input string.
bool expect_joined(const QwenPreTokenizer& pt, const std::string& text) {
    std::vector<std::string> got = pt.split(text);

    std::string joined;
    for (const std::string& piece : got) {
        joined += piece;
    }

    if (joined == text) {
        return true;
    }

    std::cout << "pre-tokenize pieces do not rejoin input: " << text << std::endl;
    std::cout << "joined: " << joined << std::endl;
    return false;
}

#include "expected/pre_tokenize.inl"

} // namespace

// Empty input should return no pieces when using the Qwen pre-tokenize regex.
int test_pre_tokenize_empty() {
    if (!expect_pieces(qwen_pre_tokenizer(), "", {})) {
        return 1;
    }

    return 0;
}

// The hardcoded Qwen regex should split known cases into expected pieces.
int test_pre_tokenize_qwen_splits() {
    const QwenPreTokenizer& pt = qwen_pre_tokenizer();

    for (const auto& [text, expected] : kSplitCases) {
        if (!expect_pieces(pt, text, expected)) {
            return 1;
        }
    }

    return 0;
}

// Split pieces from the Qwen regex should always concatenate back to the original text.
int test_pre_tokenize_qwen_roundtrip() {
    const QwenPreTokenizer& pt = qwen_pre_tokenizer();

    for (const std::string& text : kRoundtripInputs) {
        if (!expect_joined(pt, text)) {
            return 1;
        }
    }

    return 0;
}

static RegisterTest pre_tokenize_empty("pre-tokenize empty", "f16", &test_pre_tokenize_empty);
static RegisterTest pre_tokenize_qwen_splits("pre-tokenize qwen splits", "f16", &test_pre_tokenize_qwen_splits);
static RegisterTest pre_tokenize_qwen_roundtrip("pre-tokenize qwen roundtrip", "f16", &test_pre_tokenize_qwen_roundtrip);
