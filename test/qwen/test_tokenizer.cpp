#include "setup/context.h"

namespace {

bool expect_ids(const std::string& text, const std::vector<uint32_t>& expected) {
    std::vector<uint32_t> got = get_model()->tokenizer.encode(text);

    if (got != expected) {
        std::cout << "tokenizer ids mismatch for: " << text << std::endl;
        std::cout << "got:";
        for (uint32_t id : got) std::cout << " " << id;
        std::cout << std::endl;
        std::cout << "expected:";
        for (uint32_t id : expected) std::cout << " " << id;
        std::cout << std::endl;
        return false;
    }

    std::string decoded = get_model()->tokenizer.decode(got);
    if (decoded != text) {
        std::cout << "tokenizer roundtrip mismatch for: " << text << std::endl;
        std::cout << "decoded: " << decoded << std::endl;
        return false;
    }

    return true;
}

#include "expected/tokenizer.inl"

} // namespace

int test_qwen_tokenizer_encode_decode() {
    for (const auto& [text, ids] : kEncodeDecodeCases) {
        if (!expect_ids(text, ids)) {
            return 1;
        }
    }

    return 0;
}

static RegisterTest tokenizer_encode_decode_f16("tokenizer encode/decode", "f16", &test_qwen_tokenizer_encode_decode);
