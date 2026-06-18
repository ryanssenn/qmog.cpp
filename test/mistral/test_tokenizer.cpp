#include "setup/context.h"

int test_tokenizer_encode(){
    std::shared_ptr<Parameters> params = get_params();

    const std::string text = "The quick brown fox jumps over the lazy dog. Mistral models tokenize text using byte pair encoding, handling punctuation, emojis 😊, and multilingual words like déjà vu or 中文 with care.";
    std::vector<uint32_t> got = params->tokenizer.encode(text);

    const std::vector<uint32_t> expected = {
            1, 415, 2936, 9060, 285, 1142, 461, 10575, 754, 272,
            17898, 3914, 28723, 25200, 1650, 4994, 6029, 653, 2245, 1413,
            7500, 5964, 16087, 28725, 12852, 22195, 10223, 28725, 877, 6054,
            278, 28705, 30464, 28725, 304, 2531, 5708, 840, 3085, 737,
            28320, 20620, 442, 28705, 28991, 29019, 395, 1656, 28723
    };

    if (got.size() != expected.size()) {
        std::cout << "tokenizer encode: length mismatch, got "
                  << got.size() << " vs " << expected.size() << std::endl;
        return 1;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::cout << "tokenizer encode: id mismatch at " << i
                      << ", got " << got[i] << " vs " << expected[i] << std::endl;
            return 1;
        }
    }

    return 0;
}

static RegisterTest tokenizer_encode("tokenizer encode", "f32", &test_tokenizer_encode);
static RegisterTest tokenizer_encode_q("tokenizer encode", "int8", &test_tokenizer_encode);


int test_tokenizer_encode_byte_fallback(){
    std::shared_ptr<Parameters> params = get_params();

    const std::string text = std::string("A") + "\xEE\x80\x80" + "B";

    std::vector<uint32_t> got = params->tokenizer.encode(text);

    // Multiple bytes fallback from 1 token (241, 131, 131)
    const std::vector<uint32_t> expected = {
            1,
            330,
            241,
            131,
            131,
            28760
    };

    if (got.size() != expected.size()) {
        std::cout << "tokenizer encode fallback: length mismatch, got "
                  << got.size() << " vs " << expected.size() << std::endl;
        return 1;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::cout << "tokenizer encode fallback: id mismatch at " << i
                      << ", got " << got[i] << " vs " << expected[i] << std::endl;
            return 1;
        }
    }

    return 0;
}

static RegisterTest tokenizer_encode_fallback("tokenizer encode fallback", "f32", &test_tokenizer_encode_byte_fallback);
static RegisterTest tokenizer_encode_fallback_q("tokenizer encode fallback", "int8", &test_tokenizer_encode_byte_fallback);

int test_tokenizer_decode(){
    std::shared_ptr<Parameters> params = get_params();

    const std::string expected = " The quick brown fox jumps over the lazy dog.";

    const std::vector<uint32_t> ids = {
            415, 2936, 9060, 285, 1142, 461, 10575, 754, 272,
            17898, 3914, 28723
    };

    std::string got = params->tokenizer.decode(ids);

    if (expected != got){
        std::cout << "Tokenizer decode mismatch" << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest tokenizer_decode("tokenizer decode", "f32", &test_tokenizer_decode);

int test_tokenizer_decode_byte_fallback(){
    std::shared_ptr<Parameters> params = get_params();

    auto it = params->tokenizer.token_to_id.find("<0x0A>");
    if (it == params->tokenizer.token_to_id.end()) {
        std::cout << "tokenizer decode fallback: <0x0A> not in vocab" << std::endl;
        return 1;
    }

    std::string got = params->tokenizer.decode({it->second, it->second});
    const std::string expected = "\n\n";

    if (expected != got) {
        std::cout << "tokenizer decode fallback: newline mismatch" << std::endl;
        return 1;
    }

    const std::string text = "hello\nworld";
    std::vector<uint32_t> ids = params->tokenizer.encode(text);
    ids.erase(ids.begin());  // drop BOS; decode tests use raw token streams
    std::string roundtrip = params->tokenizer.decode(ids);

    if (roundtrip != " hello\nworld") {
        std::cout << "tokenizer decode fallback: roundtrip mismatch" << std::endl;
        return 1;
    }

    return 0;
}

static RegisterTest tokenizer_decode_fallback("tokenizer decode fallback", "f32", &test_tokenizer_decode_byte_fallback);
static RegisterTest tokenizer_decode_fallback_q("tokenizer decode fallback", "int8", &test_tokenizer_decode_byte_fallback);



