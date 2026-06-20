#pragma once

#include <cstdint>

namespace model_format {

constexpr char MAGIC[4] = {'M', 'O', 'G', '\0'};
constexpr uint32_t FORMAT_VERSION = 1;

enum class DType : uint8_t {
    F32 = 0,
    INT8 = 1,
};

enum class KVType : uint8_t {
    STRING = 0,
    UINT32 = 1,
    FLOAT32 = 2,
};

constexpr size_t FILE_PREFIX_SIZE = 16; // magic + version + header_size

} // namespace model_format
