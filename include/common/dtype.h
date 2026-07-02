#pragma once

#include <cstddef>
#include <cstdint>

enum class DType : uint8_t {
    F32 = 0,
    F16 = 2,
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
    }
    return 0;
}

inline DType dtype_from_file(uint8_t file_dtype) {
    return static_cast<DType>(file_dtype);
}
