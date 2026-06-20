#pragma once

#include <cstring>
#include <iostream>
#include <string>
#include <cstdint>

class BinaryReader {
    const char* data_;
    size_t size_;
    size_t pos_ = 0;

    void require(size_t n) const {
        if (pos_ + n > size_) {
            std::cerr << "truncated header" << std::endl;
            std::exit(1);
        }
    }

public:
    BinaryReader(const char* data, size_t size) : data_(data), size_(size) {}

    size_t position() const { return pos_; }

    uint8_t read_u8() {
        require(1);
        return static_cast<uint8_t>(data_[pos_++]);
    }

    uint32_t read_u32() {
        require(4);
        uint32_t v;
        std::memcpy(&v, data_ + pos_, 4);
        pos_ += 4;
        return v;
    }

    uint64_t read_u64() {
        require(8);
        uint64_t v;
        std::memcpy(&v, data_ + pos_, 8);
        pos_ += 8;
        return v;
    }

    float read_f32() {
        require(4);
        float v;
        std::memcpy(&v, data_ + pos_, 4);
        pos_ += 4;
        return v;
    }

    std::string read_string() {
        uint32_t len = read_u32();
        require(len);
        std::string s(data_ + pos_, len);
        pos_ += len;
        return s;
    }
};
