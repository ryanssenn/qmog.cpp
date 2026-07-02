#pragma once

#include <cstdint>
#include <string>

// Encode a Unicode code point as a UTF-8 byte string.
inline std::string utf8_encode(uint32_t cp) {
    std::string out;

    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }

    return out;
}

// Read the next UTF-8 code point at i, advance i past it, and return false at end of string.
inline bool utf8_next(const std::string& s, size_t& i, uint32_t& cp) {
    if (i >= s.size()) return false;

    unsigned char b0 = static_cast<unsigned char>(s[i]);

    if ((b0 & 0x80) == 0x00) {
        cp = b0;
        i += 1;
        return true;
    }

    if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
        cp = ((b0 & 0x1F) << 6)
           | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        i += 2;
        return true;
    }

    if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
        cp = ((b0 & 0x0F) << 12)
           | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6)
           | (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        i += 3;
        return true;
    }

    if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
        cp = ((b0 & 0x07) << 18)
           | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12)
           | ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6)
           | (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        i += 4;
        return true;
    }

    cp = b0;
    i += 1;
    return true;
}

// Read the UTF-8 code point at i without advancing i.
inline bool utf8_peek(const std::string& s, size_t i, uint32_t& cp) {
    size_t j = i;
    return utf8_next(s, j, cp);
}

// Return the byte index after the UTF-8 code point starting at i.
inline size_t utf8_advance(const std::string& s, size_t i) {
    uint32_t cp = 0;
    utf8_next(s, i, cp);
    return i;
}
