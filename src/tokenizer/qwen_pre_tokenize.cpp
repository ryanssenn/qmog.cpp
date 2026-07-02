#include "tokenizer/utf8.h"
#include "tokenizer/qwen_pre_tokenize.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

struct Range {
    uint32_t start;
    uint32_t end;
};

#include "unicode_ranges.inl"

bool in_ranges(uint32_t cp, const Range* ranges, size_t count) {
    size_t lo = 0;
    size_t hi = count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].start) {
            hi = mid;
        } else if (cp > ranges[mid].end) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

bool is_letter(uint32_t cp) {
    return in_ranges(cp, kLetterRanges, LetterRangesCount);
}

bool is_number(uint32_t cp) {
    return in_ranges(cp, kNumberRanges, NumberRangesCount);
}

bool is_whitespace(uint32_t cp) {
    return in_ranges(cp, kWhitespaceRanges, WhitespaceRangesCount);
}

bool is_newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

bool starts_with_ci(const std::string& text, size_t i, const char* lit) {
    for (size_t j = 0; lit[j] != '\0'; ++j) {
        if (i + j >= text.size()) {
            return false;
        }
        if (std::tolower(static_cast<unsigned char>(text[i + j])) !=
            std::tolower(static_cast<unsigned char>(lit[j]))) {
            return false;
        }
    }
    return true;
}

size_t try_contraction(const std::string& text, size_t i) {
    if (i >= text.size() || text[i] != '\'') {
        return i;
    }

    static const char* suffixes[] = {"re", "ve", "ll", "d", "s", "t", "m"};
    for (const char* suffix : suffixes) {
        if (starts_with_ci(text, i + 1, suffix)) {
            return i + 1 + std::strlen(suffix);
        }
    }

    return i;
}

size_t try_letter_run(const std::string& text, size_t i) {
    size_t j = i;
    uint32_t cp = 0;

    if (utf8_peek(text, j, cp) &&
        !is_newline(cp) && !is_letter(cp) && !is_number(cp)) {
        j = utf8_advance(text, j);
    }

    if (j >= text.size() || !utf8_peek(text, j, cp) || !is_letter(cp)) {
        return i;
    }

    while (j < text.size()) {
        if (!utf8_peek(text, j, cp) || !is_letter(cp)) {
            break;
        }
        j = utf8_advance(text, j);
    }

    return j;
}

size_t try_number(const std::string& text, size_t i) {
    uint32_t cp = 0;
    if (!utf8_peek(text, i, cp) || !is_number(cp)) {
        return i;
    }

    return utf8_advance(text, i);
}

size_t try_symbol_run(const std::string& text, size_t i) {
    size_t j = i;
    if (j < text.size() && text[j] == ' ') {
        ++j;
    }

    size_t symbol_start = j;
    uint32_t cp = 0;
    while (j < text.size()) {
        if (!utf8_peek(text, j, cp)) {
            break;
        }
        if (is_whitespace(cp) || is_letter(cp) || is_number(cp)) {
            break;
        }
        j = utf8_advance(text, j);
    }

    if (j == symbol_start) {
        return i;
    }

    while (j < text.size()) {
        if (!utf8_peek(text, j, cp) || !is_newline(cp)) {
            break;
        }
        j = utf8_advance(text, j);
    }

    return j;
}

size_t try_newline_whitespace(const std::string& text, size_t i) {
    size_t pos = i;
    uint32_t cp = 0;

    while (pos < text.size()) {
        if (!utf8_peek(text, pos, cp)) {
            break;
        }

        if (is_newline(cp)) {
            size_t j = pos;
            while (j < text.size()) {
                if (!utf8_peek(text, j, cp) || !is_newline(cp)) {
                    break;
                }
                j = utf8_advance(text, j);
            }
            return j;
        }

        if (!is_whitespace(cp)) {
            return i;
        }

        pos = utf8_advance(text, pos);
    }

    return i;
}

size_t try_trailing_whitespace(const std::string& text, size_t i) {
    if (i >= text.size()) {
        return i;
    }

    uint32_t cp = 0;
    if (!utf8_peek(text, i, cp) || !is_whitespace(cp)) {
        return i;
    }

    size_t j = i;
    size_t last_match = i;

    while (j < text.size()) {
        if (!utf8_peek(text, j, cp) || !is_whitespace(cp)) {
            break;
        }

        j = utf8_advance(text, j);

        if (j >= text.size()) {
            last_match = j;
            break;
        }

        if (utf8_peek(text, j, cp) && is_whitespace(cp)) {
            last_match = j;
        } else {
            break;
        }
    }

    return last_match > i ? last_match : i;
}

size_t try_whitespace(const std::string& text, size_t i) {
    size_t j = i;
    uint32_t cp = 0;
    size_t start = j;

    while (j < text.size()) {
        if (!utf8_peek(text, j, cp) || !is_whitespace(cp)) {
            break;
        }
        j = utf8_advance(text, j);
    }

    return j > start ? j : i;
}

size_t try_match(const std::string& text, size_t i) {
    size_t end = try_contraction(text, i);
    if (end > i) return end;

    end = try_letter_run(text, i);
    if (end > i) return end;

    end = try_number(text, i);
    if (end > i) return end;

    end = try_symbol_run(text, i);
    if (end > i) return end;

    end = try_newline_whitespace(text, i);
    if (end > i) return end;

    end = try_trailing_whitespace(text, i);
    if (end > i) return end;

    end = try_whitespace(text, i);
    if (end > i) return end;

    return i;
}

} // namespace

void QwenPreTokenizer::expect_pattern(const std::string& from_file) {
    if (from_file != pattern) {
        std::cerr << "Qwen pre_tokenize regex mismatch" << std::endl;
        std::exit(1);
    }
}

std::vector<std::string> QwenPreTokenizer::split(const std::string& text) const {
    std::vector<std::string> pieces;
    size_t i = 0;

    while (i < text.size()) {
        size_t end = try_match(text, i);
        if (end > i) {
            pieces.push_back(text.substr(i, end - i));
            i = end;
            continue;
        }

        size_t next = i;
        uint32_t cp = 0;
        utf8_next(text, next, cp);
        pieces.push_back(text.substr(i, next - i));
        i = next;
    }

    return pieces;
}
