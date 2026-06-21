#include "setup/context.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <unistd.h>

namespace {

bool color_enabled() {
    static const bool enabled = isatty(fileno(stdout)) && std::getenv("NO_COLOR") == nullptr;
    return enabled;
}

std::string paint(const std::string& code, const std::string& text) {
    if (!color_enabled()) {
        return text;
    }
    return "\033[" + code + "m" + text + "\033[0m";
}

const std::string BOLD  = "1";
const std::string DIM   = "2";
const std::string RED   = "31";
const std::string GREEN = "32";
const std::string CYAN  = "36";

std::string format_ms(double ms) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(1);
    ss << ms << " ms";
    return ss.str();
}

std::string rule(char c, size_t width) {
    return std::string(width, c);
}

void print_indented(const std::string& block) {
    std::istringstream in(block);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::cout << "        " << paint(DIM, line) << std::endl;
    }
}

} // namespace

int main() {
    load_expected_values();

    const std::string quant = get_params()->config.quant;

    size_t name_w = 0;
    int matching = 0;
    for (const TestCase& t : tests) {
        if (t.quant == quant || t.quant == "any") {
            name_w = std::max(name_w, t.name.size());
            matching++;
        }
    }
    const size_t time_w = 12;
    const size_t inner  = 2 + 1 + 2 + name_w + 2 + time_w;
    const size_t width  = std::max<size_t>(inner, 52);

    std::cout << std::endl;
    std::cout << paint(BOLD + ";" + CYAN, rule('=', width)) << std::endl;
    {
        std::string title = "  qmog.cpp \xC2\xB7 test suite";
        std::string model = "model: " + quant + "  ";
        size_t pad = width > title.size() + model.size()
                         ? width - title.size() - model.size()
                         : 1;
        std::cout << paint(BOLD, title) << std::string(pad, ' ')
                  << paint(CYAN, model) << std::endl;
    }
    std::cout << paint(BOLD + ";" + CYAN, rule('=', width)) << std::endl;
    std::cout << std::endl;

    int total  = 0;
    int failed = 0;
    auto suite_start = std::chrono::steady_clock::now();

    for (const TestCase& t : tests) {
        if (t.quant != quant && t.quant != "any") {
            continue;
        }
        total++;

        std::ostringstream captured;
        std::streambuf* old_cout = std::cout.rdbuf(captured.rdbuf());
        std::streambuf* old_cerr = std::cerr.rdbuf(captured.rdbuf());

        auto start = std::chrono::steady_clock::now();
        int result = t.func();
        auto end = std::chrono::steady_clock::now();

        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        bool ok = (result == 0);
        if (!ok) {
            failed++;
        }

        std::string mark = ok ? paint(GREEN, "\xE2\x9C\x93") : paint(RED, "\xE2\x9C\x97");
        std::string name = t.name;
        name.resize(name_w, ' ');
        std::string time_str = format_ms(ms);
        std::string time_pad(time_str.size() < time_w ? time_w - time_str.size() : 0, ' ');

        std::cout << "  " << mark << "  " << name << "  "
                  << time_pad << paint(DIM, time_str) << std::endl;

        if (!ok) {
            print_indented(captured.str());
        }
    }

    auto suite_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(suite_end - suite_start).count();
    int passed = total - failed;

    std::cout << std::endl;
    std::cout << paint(DIM, rule('-', width)) << std::endl;

    std::string status = (failed == 0)
                             ? paint(BOLD + ";" + GREEN, "PASSED")
                             : paint(BOLD + ";" + RED, "FAILED");
    std::ostringstream counts;
    counts << "   " << passed << " / " << total
           << "        " << failed << " failed"
           << "        " << format_ms(total_ms);
    std::cout << "  " << status << counts.str() << std::endl;

    std::cout << paint(BOLD + ";" + CYAN, rule('=', width)) << std::endl;
    std::cout << std::endl;

    return failed == 0 ? 0 : 1;
}
