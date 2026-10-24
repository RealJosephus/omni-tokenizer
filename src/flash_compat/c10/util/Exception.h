#pragma once
#include <sstream>
#include <stdexcept>

template <typename... Args>
[[noreturn]] inline void standalone_attention_error(const Args &...args) {
    std::ostringstream message;
    (message << ... << args);
    throw std::runtime_error(message.str());
}
#define TORCH_CHECK(condition, ...)                                                                \
    do {                                                                                           \
        if (!(condition))                                                                          \
            standalone_attention_error(__VA_ARGS__);                                               \
    } while (false)
