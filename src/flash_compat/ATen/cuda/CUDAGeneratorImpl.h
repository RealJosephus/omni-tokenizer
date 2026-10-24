#pragma once
#include <cstdint>
#ifndef FLASHATTENTION_DISABLE_DROPOUT
#error This standalone adapter only supports dropout disabled.
#endif
namespace at {
struct PhiloxCudaState {
    uint64_t seed, offset;
    bool captured_ = false;
    PhiloxCudaState(uint64_t s = 0, uint64_t o = 0) : seed(s), offset(o) {}
};
} // namespace at
