#pragma once
#include "internal.h"
#include <algorithm>
#include <stdexcept>

inline void check_fp32_vq_refinement(omni::Ops &ops) {
    constexpr size_t rows = 3, width = 1280, codes = 1027;
    std::vector<float> input(rows * width, 1000.f), codebook(codes * width, 1000.25f);
    std::fill_n(input.data() + width, width, 1002.f);
    std::fill_n(input.data() + 2 * width, width, 1e20f);
    // Near neighbours share a large common component. Expanded FP32 distances
    // lose their ordering; exact duplicates also exercise ties across tiles.
    std::fill_n(codebook.data() + 7 * width, width, 1000.f + 1.f / 1024);
    for (size_t id : {size_t(1024), size_t(1025)})
        std::copy_n(input.data(), width, codebook.data() + id * width);
    std::copy_n(input.data() + width, width, codebook.data() + 3 * width);
    // Squared FP32 norms overflow, but direct FP64 distances remain finite.
    std::copy_n(input.data() + 2 * width, width, codebook.data() + 1026 * width);
    ops.begin_request();
    {
        auto x = ops.upload(input.data(), rows, width, false);
        auto cb = ops.upload(codebook.data(), codes, width, false);
        auto norms = ops.squares(cb);
        std::atomic<bool> cancel{false};
        const auto actual = ops.quantize(x, cb, norms, cancel);
        if (actual != std::vector<uint16_t>{1024, 3, 1026})
            throw std::runtime_error("FP32 VQ lost a nearest code to cancellation or overflow");
    }
    ops.end_request();
}
