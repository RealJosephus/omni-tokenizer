#include "internal.h"
#include "native_vq_cases.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
struct Shape {
    size_t rows, heads, head_dim, block, valid;
    bool fused, bf16;
};

std::vector<float> values(size_t count, uint32_t seed) {
    std::vector<float> result(count);
    for (auto &value : result) {
        seed = 1664525U * seed + 1013904223U;
        value = float(int32_t(seed >> 8) - 8388608) / 8388608.f;
    }
    return result;
}

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "CPU attention failure: %s\n", message);
        std::exit(1);
    }
}

// Scalar reference fixes the reduction order of the existing CPU profile.
std::vector<float> reference(const Shape &shape, const std::vector<float> &q,
                             const std::vector<float> &k, const std::vector<float> &v) {
    const size_t d = shape.heads * shape.head_dim, cols = d * (shape.fused ? 3 : 1);
    const float scale = 1 / std::sqrt(float(shape.head_dim));
    const float split_scale = float(std::pow(double(shape.head_dim), -.25));
    std::vector<float> output(shape.rows * d);
    for (size_t t = 0; t < shape.rows; ++t)
        for (size_t h = 0; h < shape.heads; ++h) {
            size_t limit = std::min(shape.rows, shape.valid);
            if (shape.block)
                limit = std::min(limit, (t / shape.block + 1) * shape.block);
            std::vector<float> scores(limit);
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t s = 0; s < limit; ++s) {
                float dot = 0;
                for (size_t j = 0; j < shape.head_dim; ++j) {
                    const float query = q[t * cols + h * shape.head_dim + j];
                    const float key = shape.fused ? q[s * cols + d + h * shape.head_dim + j]
                                                  : k[s * d + h * shape.head_dim + j];
                    dot += shape.fused ? (query * split_scale) * (key * split_scale)
                                       : (query * scale) * key;
                }
                scores[s] = dot;
                maximum = std::max(maximum, dot);
            }
            float total = 0;
            for (auto &score : scores) {
                score = std::exp(score - maximum);
                total += score;
            }
            for (size_t j = 0; j < shape.head_dim; ++j) {
                float sum = 0;
                for (size_t s = 0; s < limit; ++s) {
                    const float value = shape.fused ? q[s * cols + 2 * d + h * shape.head_dim + j]
                                                    : v[s * d + h * shape.head_dim + j];
                    sum += scores[s] / total * value;
                }
                output[t * d + h * shape.head_dim + j] = shape.bf16 ? omni::round_bf16(sum) : sum;
            }
        }
    return output;
}

void numerical_case(const Shape &shape) {
    auto ops = omni::make_cpu();
    const size_t d = shape.heads * shape.head_dim, cols = d * (shape.fused ? 3 : 1);
    auto q_values = values(shape.rows * cols, 123);
    auto k_values = values(shape.rows * d, 456), v_values = values(shape.rows * d, 789);
    auto q = ops->upload(q_values.data(), shape.rows, cols, shape.bf16);
    auto k = ops->upload(k_values.data(), shape.rows, d, shape.bf16);
    auto v = ops->upload(v_values.data(), shape.rows, d, shape.bf16);
    auto actual =
        ops->download(ops->attention(q, shape.fused ? nullptr : &k, shape.fused ? nullptr : &v,
                                     shape.heads, shape.block, shape.valid, shape.fused));
    auto expected = reference(shape, ops->download(q), ops->download(k), ops->download(v));
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) != 0) {
        std::fprintf(stderr,
                     "Shape: rows=%zu heads=%zu head_dim=%zu block=%zu valid=%zu "
                     "fused=%d bf16=%d\n",
                     shape.rows, shape.heads, shape.head_dim, shape.block, shape.valid,
                     int(shape.fused), int(shape.bf16));
        check(false, "Scalar reduction order changed");
    }
}

void valid_key_and_block_mask() {
    auto ops = omni::make_cpu();
    constexpr size_t n = 13, d = 6, heads = 2, block = 5, valid = 7;
    std::vector<float> zero(n * d), value(n * d);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            value[t * d + j] = float(t + j + 1);
    auto q = ops->upload(zero.data(), n, d, false);
    auto k = ops->upload(zero.data(), n, d, false);
    auto run = [&](const std::vector<float> &input) {
        auto v = ops->upload(input.data(), n, d, false);
        return ops->download(ops->attention(q, &k, &v, heads, block, valid, false));
    };
    auto original = run(value);
    for (size_t t = valid; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            value[t * d + j] = 10000.f;
    auto invalid_keys_changed = run(value);
    check(original == invalid_keys_changed, "Invalid keys affected attention");
    for (size_t t = block; t < valid; ++t)
        for (size_t j = 0; j < d; ++j)
            value[t * d + j] = -10000.f;
    auto later_block_changed = run(value);
    check(std::equal(original.begin(), original.begin() + block * d, later_block_changed.begin()),
          "A later block affected earlier queries");
    check(original.back() != 0.f, "Padded queries were incorrectly cleared");
    check(later_block_changed.back() != original.back(), "Earlier blocks were inaccessible");
}
} // namespace

int main() {
    const Shape shapes[] = {
        {1, 1, 1, 0, 1, false, false},        {17, 3, 7, 5, 11, false, false},
        {17, 3, 7, 5, 11, true, true},        {201, 2, 13, 200, 199, false, false},
        {257, 3, 17, 200, 205, false, false}, {33, 3, 11, 0, 33, true, true},
        {71, 2, 64, 200, 69, false, false},   {71, 2, 96, 0, 71, true, true},
    };
    for (const auto &shape : shapes)
        numerical_case(shape);
    valid_key_and_block_mask();
    auto ops = omni::make_cpu();
    check_fp32_vq_refinement(*ops);
    return 0;
}
