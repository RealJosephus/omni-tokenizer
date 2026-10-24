#include "internal.h"
#include "native_vq_cases.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<float> values(size_t count, uint32_t seed) {
    std::vector<float> result(count);
    for (auto &value : result) {
        seed = 1664525U * seed + 1013904223U;
        value = float(int32_t(seed >> 8) - 8388608) / 8388608.f;
    }
    return result;
}

std::vector<float> run(omni::Ops &ops, const std::vector<float> &q, const std::vector<float> &k,
                       const std::vector<float> &v, size_t n, size_t heads, size_t hd, size_t block,
                       size_t valid, bool fused = false) {
    ops.begin_request();
    struct EndRequest {
        omni::Ops &ops;
        ~EndRequest() { ops.end_request(); }
    } guard{ops};
    const size_t d = heads * hd;
    auto qv = ops.upload(q.data(), n, d * (fused ? 3 : 1), fused);
    omni::Value kv, vv;
    if (!fused) {
        kv = ops.upload(k.data(), n, d, false);
        vv = ops.upload(v.data(), n, d, false);
    }
    return ops.download(ops.attention(qv, fused ? nullptr : &kv, fused ? nullptr : &vv, heads,
                                      block, valid, fused));
}

void numerical_case(omni::Ops &ops, size_t n, size_t heads, size_t hd, size_t block, size_t valid,
                    bool fused = false) {
    const size_t d = heads * hd, cols = d * (fused ? 3 : 1);
    auto q = values(n * cols, 123), k = values(n * d, 456), v = values(n * d, 789);
    if (fused)
        for (auto &value : q)
            value = omni::round_bf16(value);
    auto actual = run(ops, q, k, v, n, heads, hd, block, valid, fused);
    for (float value : actual)
        check(std::isfinite(value), "Attention returned a non-finite value");
    // Check tile, block and valid-tail boundaries against a scalar reference.
    std::vector<size_t> queries{0, n - 1, std::min(n - 1, valid - 1)};
    for (size_t boundary : {size_t(128), size_t(256), block, valid}) {
        if (boundary && boundary <= n)
            queries.push_back(boundary - 1);
        if (boundary < n)
            queries.push_back(boundary);
    }
    const float scale = fused ? float(std::pow(double(hd), -.25)) : 1 / std::sqrt(float(hd));
    for (size_t t : queries)
        for (size_t h = 0; h < heads; ++h) {
            size_t limit = std::min(n, valid);
            if (block)
                limit = std::min(limit, (t / block + 1) * block);
            std::vector<double> scores(limit);
            for (size_t s = 0; s < limit; ++s) {
                double dot = 0;
                for (size_t j = 0; j < hd; ++j) {
                    float a = q[t * cols + h * hd + j] * scale;
                    float b = fused ? q[s * cols + d + h * hd + j] * scale : k[s * d + h * hd + j];
                    dot += double(a) * b;
                }
                scores[s] = dot;
            }
            double maximum = *std::max_element(scores.begin(), scores.end()), total = 0;
            for (auto &score : scores) {
                score = std::exp(score - maximum);
                total += score;
            }
            for (size_t j = 0; j < hd; ++j) {
                double expected = 0;
                for (size_t s = 0; s < limit; ++s) {
                    float value = fused ? q[s * cols + 2 * d + h * hd + j] : v[s * d + h * hd + j];
                    expected += scores[s] / total * value;
                }
                double tolerance = fused ? 2e-3 : 3e-6;
                check(std::abs(actual[t * d + h * hd + j] - expected) <= tolerance,
                      "Attention disagrees with the scalar mask/reference");
            }
        }
    check(ops.memory->live.load() <= 160ULL * 1024 * 1024,
          "Attention retained more than the workspace and idle pool budget");
}

void image_patch_case(omni::Ops &ops) {
    for (size_t patch : {size_t(1), size_t(2), size_t(16)}) {
        const size_t height = patch * 2, width = patch * 3, pd = 8 * patch * patch;
        auto pixels = values(3 * height * width, 314159);
        ops.begin_request();
        {
            auto patches = ops.download(ops.image_patches(pixels.data(), height, width, patch));
            for (size_t y = 0; y < 2; ++y)
                for (size_t x = 0; x < 3; ++x)
                    for (size_t dy = 0; dy < patch; ++dy)
                        for (size_t dx = 0; dx < patch; ++dx)
                            for (size_t channel = 0; channel < 8; ++channel) {
                                float expected =
                                    channel < 3
                                        ? omni::round_bf16(
                                              pixels[channel * height * width +
                                                     (y * patch + dy) * width + x * patch + dx])
                                        : 0;
                                check(patches[(y * 3 + x) * pd + (dy * patch + dx) * 8 + channel] ==
                                          expected,
                                      "Image patch packing changed pixel order");
                            }
        }
        ops.end_request();
    }
}

void linear_case(omni::Ops &ops, size_t rows, size_t columns, size_t width) {
    auto x = values(rows * width, 1357), w = values(columns * width, 2468);
    auto bias = values(columns, 9753);
    // These values distinguish reconstructed products from a single TF32
    // product, which would discard their low components.
    std::fill_n(x.data(), width, 1.f + 1.f / 8192);
    std::fill_n(w.data(), width, 1.f + 1.f / 4096);
    ops.begin_request();
    {
        auto input = ops.upload(x.data(), rows, width, false);
        auto weight = ops.upload(w.data(), columns, width, false);
        auto offset = ops.upload(bias.data(), 1, columns, false);
        auto actual = ops.download(ops.linear(input, weight, &offset));
        for (size_t row : {size_t(0), rows / 2, rows - 1})
            for (size_t column : {size_t(0), columns / 2, columns - 1}) {
                double expected = bias[column], magnitude = std::abs(expected);
                for (size_t k = 0; k < width; ++k) {
                    const double product = double(x[row * width + k]) * w[column * width + k];
                    expected += product;
                    magnitude += std::abs(product);
                }
                check(std::abs(actual[row * columns + column] - expected) <=
                          2e-6 * magnitude + 1e-6,
                      "Biased linear lost precision or changed its layout");
            }
    }
    ops.end_request();
}

void quantization_case(omni::Ops &ops, bool bf) {
    constexpr size_t rows = 9, width = 16, codes = 2051;
    std::vector<float> x(rows * width), codebook(codes * width);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = float(int(i * 7 % 13) - 6) * 0.25f;
    for (size_t i = 0; i < codebook.size(); ++i)
        codebook[i] = float(int(i * 11 % 17) - 8) * 0.25f;
    // Exact duplicate winners span the tile boundary and the partial final tile.
    for (size_t index : {size_t(7), size_t(1024), codes - 1})
        std::copy_n(x.data(), width, codebook.data() + index * width);
    ops.begin_request();
    {
        auto xv = ops.upload(x.data(), rows, width, bf);
        auto cb = ops.upload(codebook.data(), codes, width, bf);
        auto norms = ops.squares(cb);
        std::atomic<bool> cancel{false};
        auto actual = ops.quantize(xv, cb, norms, cancel);
        auto round = [bf](float v) { return bf ? omni::round_bf16(v) : v; };
        for (size_t row = 0; row < rows; ++row) {
            float xn = 0, best = INFINITY;
            size_t expected = 0;
            for (size_t j = 0; j < width; ++j)
                xn += round(x[row * width + j] * x[row * width + j]);
            xn = round(xn);
            for (size_t code = 0; code < codes; ++code) {
                float cn = 0, dot = 0;
                for (size_t j = 0; j < width; ++j) {
                    cn += round(codebook[code * width + j] * codebook[code * width + j]);
                    dot += x[row * width + j] * codebook[code * width + j];
                }
                cn = round(cn);
                dot = round(dot);
                float distance =
                    bf ? round(round(xn + cn) - round(2 * dot)) : std::fma(-2.f, dot, xn + cn);
                if (distance < best) {
                    best = distance;
                    expected = code;
                }
            }
            check(actual[row] == expected, "VQ changed the distance formula or tie ordering");
        }
        check(actual[0] == 7, "VQ did not choose the smallest duplicate code ID");
    }
    ops.end_request();
}

void mask_case(omni::Ops &ops, size_t n, size_t hd) {
    constexpr size_t heads = 2, block = 200, valid = 205;
    const size_t d = heads * hd;
    std::vector<float> zero(n * d), v(n * d);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            v[t * d + j] = float(t + j + 1);
    auto before = run(ops, zero, zero, v, n, heads, hd, block, valid);
    for (size_t t = valid; t < n; ++t)
        std::fill_n(v.data() + t * d, d, 10000.f);
    auto invalid_changed = run(ops, zero, zero, v, n, heads, hd, block, valid);
    check(before == invalid_changed, "Invalid keys affected attention");
    for (size_t t = block; t < valid; ++t)
        std::fill_n(v.data() + t * d, d, -10000.f);
    auto future_changed = run(ops, zero, zero, v, n, heads, hd, block, valid);
    check(std::equal(before.begin(), before.begin() + block * d, future_changed.begin()),
          "A future block affected earlier queries");
    check(before.back() != 0.f && before.back() != future_changed.back(),
          "Padded queries lost access to valid keys");
}
} // namespace

int main() {
    int devices = 0;
    auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
        std::puts("CUDA attention test skipped: no usable CUDA device");
        return 77;
    }
    try {
        check(status == cudaSuccess, "Cannot query CUDA devices");
        auto ops = omni::make_cuda(0);
        ops->enable_reuse();
        for (size_t n : {1, 129, 257, 511, 512, 513, 1500, 201, 401, 1}) {
            numerical_case(*ops, n, 2, 7, 0, n);
            numerical_case(*ops, n, 4, 64, 200, n > 3 ? n - 3 : 1);
            numerical_case(*ops, n, 2, 7, 7, 1);
        }
        for (size_t n : {31, 32, 129, 255, 256, 257, 385, 512, 2049})
            numerical_case(*ops, n, 2, 96, 0, n, true);
        for (size_t n : {1, 129, 257, 576, 768, 1080, 2049})
            numerical_case(*ops, n, 16, 96, 0, n, true);
        numerical_case(*ops, 1, 4, 64, 200, 5);
        for (size_t n : {401, 601, 1025})
            for (size_t hd : {7, 64})
                mask_case(*ops, n, hd);
        image_patch_case(*ops);
        for (size_t rows : {255, 256, 257, 512}) {
            linear_case(*ops, rows, 32, 256);
            linear_case(*ops, rows, 17, 31);
        }
        linear_case(*ops, 257, 1280, 1280);
        linear_case(*ops, 512, 1280, 5120);
        quantization_case(*ops, false);
        quantization_case(*ops, true);
        check_fp32_vq_refinement(*ops);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "CUDA attention failure: %s\n", error.what());
        return 1;
    }
}
