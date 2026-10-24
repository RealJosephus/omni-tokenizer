#include "internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef OMNI_CBLAS
#include <cblas.h>
#endif

namespace omni {
namespace {
class Cpu final : public Ops {
    Value allocate(size_t rows, size_t cols, bool bf16) {
        const size_t n = rows * cols;
        auto *raw = new float[n];
        auto mem = memory;
        mem->add(n * sizeof(float));
        auto ptr = std::shared_ptr<void>(raw, [mem, n](void *p) {
            delete[] static_cast<float *>(p);
            mem->live.fetch_sub(n * sizeof(float));
        });
        return {std::move(ptr), rows, cols, bf16};
    }
    static float *p(const Value &v) { return static_cast<float *>(v.data()); }
    static float r(float x, bool b) { return b ? round_bf16(x) : x; }

  public:
    Value upload(const float *x, size_t rows, size_t cols, bool b) override {
        auto y = allocate(rows, cols, b);
        for (size_t i = 0; i < y.size(); ++i)
            p(y)[i] = r(x[i], b);
        return y;
    }
    Value raw_upload(const void *x, size_t rows, size_t cols, bool b) override {
        auto y = allocate(rows, cols, b);
        if (!b)
            std::memcpy(y.data(), x, y.size() * 4);
        else {
            const auto *s = static_cast<const uint16_t *>(x);
            for (size_t i = 0; i < y.size(); ++i) {
                uint32_t u = uint32_t(s[i]) << 16;
                std::memcpy(p(y) + i, &u, 4);
            }
        }
        return y;
    }
    std::vector<float> download(const Value &v) override { return {p(v), p(v) + v.size()}; }
    Value linear(const Value &x, const Value &w, const Value *b) override {
        require(x.cols == w.cols, "Linear dimensions disagree", OMNI_INVALID_MODEL);
        auto y = allocate(x.rows, w.rows, x.bf16);
        const size_t m = x.rows, n = w.rows, k = x.cols;
#ifdef OMNI_CBLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(m),
                    static_cast<int>(n), static_cast<int>(k), 1, p(x), static_cast<int>(k), p(w),
                    static_cast<int>(k), 0, p(y), static_cast<int>(n));
#else
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(m); ++i)
            for (size_t j = 0; j < n; ++j) {
                float s = 0;
                for (size_t z = 0; z < k; ++z)
                    s += p(x)[i * k + z] * p(w)[j * k + z];
                p(y)[i * n + j] = s;
            }
#endif
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(m * n); ++i)
            p(y)[i] = r(p(y)[i] + (b ? p(*b)[i % n] : 0), y.bf16);
        return y;
    }
    Value norm(const Value &x, const Value &w, const Value &b, float eps) override {
        auto y = allocate(x.rows, x.cols, x.bf16);
        const auto d = x.cols;
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(x.rows); ++i) {
            float mean = 0;
            for (size_t j = 0; j < d; ++j)
                mean += p(x)[i * d + j];
            mean /= float(d);
            float var = 0;
            for (size_t j = 0; j < d; ++j) {
                float z = p(x)[i * d + j] - mean;
                var += z * z;
            }
            float inv = 1 / std::sqrt(var / float(d) + eps);
            for (size_t j = 0; j < d; ++j)
                p(y)[i * d + j] = r((p(x)[i * d + j] - mean) * inv * p(w)[j] + p(b)[j], x.bf16);
        }
        return y;
    }
    void gelu(Value &x) override {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(x.size()); ++i) {
            float z = p(x)[i];
            p(x)[i] = r(0.5f * z * (1 + std::erf(z * 0.7071067811865475244f)), x.bf16);
        }
    }
    void add(Value &x, const Value &y) override {
        require(x.cols == y.cols && y.rows >= x.rows, "Residual dimensions disagree",
                OMNI_INTERNAL_ERROR);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(x.size()); ++i)
            p(x)[i] = r(p(x)[i] + p(y)[i], x.bf16);
    }
    void bias(Value &x, const Value &b) override {
        require(b.size() == x.cols, "Bias dimensions disagree", OMNI_INTERNAL_ERROR);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(x.size()); ++i)
            p(x)[i] = r(p(x)[i] + p(b)[i % x.cols], x.bf16);
    }
    Value conv_gather(const Value &x, size_t stride) override {
        auto y = allocate((x.rows + stride - 1) / stride, x.cols * 3, x.bf16);
#pragma omp parallel for schedule(static)
        for (int64_t t = 0; t < static_cast<int64_t>(y.rows); ++t)
            for (size_t c = 0; c < x.cols; ++c)
                for (int k = 0; k < 3; ++k) {
                    auto s = t * static_cast<int64_t>(stride) + k - 2;
                    p(y)[t * y.cols + c * 3 + k] = s >= 0 ? p(x)[s * x.cols + c] : 0;
                }
        return y;
    }
    Value attention(const Value &q, const Value *k, const Value *v, size_t heads, size_t block,
                    size_t valid, bool fused) override {
        size_t n = q.rows, d = fused ? q.cols / 3 : q.cols, hd = d / heads;
        auto y = allocate(n, d, q.bf16);
        float scale = 1 / std::sqrt(float(hd));
        float split_scale = float(std::pow(double(hd), -.25));
#pragma omp parallel
        {
            // Each worker keeps its score storage across query rows.
            std::vector<float> scores;
#pragma omp for schedule(static)
            for (int64_t task = 0; task < static_cast<int64_t>(n * heads); ++task) {
                size_t t = static_cast<size_t>(task) / heads, h = static_cast<size_t>(task) % heads,
                       limit = std::min(n, valid);
                if (block)
                    limit = std::min(limit, (t / block + 1) * block);
                scores.resize(limit);
                float mx = -std::numeric_limits<float>::infinity();
                for (size_t s = 0; s < limit; ++s) {
                    float z = 0;
                    for (size_t j = 0; j < hd; ++j) {
                        float a = p(q)[t * q.cols + h * hd + j];
                        float b =
                            fused ? p(q)[s * q.cols + d + h * hd + j] : p(*k)[s * d + h * hd + j];
                        z += fused ? (a * split_scale) * (b * split_scale) : (a * scale) * b;
                    }
                    scores[s] = z;
                    mx = std::max(mx, scores[s]);
                }
                float total = 0;
                for (auto &z : scores) {
                    z = std::exp(z - mx);
                    total += z;
                }
                auto *output = p(y) + t * d + h * hd;
                std::fill(output, output + hd, 0.f);
                for (size_t s = 0; s < limit; ++s) {
                    const float probability = scores[s] / total;
                    const float *value =
                        fused ? p(q) + s * q.cols + 2 * d + h * hd : p(*v) + s * d + h * hd;
                    // Keep each output's key order while traversing contiguous values.
                    for (size_t j = 0; j < hd; ++j)
                        output[j] += probability * value[j];
                }
                if (q.bf16)
                    for (size_t j = 0; j < hd; ++j)
                        output[j] = r(output[j], true);
            }
        }
        return y;
    }
    Value pool(const Value &x, size_t factor) override {
        auto y = allocate((x.rows + factor - 1) / factor, x.cols, x.bf16);
        for (size_t t = 0; t < y.rows; ++t)
            for (size_t j = 0; j < x.cols; ++j) {
                float sum = 0;
                for (size_t k = 0; k < factor && t * factor + k < x.rows; ++k)
                    sum += p(x)[(t * factor + k) * x.cols + j];
                p(y)[t * x.cols + j] = r(sum / float(factor), x.bf16);
            }
        return y;
    }
    Value interpolate(const Value &x, size_t source, size_t height, size_t width) override {
        auto y = allocate(height * width, x.cols, x.bf16);
        for (size_t i = 0; i < height; ++i)
            for (size_t j = 0; j < width; ++j) {
                float sy = std::max(0.f, (float(i) + .5f) * (float(source) / float(height)) - .5f),
                      sx = std::max(0.f, (float(j) + .5f) * (float(source) / float(width)) - .5f);
                size_t y0 = static_cast<size_t>(sy), x0 = static_cast<size_t>(sx),
                       y1 = std::min(y0 + 1, source - 1), x1 = std::min(x0 + 1, source - 1);
                float fy = sy - float(y0), fx = sx - float(x0);
                for (size_t c = 0; c < x.cols; ++c) {
                    float a = std::fma(1 - fx, p(x)[(y0 * source + x0) * x.cols + c],
                                       fx * p(x)[(y0 * source + x1) * x.cols + c]);
                    float b = std::fma(1 - fx, p(x)[(y1 * source + x0) * x.cols + c],
                                       fx * p(x)[(y1 * source + x1) * x.cols + c]);
                    p(y)[(i * width + j) * x.cols + c] = r(std::fma(1 - fy, a, fy * b), x.bf16);
                }
            }
        return y;
    }
    Value normalize(const Value &x) override {
        auto y = allocate(x.rows, x.cols, x.bf16);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(x.rows); ++i) {
            float n = 0;
            for (size_t j = 0; j < x.cols; ++j)
                n += p(x)[i * x.cols + j] * p(x)[i * x.cols + j];
            n = std::max(r(std::sqrt(n), x.bf16), r(1e-12f, x.bf16));
            for (size_t j = 0; j < x.cols; ++j)
                p(y)[i * x.cols + j] = r(p(x)[i * x.cols + j] / n, x.bf16);
        }
        return y;
    }
    Value squares(const Value &x) override {
        auto y = allocate(x.rows, 1, x.bf16);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(x.rows); ++i) {
            float n = 0;
            for (size_t j = 0; j < x.cols; ++j)
                n += r(p(x)[i * x.cols + j] * p(x)[i * x.cols + j], x.bf16);
            p(y)[i] = r(n, x.bf16);
        }
        return y;
    }
    std::vector<uint16_t> quantize(const Value &x, const Value &cb, const Value &cbnorm,
                                   std::atomic<bool> &cancel) override {
        auto xn = squares(x);
        std::vector<uint16_t> ids(x.rows);
        std::vector<uint16_t> approximate_ids(x.bf16 ? 0 : x.rows);
        auto &search_ids = x.bf16 ? ids : approximate_ids;
        std::vector<float> best(x.rows, std::numeric_limits<float>::infinity());
        std::vector<double> precise_best(x.bf16 ? 0 : x.rows,
                                         std::numeric_limits<double>::infinity());
        const double error_factor = fp32_distance_error_factor(x.cols);
        constexpr size_t tile = 256;
        for (size_t base = 0; base < cb.rows; base += tile) {
            if (cancel.load())
                throw Error(OMNI_CANCELLED, "Inference cancelled");
            size_t n = std::min(tile, cb.rows - base);
            Value part{std::shared_ptr<void>(cb.storage, p(cb) + base * cb.cols), n, cb.cols,
                       cb.bf16};
            auto dots = linear(x, part, nullptr);
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(x.rows); ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float dot = p(dots)[i * n + j];
                    float dist;
                    if (x.bf16) {
                        float sum = r(p(xn)[i] + p(cbnorm)[base + j], true);
                        dist = r(sum - r(2 * dot, true), true);
                    } else
                        dist = std::fma(-2.f, dot, p(xn)[i] + p(cbnorm)[base + j]);
                    if (dist < best[i]) {
                        best[i] = dist;
                        search_ids[i] = static_cast<uint16_t>(base + j);
                    }
                }
                if (x.bf16)
                    continue;
                // Expanded L2 can lose a small distance to cancellation. Only
                // candidates inside its error bound need a direct FP64 sum.
                for (size_t j = 0; j < n; ++j) {
                    const size_t code = base + j;
                    const float distance =
                        std::fma(-2.f, p(dots)[i * n + j], p(xn)[i] + p(cbnorm)[code]);
                    const double tolerance =
                        error_factor *
                        (2 * double(p(xn)[i]) + p(cbnorm)[code] + p(cbnorm)[search_ids[i]]);
                    if (double(distance) > double(best[i]) + tolerance)
                        continue;
                    double direct = 0;
                    for (size_t k = 0; k < x.cols; ++k) {
                        const double delta =
                            double(p(x)[i * x.cols + k]) - p(cb)[code * x.cols + k];
                        direct += delta * delta;
                    }
                    if (direct < precise_best[i] || (direct == precise_best[i] && code < ids[i])) {
                        precise_best[i] = direct;
                        ids[i] = static_cast<uint16_t>(code);
                    }
                }
            }
        }
        return ids;
    }
    void synchronize() override {}
};
} // namespace
std::unique_ptr<Ops> make_cpu() { return std::make_unique<Cpu>(); }
} // namespace omni
