#include "internal.h"
#include "flash_attention.h"
#include "audio_attention.h"
#include "audio_linear.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>

namespace omni {
namespace {
void check_cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw Error(e == cudaErrorMemoryAllocation ? OMNI_OUT_OF_MEMORY : OMNI_INTERNAL_ERROR,
                    std::string("CUDA: ") + cudaGetErrorString(e));
}
void check_blas(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw Error(e == CUBLAS_STATUS_ALLOC_FAILED ? OMNI_OUT_OF_MEMORY : OMNI_INTERNAL_ERROR,
                    "cuBLAS status " + std::to_string(int(e)));
}
__device__ float get(const void *p, size_t i, bool b) {
    return b ? __bfloat162float(static_cast<const __nv_bfloat16 *>(p)[i])
             : static_cast<const float *>(p)[i];
}
__device__ void put(void *p, size_t i, float v, bool b) {
    if (b)
        static_cast<__nv_bfloat16 *>(p)[i] = __float2bfloat16_rn(v);
    else
        static_cast<float *>(p)[i] = v;
}
__device__ float rnd(float x, bool b) { return b ? __bfloat162float(__float2bfloat16_rn(x)) : x; }
__device__ float sum_reduce(float x) {
    __shared__ float s[256];
    s[threadIdx.x] = x;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    float result = s[0];
    __syncthreads();
    return result;
}
__device__ float max_reduce(float x) {
    __shared__ float s[256];
    s[threadIdx.x] = x;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    float result = s[0];
    __syncthreads();
    return result;
}
__global__ void convert_kernel(const float *x, void *y, size_t n, bool b) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        put(y, i, x[i], b);
}
__global__ void image_patches_kernel(const float *pixels, void *patches, size_t height,
                                     size_t width, size_t patch, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const size_t channel = i % 8;
    float value = 0;
    if (channel < 3) {
        const size_t pixel = i / 8, patch_area = patch * patch, gw = width / patch;
        const size_t tile = pixel / patch_area, inside = pixel % patch_area;
        const size_t y = tile / gw * patch + inside / patch;
        const size_t x = tile % gw * patch + inside % patch;
        value = pixels[channel * height * width + y * width + x];
    }
    put(patches, i, value, true);
}
__global__ void download_kernel(const void *x, float *y, size_t n, bool b) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = get(x, i, b);
}
__global__ void gelu_kernel(void *x, size_t n, bool b) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = get(x, i, b);
        put(x, i, .5f * z * (1 + erff(z * .7071067811865475244f)), b);
    }
}
__global__ void add_kernel(void *x, const void *y, size_t n, bool xb, bool yb) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        put(x, i, get(x, i, xb) + get(y, i, yb), xb);
}
__global__ void bias_kernel(void *x, const void *b, size_t n, size_t d, bool bf, bool bb) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        put(x, i, get(x, i, bf) + get(b, i % d, bb), bf);
}
__global__ void norm_kernel(const void *x, void *y, const void *w, const void *b, size_t d,
                            float eps, bool bf, bool wb, bool bb) {
    size_t row = blockIdx.x;
    float sum = 0;
    for (size_t j = threadIdx.x; j < d; j += 256)
        sum += get(x, row * d + j, bf);
    float mean = sum_reduce(sum) / float(d);
    sum = 0;
    for (size_t j = threadIdx.x; j < d; j += 256) {
        float z = get(x, row * d + j, bf) - mean;
        sum += z * z;
    }
    float inv = rsqrtf(sum_reduce(sum) / float(d) + eps);
    for (size_t j = threadIdx.x; j < d; j += 256)
        put(y, row * d + j, (get(x, row * d + j, bf) - mean) * inv * get(w, j, wb) + get(b, j, bb),
            bf);
}
struct Moments {
    float mean, variance, count;
};
// Reduction order is part of the BF16 profile: small rounding changes can alter codes.
__device__ Moments merge(Moments a, Moments b) {
    float count = a.count + b.count;
    if (count == 0)
        return {0, 0, 0};
    float delta = b.mean - a.mean, inv = 1 / count, wa = a.count * inv, wb = b.count * inv;
    return {wa * a.mean + wb * b.mean, a.variance + b.variance + delta * delta * a.count * wb,
            count};
}
__global__ void norm_welford_kernel(const void *x, void *y, const void *w, const void *b, size_t d,
                                    float eps) {
    size_t row = blockIdx.x;
    unsigned tid = threadIdx.x;
    Moments m{0, 0, 0};
    for (size_t base = tid * 4; base < d; base += 512)
        for (size_t j = base; j < base + 4; ++j) {
            float z = get(x, row * d + j, true), delta = z - m.mean, count = m.count + 1,
                  mean = m.mean + delta * (1 / count);
            m = {mean, m.variance + delta * (z - mean), count};
        }
    for (int step = 16; step; step /= 2) {
        Moments other{__shfl_down_sync(0xffffffff, m.mean, step),
                      __shfl_down_sync(0xffffffff, m.variance, step),
                      __shfl_down_sync(0xffffffff, m.count, step)};
        m = merge(other, m);
    }
    __shared__ Moments warps[4];
    if (tid % 32 == 0)
        warps[tid / 32] = m;
    __syncthreads();
    for (int step = 2; step; step /= 2) {
        if (tid < unsigned(step))
            warps[tid] = merge(warps[tid + step], warps[tid]);
        __syncthreads();
    }
    float mean = warps[0].mean, inv = rsqrtf(warps[0].variance / float(d) + eps);
    for (size_t j = tid; j < d; j += 128)
        put(y, row * d + j,
            get(w, j, true) * (inv * (get(x, row * d + j, true) - mean)) + get(b, j, true), true);
}
__global__ void gather_kernel(const void *x, void *y, size_t rows, size_t d, size_t stride,
                              bool bf) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * d * 3)
        return;
    size_t t = i / (d * 3), c = (i / 3) % d, k = i % 3;
    int64_t src = int64_t(t * stride + k) - 2;
    put(y, i, src < 0 ? 0 : get(x, size_t(src) * d + c, bf), bf);
}
__global__ void pool_kernel(const void *x, void *y, size_t n, size_t rows, size_t d, size_t factor,
                            bool bf) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    size_t row = i / d, col = i % d;
    float sum = 0;
    for (size_t k = 0; k < factor && row * factor + k < rows; ++k)
        sum += get(x, (row * factor + k) * d + col, bf);
    put(y, i, sum / float(factor), bf);
}
__global__ void position_kernel(const void *x, void *y, size_t sh, size_t h, size_t w, size_t d,
                                bool bf) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= h * w * d)
        return;
    size_t row = i / d / w, col = (i / d) % w, c = i % d;
    float sy = fmaxf(0, (float(row) + .5f) * (float(sh) / float(h)) - .5f),
          sx = fmaxf(0, (float(col) + .5f) * (float(sh) / float(w)) - .5f);
    size_t y0 = size_t(sy), x0 = size_t(sx), y1 = min(y0 + 1, sh - 1), x1 = min(x0 + 1, sh - 1);
    float fy = sy - float(y0), fx = sx - float(x0);
    float a = __fmaf_rn(1 - fx, get(x, (y0 * sh + x0) * d + c, bf),
                        __fmul_rn(fx, get(x, (y0 * sh + x1) * d + c, bf)));
    float b = __fmaf_rn(1 - fx, get(x, (y1 * sh + x0) * d + c, bf),
                        __fmul_rn(fx, get(x, (y1 * sh + x1) * d + c, bf)));
    put(y, i, __fmaf_rn(1 - fy, a, __fmul_rn(fy, b)), bf);
}
__global__ void normalize_kernel(const void *x, void *y, size_t d, bool bf) {
    size_t row = blockIdx.x;
    float sum = 0;
    for (size_t j = threadIdx.x; j < d; j += 256) {
        float z = get(x, row * d + j, bf);
        sum += z * z;
    }
    float norm = fmaxf(rnd(sqrtf(sum_reduce(sum)), bf), rnd(1e-12f, bf));
    for (size_t j = threadIdx.x; j < d; j += 256)
        put(y, row * d + j, get(x, row * d + j, bf) / norm, bf);
}
__global__ void squares_kernel(const void *x, void *y, size_t d, bool bf) {
    size_t row = blockIdx.x;
    float sum = 0;
    for (size_t j = threadIdx.x; j < d; j += 256) {
        float z = get(x, row * d + j, bf);
        sum += rnd(z * z, bf);
    }
    float result = sum_reduce(sum);
    if (threadIdx.x == 0)
        put(y, row, result, bf);
}
__global__ void pack_kernel(const void *q, const void *k, const void *v, float *qp, float *kp,
                            float *vp, size_t n, size_t d, size_t heads, bool bf, bool fused,
                            float factor) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n * d)
        return;
    size_t hd = d / heads, h = i / (n * hd), t = (i / hd) % n, j = i % hd;
    size_t source = t * (fused ? 3 * d : d) + h * hd + j;
    float qv = get(q, source, bf);
    qp[i] = qv * factor;
    kp[i] = fused ? get(q, source + d, bf) * factor : get(k, t * d + h * hd + j, bf);
    vp[i] = fused ? get(q, source + 2 * d, bf) : get(v, t * d + h * hd + j, bf);
}
__global__ void softmax_kernel(float *s, size_t n, size_t tile, size_t base, size_t block,
                               size_t valid) {
    size_t row = blockIdx.x, t = row % tile + base, limit = min(n, valid);
    if (block)
        limit = min(limit, (t / block + 1) * block);
    float mx = -CUDART_INF_F;
    for (size_t j = threadIdx.x; j < limit; j += 256)
        mx = fmaxf(mx, s[row * n + j]);
    mx = max_reduce(mx);
    float total = 0;
    for (size_t j = threadIdx.x; j < n; j += 256) {
        float z = j < limit ? expf(s[row * n + j] - mx) : 0;
        s[row * n + j] = z;
        total += z;
    }
    total = sum_reduce(total);
    for (size_t j = threadIdx.x; j < n; j += 256)
        s[row * n + j] /= total;
}
__global__ void softmax_warp_kernel(float *s, size_t n, size_t rows) {
    // Match the FP32 persistent softmax used by the SDPA math reference.
    size_t row = size_t(blockIdx.x) * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows)
        return;
    float mx = -CUDART_INF_F;
    for (size_t j = lane; j < n; j += 32)
        mx = fmaxf(mx, s[row * n + j]);
    for (int step = 16; step; step /= 2)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, step));
    float sum = 0;
    for (size_t j = lane; j < n; j += 32) {
        float z = expf(s[row * n + j] - mx);
        s[row * n + j] = z;
        sum += z;
    }
    for (int step = 16; step; step /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, step);
    for (size_t j = lane; j < n; j += 32)
        s[row * n + j] /= sum;
}
template <bool maximum> __device__ float reduce_warps(float value, float *shared) {
    unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int step = 16; step; step /= 2) {
        float other = __shfl_down_sync(0xffffffff, value, step);
        value = maximum ? fmaxf(value, other) : value + other;
    }
    __syncthreads();
    if (lane == 0)
        shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = shared[lane];
        for (int step = 16; step; step /= 2) {
            float other = __shfl_down_sync(0xffffffff, value, step);
            value = maximum ? fmaxf(value, other) : value + other;
        }
        if (lane == 0)
            shared[0] = value;
    }
    __syncthreads();
    float result = shared[0];
    __syncthreads();
    return result;
}
__global__ void softmax_large_kernel(float *s, size_t n) {
    // Larger rows use sequential per-thread sums followed by two warp reductions.
    size_t row = blockIdx.x;
    __shared__ float shared[32];
    float mx = -CUDART_INF_F;
    for (size_t j = threadIdx.x; j < n; j += 1024)
        mx = fmaxf(mx, s[row * n + j]);
    mx = reduce_warps<true>(mx, shared);
    float sum = 0;
    for (size_t j = threadIdx.x; j < n; j += 1024) {
        float z = expf(s[row * n + j] - mx);
        s[row * n + j] = z;
        sum += z;
    }
    sum = reduce_warps<false>(sum, shared);
    for (size_t j = threadIdx.x; j < n; j += 1024)
        s[row * n + j] /= sum;
}
__global__ void unpack_kernel(const float *x, void *y, size_t n, size_t d, size_t heads,
                              size_t tile, size_t base, bool bf) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tile * d)
        return;
    size_t hd = d / heads, h = i / (tile * hd), t = (i / hd) % tile, j = i % hd;
    put(y, (base + t) * d + h * hd + j, x[i], bf);
}
__global__ void best_init(float *best, uint16_t *ids, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        best[i] = CUDART_INF_F;
        ids[i] = 0;
    }
}
__global__ void argmin_kernel(const void *dots, const void *xn, const void *cn, float *best,
                              uint16_t *ids, size_t tile, size_t base, bool bf) {
    const size_t row = blockIdx.x;
    const float x = get(xn, row, bf);
    float score = best[row];
    unsigned id = ids[row];
    // Each lane scans interleaved codes; equal distances retain the smallest code ID.
    for (size_t j = threadIdx.x; j < tile; j += blockDim.x) {
        float dot = get(dots, row * tile + j, bf), c = get(cn, base + j, bf);
        float dist = bf ? rnd(rnd(x + c, true) - rnd(2 * dot, true), true) : fmaf(-2.f, dot, x + c);
        unsigned candidate = unsigned(base + j);
        if (dist < score || (dist == score && candidate < id)) {
            score = dist;
            id = candidate;
        }
    }
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, score, offset);
        unsigned other_id = __shfl_down_sync(0xffffffff, id, offset);
        if (other < score || (other == score && other_id < id)) {
            score = other;
            id = other_id;
        }
    }
    __shared__ float scores[8];
    __shared__ unsigned indices[8];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (!lane) {
        scores[warp] = score;
        indices[warp] = id;
    }
    __syncthreads();
    if (!warp) {
        score = lane < 8 ? scores[lane] : CUDART_INF_F;
        id = lane < 8 ? indices[lane] : UINT32_MAX;
        for (int offset = 16; offset; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, score, offset);
            unsigned other_id = __shfl_down_sync(0xffffffff, id, offset);
            if (other < score || (other == score && other_id < id)) {
                score = other;
                id = other_id;
            }
        }
        if (!lane) {
            best[row] = score;
            ids[row] = uint16_t(id);
        }
    }
}

__global__ void refine_fp32_vq_kernel(const float *x, const float *cb, const float *dots,
                                      const float *xn, const float *cn,
                                      const float *approximate_best,
                                      const uint16_t *approximate_ids, double *best, uint16_t *ids,
                                      size_t width, size_t tile, size_t base, double error_factor) {
    const size_t row = blockIdx.x;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    double score = base ? best[row] : CUDART_INF;
    unsigned id = base ? ids[row] : 0;
    __shared__ unsigned candidates[1024], count;
    if (!threadIdx.x)
        count = 0;
    __syncthreads();
    for (size_t j = threadIdx.x; j < tile; j += blockDim.x) {
        const size_t code = base + j;
        const float distance = fmaf(-2.f, dots[row * tile + j], xn[row] + cn[code]);
        const double tolerance =
            error_factor * (2 * double(xn[row]) + cn[code] + cn[approximate_ids[row]]);
        // NaN/overflow in the approximate expression must not discard a code.
        if (!(double(distance) > double(approximate_best[row]) + tolerance))
            candidates[atomicAdd(&count, 1U)] = unsigned(code);
    }
    __syncthreads();
    for (size_t candidate = warp; candidate < count; candidate += 8) {
        const size_t code = candidates[candidate];
        double distance = 0;
        for (size_t k = lane; k < width; k += 32) {
            const double delta = double(x[row * width + k]) - cb[code * width + k];
            distance += delta * delta;
        }
        for (int step = 16; step; step >>= 1)
            distance += __shfl_down_sync(0xffffffff, distance, step);
        if (!lane && (distance < score || (distance == score && code < id))) {
            score = distance;
            id = unsigned(code);
        }
    }
    __shared__ double scores[8];
    __shared__ unsigned indices[8];
    if (!lane) {
        scores[warp] = score;
        indices[warp] = id;
    }
    __syncthreads();
    if (!warp) {
        score = lane < 8 ? scores[lane] : CUDART_INF;
        id = lane < 8 ? indices[lane] : UINT32_MAX;
        for (int step = 16; step; step >>= 1) {
            const double other = __shfl_down_sync(0xffffffff, score, step);
            const unsigned other_id = __shfl_down_sync(0xffffffff, id, step);
            if (other < score || (other == score && other_id < id)) {
                score = other;
                id = other_id;
            }
        }
        if (!lane) {
            best[row] = score;
            ids[row] = uint16_t(id);
        }
    }
}

struct Pool {
    int device;
    std::shared_ptr<Memory> memory;
    struct Cached {
        void *pointer;
        uint64_t released;
    };
    std::multimap<size_t, Cached> free;
    size_t cached = 0, active = 0, peak_active = 0, largest_request = 0;
    uint64_t release_clock = 0;
    bool in_request = false;
    std::mutex mutex;
    static constexpr size_t limit = 128 * 1024 * 1024;
    Pool(int d, std::shared_ptr<Memory> m) : device(d), memory(std::move(m)) {}
    ~Pool() {
        cudaSetDevice(device);
        for (auto &p : free) {
            cudaFree(p.second.pointer);
            memory->live.fetch_sub(p.first);
        }
    }
    void trim(size_t bytes) noexcept {
        while (cached > bytes) {
            auto oldest =
                std::min_element(free.begin(), free.end(), [](const auto &a, const auto &b) {
                    return a.second.released < b.second.released;
                });
            cudaSetDevice(device);
            cudaFree(oldest->second.pointer);
            memory->live.fetch_sub(oldest->first);
            cached -= oldest->first;
            free.erase(oldest);
        }
    }
    void begin_request() noexcept {
        std::lock_guard<std::mutex> guard(mutex);
        peak_active = active;
        largest_request = 0;
        in_request = true;
    }
    void end_request() noexcept {
        std::lock_guard<std::mutex> guard(mutex);
        in_request = false;
        trim(limit);
    }
    std::pair<void *, size_t> acquire(size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex);
        auto it = free.lower_bound(bytes);
        largest_request = std::max(largest_request, bytes);
        if (it != free.end() && it->first <= bytes + bytes / 2) {
            auto result = std::make_pair(it->second.pointer, it->first);
            cached -= it->first;
            active += it->first;
            peak_active = std::max(peak_active, active);
            free.erase(it);
            return result;
        }
        const size_t next_active = active + bytes;
        const size_t next_peak = std::max(peak_active, next_active);
        if (in_request) {
            // One largest-request reserve lets differently shaped temporaries rotate.
            // Cached bytes never feed the high-water mark, and request exit trims them.
            const size_t budget = std::max(limit, next_peak + largest_request);
            trim(budget - next_active);
        }
        void *p = nullptr;
        auto status = cudaMalloc(&p, bytes);
        if (status == cudaErrorMemoryAllocation) {
            if (cudaPeekAtLastError() == cudaErrorMemoryAllocation)
                cudaGetLastError();
            if (cached) {
                // Cached storage is optional. Reclaim it before reporting memory pressure.
                trim(0);
                p = nullptr;
                status = cudaMalloc(&p, bytes);
                if (status == cudaErrorMemoryAllocation &&
                    cudaPeekAtLastError() == cudaErrorMemoryAllocation)
                    cudaGetLastError();
            }
        }
        check_cuda(status);
        memory->add(bytes);
        active = next_active;
        peak_active = next_peak;
        return {p, bytes};
    }
    void release(void *p, size_t bytes) noexcept {
        std::lock_guard<std::mutex> guard(mutex);
        active -= bytes;
        if (in_request || bytes <= limit) {
            try {
                free.emplace(bytes, Cached{p, ++release_clock});
                cached += bytes;
                if (!in_request)
                    trim(limit);
                return;
            } catch (...) {
            }
        }
        cudaSetDevice(device);
        cudaFree(p);
        memory->live.fetch_sub(bytes);
    }
};

class Cuda final : public Ops {
    size_t fa2_length_ = 0;
    int fa2_splits_ = 1;
    int device_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    cublasLtHandle_t lt_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    cublasLtMatmulPreference_t biased_preference_ = nullptr;
    void *workspace_ = nullptr;
    static constexpr size_t workspace_bytes = 32 * 1024 * 1024;
    using Key = std::tuple<size_t, size_t, size_t, bool, bool>;
    struct MatmulPlan {
        cublasLtMatmulDesc_t op = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
        cublasLtMatmulAlgo_t algorithm{};
        uint64_t last_use = 0;
        ~MatmulPlan() {
            if (c)
                cublasLtMatrixLayoutDestroy(c);
            if (b)
                cublasLtMatrixLayoutDestroy(b);
            if (a)
                cublasLtMatrixLayoutDestroy(a);
            if (op)
                cublasLtMatmulDescDestroy(op);
        }
    };
    static constexpr size_t max_plans = 64;
    std::map<Key, std::unique_ptr<MatmulPlan>> plans_;
    uint64_t plan_clock_ = 0;
    std::shared_ptr<Pool> pool_;
    void cleanup() noexcept {
        cudaSetDevice(device_);
        if (stream_)
            cudaStreamSynchronize(stream_);
        plans_.clear();
        if (biased_preference_) {
            cublasLtMatmulPreferenceDestroy(biased_preference_);
            biased_preference_ = nullptr;
        }
        if (preference_) {
            cublasLtMatmulPreferenceDestroy(preference_);
            preference_ = nullptr;
        }
        if (workspace_) {
            cudaFree(workspace_);
            workspace_ = nullptr;
            memory->live.fetch_sub(workspace_bytes);
        }
        if (lt_) {
            cublasLtDestroy(lt_);
            lt_ = nullptr;
        }
        if (blas_) {
            cublasDestroy(blas_);
            blas_ = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }
    Value allocate(size_t rows, size_t cols, bool bf) {
        check_cuda(cudaSetDevice(device_));
        size_t bytes = rows * cols * (bf ? 2 : 4);
        if (pool_) {
            auto pool = pool_;
            auto allocation = pool->acquire(bytes);
            return {std::shared_ptr<void>(allocation.first, [pool, n = allocation.second](
                                                                void *q) { pool->release(q, n); }),
                    rows, cols, bf};
        }
        void *p = nullptr;
        check_cuda(cudaMalloc(&p, bytes));
        auto mem = memory;
        mem->add(bytes);
        int dev = device_;
        return {std::shared_ptr<void>(p,
                                      [mem, bytes, dev](void *q) {
                                          cudaSetDevice(dev);
                                          cudaFree(q);
                                          mem->live.fetch_sub(bytes);
                                      }),
                rows, cols, bf};
    }
    static unsigned blocks(size_t n) { return static_cast<unsigned>((n + 255) / 256); }
    void launch_check() { check_cuda(cudaGetLastError()); }

    MatmulPlan &matmul_plan(const Value &x, const Value &w, const Value *bias) {
        const Key key{x.rows, w.rows, x.cols, x.bf16, bias != nullptr};
        auto found = plans_.find(key);
        if (found == plans_.end()) {
            auto plan = std::make_unique<MatmulPlan>();
            const auto type = x.bf16 ? CUDA_R_16BF : CUDA_R_32F;
            const auto compute = x.bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC;
            check_blas(cublasLtMatmulDescCreate(&plan->op, compute, CUDA_R_32F));
            cublasOperation_t trans = CUBLAS_OP_T;
            check_blas(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans,
                                                      sizeof(trans)));
            if (bias) {
                auto epilogue = CUBLASLT_EPILOGUE_BIAS;
                const void *ptr = bias->data();
                check_blas(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                          &epilogue, sizeof(epilogue)));
                check_blas(cublasLtMatmulDescSetAttribute(
                    plan->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &ptr, sizeof(ptr)));
            }
            check_blas(cublasLtMatrixLayoutCreate(&plan->a, type, w.cols, w.rows, int64_t(w.cols)));
            check_blas(cublasLtMatrixLayoutCreate(&plan->b, type, x.cols, x.rows, int64_t(x.cols)));
            check_blas(cublasLtMatrixLayoutCreate(&plan->c, type, w.rows, x.rows, int64_t(w.rows)));
            cublasLtMatmulHeuristicResult_t heuristic{};
            int returned = 0;
            // Large FP32 biased linears benefit from the same bounded heuristic
            // workspace as BF16. Keep short audio linears on the wider search.
            const auto preference =
                bias && (x.bf16 || x.rows >= 512) ? biased_preference_ : preference_;
            check_blas(cublasLtMatmulAlgoGetHeuristic(lt_, plan->op, plan->a, plan->b, plan->c,
                                                      plan->c, preference, 1, &heuristic,
                                                      &returned));
            require(returned > 0, "cuBLASLt has no compatible algorithm", OMNI_UNSUPPORTED);
            plan->algorithm = heuristic.algo;
            if (plans_.size() == max_plans) {
                auto oldest = std::min_element(plans_.begin(), plans_.end(),
                                               [](const auto &a, const auto &b) {
                                                   return a.second->last_use < b.second->last_use;
                                               });
                plans_.erase(oldest);
            }
            found = plans_.emplace(key, std::move(plan)).first;
        }
        auto &plan = *found->second;
        plan.last_use = ++plan_clock_;
        // A shape is shared by different layers; never retain the previous layer's bias.
        if (bias) {
            const void *ptr = bias->data();
            check_blas(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &ptr, sizeof(ptr)));
        }
        return plan;
    }

  public:
    explicit Cuda(int device) : device_(device) {
        try {
            check_cuda(cudaSetDevice(device));
            check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            check_blas(cublasCreate(&blas_));
            check_blas(cublasSetStream(blas_, stream_));
            check_blas(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH));
            check_blas(cublasLtCreate(&lt_));
            check_blas(cublasLtMatmulPreferenceCreate(&preference_));
            check_blas(cublasLtMatmulPreferenceCreate(&biased_preference_));
            // Match the original BF16 biased linear path, including its reduction choices.
            size_t biased_ws = 1024 * 1024;
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                biased_preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &biased_ws,
                sizeof(biased_ws)));
            size_t ws = workspace_bytes;
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
            uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            check_blas(cublasLtMatmulPreferenceSetAttribute(
                preference_, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                sizeof(reduction)));
            check_cuda(cudaMalloc(&workspace_, workspace_bytes));
            memory->add(workspace_bytes);
        } catch (...) {
            cleanup();
            throw;
        }
    }
    void enable_reuse() override { pool_ = std::make_shared<Pool>(device_, memory); }
    void begin_request() noexcept override {
        if (pool_)
            pool_->begin_request();
    }
    void end_request() noexcept override {
        if (pool_)
            pool_->end_request();
    }
    ~Cuda() override { cleanup(); }
    Value raw_upload(const void *x, size_t rows, size_t cols, bool b) override {
        auto y = allocate(rows, cols, b);
        check_cuda(
            cudaMemcpyAsync(y.data(), x, y.size() * (b ? 2 : 4), cudaMemcpyHostToDevice, stream_));
        synchronize();
        return y;
    }
    Value upload(const float *x, size_t rows, size_t cols, bool b) override {
        if (!b)
            return raw_upload(x, rows, cols, false);
        auto fp = raw_upload(x, rows, cols, false);
        auto y = allocate(rows, cols, b);
        convert_kernel<<<blocks(y.size()), 256, 0, stream_>>>(static_cast<float *>(fp.data()),
                                                              y.data(), y.size(), b);
        launch_check();
        return y;
    }
    Value image_patches(const float *pixels, size_t height, size_t width, size_t patch) override {
        auto input = raw_upload(pixels, 3, height * width, false);
        auto output = allocate((height / patch) * (width / patch), 8 * patch * patch, true);
        image_patches_kernel<<<blocks(output.size()), 256, 0, stream_>>>(
            static_cast<const float *>(input.data()), output.data(), height, width, patch,
            output.size());
        launch_check();
        return output;
    }
    std::vector<float> download(const Value &x) override {
        check_cuda(cudaSetDevice(device_));
        std::vector<float> y(x.size());
        if (!x.bf16) {
            check_cuda(
                cudaMemcpyAsync(y.data(), x.data(), y.size() * 4, cudaMemcpyDeviceToHost, stream_));
        } else {
            auto fp = allocate(x.rows, x.cols, false);
            download_kernel<<<blocks(x.size()), 256, 0, stream_>>>(
                x.data(), static_cast<float *>(fp.data()), x.size(), true);
            launch_check();
            check_cuda(cudaMemcpyAsync(y.data(), fp.data(), y.size() * 4, cudaMemcpyDeviceToHost,
                                       stream_));
            synchronize();
        }
        synchronize();
        return y;
    }
    Value linear(const Value &x, const Value &w, const Value *b) override {
        require(x.cols == w.cols && x.bf16 == w.bf16, "CUDA linear shape/dtype mismatch",
                OMNI_INVALID_MODEL);
        require(!b || b->bf16 == x.bf16, "CUDA bias dtype mismatch", OMNI_INVALID_MODEL);
        auto y = allocate(x.rows, w.rows, x.bf16);
        if (!x.bf16 && b && x.rows >= 256 &&
            audio_linear(static_cast<const float *>(x.data()),
                         static_cast<const float *>(w.data()),
                         static_cast<const float *>(b->data()), static_cast<float *>(y.data()),
                         x.rows, w.rows, x.cols, stream_))
            return y;
        auto &plan = matmul_plan(x, w, b);
        float alpha = 1, beta = 0;
        check_blas(cublasLtMatmul(lt_, plan.op, &alpha, w.data(), plan.a, x.data(), plan.b, &beta,
                                  y.data(), plan.c, y.data(), plan.c, &plan.algorithm, workspace_,
                                  workspace_bytes, stream_));
        return y;
    }
    Value norm(const Value &x, const Value &w, const Value &b, float eps) override {
        auto y = allocate(x.rows, x.cols, x.bf16);
        if (x.bf16 && x.cols % 4 == 0)
            norm_welford_kernel<<<unsigned(x.rows), 128, 0, stream_>>>(x.data(), y.data(), w.data(),
                                                                       b.data(), x.cols, eps);
        else
            norm_kernel<<<unsigned(x.rows), 256, 0, stream_>>>(
                x.data(), y.data(), w.data(), b.data(), x.cols, eps, x.bf16, w.bf16, b.bf16);
        launch_check();
        return y;
    }
    void gelu(Value &x) override {
        gelu_kernel<<<blocks(x.size()), 256, 0, stream_>>>(x.data(), x.size(), x.bf16);
        launch_check();
    }
    void add(Value &x, const Value &y) override {
        require(x.cols == y.cols && y.rows >= x.rows, "CUDA residual dimensions disagree",
                OMNI_INTERNAL_ERROR);
        add_kernel<<<blocks(x.size()), 256, 0, stream_>>>(x.data(), y.data(), x.size(), x.bf16,
                                                          y.bf16);
        launch_check();
    }
    void bias(Value &x, const Value &b) override {
        require(b.size() == x.cols, "CUDA bias dimensions disagree", OMNI_INTERNAL_ERROR);
        bias_kernel<<<blocks(x.size()), 256, 0, stream_>>>(x.data(), b.data(), x.size(), x.cols,
                                                           x.bf16, b.bf16);
        launch_check();
    }
    Value conv_gather(const Value &x, size_t stride) override {
        auto y = allocate((x.rows + stride - 1) / stride, x.cols * 3, x.bf16);
        gather_kernel<<<blocks(y.size()), 256, 0, stream_>>>(x.data(), y.data(), y.rows, x.cols,
                                                             stride, x.bf16);
        launch_check();
        return y;
    }
    Value attention(const Value &q, const Value *k, const Value *v, size_t heads, size_t block,
                    size_t valid, bool fused) override {
        const size_t n = q.rows, d = fused ? q.cols / 3 : q.cols, hd = d / heads;
        auto y = allocate(n, d, q.bf16);
        if (!fused && !q.bf16 && hd == 64) {
            audio_attention(static_cast<const float *>(q.data()),
                            static_cast<const float *>(k->data()),
                            static_cast<const float *>(v->data()), static_cast<float *>(y.data()),
                            n, heads, block, valid, stream_);
            return y;
        }
        if (fused && q.bf16 && heads == 16 && hd == 96) {
            if (fa2_length_ != n) {
                fa2_splits_ = flash_attention_splits(n);
                fa2_length_ = n;
            }
            auto lse = allocate(heads, n, false);
            Value accum;
            if (fa2_splits_ > 1)
                accum = allocate(size_t(fa2_splits_) * heads, n * (hd + 1), false);
            flash_attention(q.data(), y.data(), static_cast<float *>(lse.data()),
                            fa2_splits_ > 1 ? static_cast<float *>(accum.data()) : nullptr, n,
                            fa2_splits_, stream_);
            return y;
        }

        // Larger FP32 audio tiles give the small per-head GEMMs enough parallel
        // work, while keeping score storage bounded independently of length.
        const size_t max_tile = fused ? (n >= 256 ? 256 : 128) : 512;
        auto qp = allocate(n, d, false), kp = allocate(n, d, false), vp = allocate(n, d, false);
        auto scores = allocate(heads * std::min(max_tile, n), n, false),
             output = allocate(heads * std::min(max_tile, n), hd, false);
        pack_kernel<<<blocks(n * d), 256, 0, stream_>>>(
            q.data(), k ? k->data() : nullptr, v ? v->data() : nullptr,
            static_cast<float *>(qp.data()), static_cast<float *>(kp.data()),
            static_cast<float *>(vp.data()), n, d, heads, q.bf16, fused,
            fused ? float(std::pow(double(hd), -.25)) : 1 / std::sqrt(float(hd)));
        launch_check();
        for (size_t base = 0; base < n;) {
            size_t tile = std::min(max_tile, n - base);
            // Preserve the original GEMM shape for a partial final tile.
            if (fused && tile > 128 && tile < 256)
                tile = 128;
            // Omit only future blocks shared by every query in this tile.
            // Keep valid-key padding and padded queries in the numerical path.
            const size_t keys =
                !fused && block ? std::min(n, ((base + tile - 1) / block + 1) * block) : n;
            float alpha = 1, beta = 0;
            check_blas(cublasSgemmStridedBatched(
                blas_, CUBLAS_OP_T, CUBLAS_OP_N, int(keys), int(tile), int(hd), &alpha,
                static_cast<float *>(kp.data()), int(hd), n * hd,
                static_cast<float *>(qp.data()) + base * hd, int(hd), n * hd, &beta,
                static_cast<float *>(scores.data()), int(keys), tile * keys, int(heads)));
            if (fused && n >= 32 && n <= 2048)
                softmax_warp_kernel<<<unsigned((heads * tile + 3) / 4), 128, 0, stream_>>>(
                    static_cast<float *>(scores.data()), n, heads * tile);
            else if (fused && n > 2048)
                softmax_large_kernel<<<unsigned(heads * tile), 1024, 0, stream_>>>(
                    static_cast<float *>(scores.data()), n);
            else
                softmax_kernel<<<unsigned(heads * tile), 256, 0, stream_>>>(
                    static_cast<float *>(scores.data()), keys, tile, base, block, valid);
            launch_check();
            alpha = 1;
            check_blas(cublasSgemmStridedBatched(
                blas_, CUBLAS_OP_N, CUBLAS_OP_N, int(hd), int(tile), int(keys), &alpha,
                static_cast<float *>(vp.data()), int(hd), n * hd,
                static_cast<float *>(scores.data()), int(keys), tile * keys, &beta,
                static_cast<float *>(output.data()), int(hd), tile * hd, int(heads)));
            unpack_kernel<<<blocks(tile * d), 256, 0, stream_>>>(
                static_cast<float *>(output.data()), y.data(), n, d, heads, tile, base, q.bf16);
            launch_check();
            base += tile;
        }
        return y;
    }
    Value pool(const Value &x, size_t f) override {
        auto y = allocate((x.rows + f - 1) / f, x.cols, x.bf16);
        pool_kernel<<<blocks(y.size()), 256, 0, stream_>>>(x.data(), y.data(), y.size(), x.rows,
                                                           x.cols, f, x.bf16);
        launch_check();
        return y;
    }
    Value interpolate(const Value &x, size_t s, size_t h, size_t w) override {
        auto y = allocate(h * w, x.cols, x.bf16);
        position_kernel<<<blocks(y.size()), 256, 0, stream_>>>(x.data(), y.data(), s, h, w, x.cols,
                                                               x.bf16);
        launch_check();
        return y;
    }
    Value normalize(const Value &x) override {
        auto y = allocate(x.rows, x.cols, x.bf16);
        normalize_kernel<<<unsigned(x.rows), 256, 0, stream_>>>(x.data(), y.data(), x.cols, x.bf16);
        launch_check();
        return y;
    }
    Value squares(const Value &x) override {
        auto y = allocate(x.rows, 1, x.bf16);
        squares_kernel<<<unsigned(x.rows), 256, 0, stream_>>>(x.data(), y.data(), x.cols, x.bf16);
        launch_check();
        return y;
    }
    std::vector<uint16_t> quantize(const Value &x, const Value &cb, const Value &cn,
                                   std::atomic<bool> &cancel) override {
        auto xn = squares(x), best = allocate(x.rows, 1, false), ids = allocate(x.rows, 1, true);
        Value precise_best, precise_ids;
        if (!x.bf16) {
            precise_best = allocate(x.rows, 2, false); // One FP64 distance per query.
            precise_ids = allocate(x.rows, 1, true);
        }
        const double error_factor = fp32_distance_error_factor(x.cols);
        best_init<<<blocks(x.rows), 256, 0, stream_>>>(static_cast<float *>(best.data()),
                                                       static_cast<uint16_t *>(ids.data()), x.rows);
        launch_check();
        for (size_t base = 0; base < cb.rows; base += 1024) {
            if (cancel.load())
                throw Error(OMNI_CANCELLED, "Inference cancelled");
            size_t tile = std::min(size_t(1024), cb.rows - base);
            char *ptr = static_cast<char *>(cb.data()) + base * cb.cols * (cb.bf16 ? 2 : 4);
            Value part{std::shared_ptr<void>(cb.storage, ptr), tile, cb.cols, cb.bf16};
            auto dots = linear(x, part, nullptr);
            argmin_kernel<<<unsigned(x.rows), 256, 0, stream_>>>(
                dots.data(), xn.data(), cn.data(), static_cast<float *>(best.data()),
                static_cast<uint16_t *>(ids.data()), tile, base, x.bf16);
            launch_check();
            if (!x.bf16) {
                refine_fp32_vq_kernel<<<unsigned(x.rows), 256, 0, stream_>>>(
                    static_cast<const float *>(x.data()), static_cast<const float *>(cb.data()),
                    static_cast<const float *>(dots.data()), static_cast<const float *>(xn.data()),
                    static_cast<const float *>(cn.data()), static_cast<const float *>(best.data()),
                    static_cast<const uint16_t *>(ids.data()),
                    static_cast<double *>(precise_best.data()),
                    static_cast<uint16_t *>(precise_ids.data()), x.cols, tile, base, error_factor);
                launch_check();
            }
        }
        std::vector<uint16_t> out(x.rows);
        check_cuda(cudaMemcpyAsync(out.data(), x.bf16 ? ids.data() : precise_ids.data(),
                                   out.size() * 2, cudaMemcpyDeviceToHost, stream_));
        synchronize();
        return out;
    }
    void synchronize() override {
        check_cuda(cudaSetDevice(device_));
        check_cuda(cudaStreamSynchronize(stream_));
    }
};
} // namespace
std::unique_ptr<Ops> make_cuda(int device) { return std::make_unique<Cuda>(device); }
} // namespace omni
