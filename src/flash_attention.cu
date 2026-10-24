#include "flash_attention.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "flash_fwd_launch_template.h"

namespace omni {
int flash_attention_splits(size_t length) {
    if (!length || length > INT32_MAX / 4608)
        throw std::invalid_argument("Invalid FlashAttention sequence length");
    int device = 0, multiprocessors = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int n = int(length);
    const int query_blocks = 16 * ((n + 63) / 64);
    const int key_blocks = (n + 127) / 128;
    const int capacity = multiprocessors * 2;
    // Match the upstream forward dispatch, including short sequences that use split KV.
    if (query_blocks >= 0.8f * capacity)
        return 1;
    const int maximum = std::min({128, capacity, key_blocks});
    std::vector<float> efficiency;
    float best = 0;
    auto ceiling = [](int a, int b) { return (a + b - 1) / b; };
    auto eligible = [&](int split) {
        return split == 1 || ceiling(key_blocks, split) != ceiling(key_blocks, split - 1);
    };
    for (int split = 1; split <= maximum; ++split) {
        float current = 0;
        if (eligible(split)) {
            float waves = float(query_blocks * split) / capacity;
            current = waves / std::ceil(waves);
            best = std::max(best, current);
        }
        efficiency.push_back(current);
    }
    for (int split = 1; split <= maximum; ++split)
        if (eligible(split) && efficiency[split - 1] >= 0.85 * best)
            return split;
    return 1;
}

void flash_attention(void *qkv, void *output, float *lse, float *accum,
                     size_t length, int splits, cudaStream_t stream) {
    if (!qkv || !output || !lse || splits <= 0 || (splits > 1 && !accum) ||
        !length || length > INT32_MAX / 4608)
        throw std::invalid_argument("Invalid packed BF16 FlashAttention arguments");
    constexpr int heads = 16, dim = 96, width = heads * dim;
    const int n = int(length);
    flash::Flash_fwd_params params{};
    params.q_ptr = qkv;
    params.k_ptr = static_cast<uint16_t *>(qkv) + width;
    params.v_ptr = static_cast<uint16_t *>(qkv) + 2 * width;
    params.o_ptr = output;
    params.softmax_lse_ptr = lse;
    params.q_batch_stride = params.k_batch_stride = params.v_batch_stride = int64_t(n) * 3 * width;
    params.q_row_stride = params.k_row_stride = params.v_row_stride = 3 * width;
    params.q_head_stride = params.k_head_stride = params.v_head_stride = dim;
    params.o_batch_stride = int64_t(n) * width;
    params.o_row_stride = width;
    params.o_head_stride = dim;
    params.b = 1;
    params.h = params.h_k = heads;
    params.h_h_k_ratio = 1;
    params.seqlen_q = params.seqlen_k = n;
    params.seqlen_q_rounded = params.seqlen_k_rounded = (n + 127) / 128 * 128;
    params.d = params.d_rounded = dim;
    params.total_q = n;
    params.num_splits = splits;
    if (splits > 1) {
        params.softmax_lseaccum_ptr = accum;
        params.oaccum_ptr = accum + size_t(splits) * heads * n;
    }
    params.scale_softmax = float(std::pow(double(dim), -0.5));
    params.scale_softmax_log2 = params.scale_softmax * 1.4426950408889634074;
    params.p_dropout = params.rp_dropout = 1;
    params.p_dropout_in_uint8_t = 255;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.window_size_left = params.window_size_right = -1;
    params.is_bf16 = true;
    params.is_seqlens_k_cumulative = true;
    using Traits = Flash_fwd_kernel_traits<96, 128, 64, 4, false, false, cutlass::bfloat16_t>;
    if (splits > 1)
        flash::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, false>(params, stream);
    else
        flash::run_flash_fwd<Traits, false, false>(params, stream);
}
} // namespace omni
