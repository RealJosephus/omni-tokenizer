#include "audio_attention.h"
#include <algorithm>
#include <stdexcept>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

namespace omni {
using AudioKernel = PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64,
                                                            64, 64, false, false>;

void audio_attention(const float *q, const float *k, const float *v, float *output, size_t length,
                     size_t heads, size_t block, size_t valid, cudaStream_t stream) {
    if (!q || !k || !v || !output || !length || !heads || !valid)
        throw std::invalid_argument("Invalid audio attention arguments");
    AudioKernel::Params params;
    params.head_dim = params.head_dim_value = 64;
    params.num_heads = int(heads);
    params.num_batches = 1;
    params.scale = 0.125f;
    params.q_strideH = params.k_strideH = params.v_strideH = 64;
    params.q_strideM = params.k_strideM = params.v_strideM = params.o_strideM = int(heads * 64);
    const size_t step = block ? block : length;
    for (size_t base = 0; base < length; base += step) {
        params.query_ptr = q + base * heads * 64;
        params.key_ptr = k;
        params.value_ptr = v;
        params.output_ptr = output + base * heads * 64;
        params.num_queries = int(std::min(step, length - base));
        // Every query in this block can see the whole valid prefix, including
        // earlier blocks. Padded query rows still produce outputs for pooling.
        params.num_keys = int(std::min({valid, length, base + step}));
        const auto kernel = PyTorchMemEffAttention::attention_kernel_batched_impl<AudioKernel>;
        kernel<<<params.getBlocksGrid(), params.getThreadsGrid(),
                 sizeof(AudioKernel::SharedStorage), stream>>>(params);
    }
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace omni
