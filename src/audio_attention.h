#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace omni {
void audio_attention(const float *q, const float *k, const float *v, float *output, size_t length,
                     size_t heads, size_t block, size_t valid, cudaStream_t stream);
}
