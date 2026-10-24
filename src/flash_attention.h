#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace omni {
int flash_attention_splits(size_t length);
void flash_attention(void *qkv, void *output, float *lse, float *accum,
                     size_t length, int splits, cudaStream_t stream);
} // namespace omni
