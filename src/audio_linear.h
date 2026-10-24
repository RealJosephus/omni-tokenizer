#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace omni {
bool audio_linear(const float *input, const float *weight, const float *bias, float *output,
                  size_t rows, size_t columns, size_t width, cudaStream_t stream);
}
