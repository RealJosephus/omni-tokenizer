#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
inline void standalone_cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
#define C10_CUDA_CHECK(call) standalone_cuda_check(call)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() standalone_cuda_check(cudaGetLastError())
