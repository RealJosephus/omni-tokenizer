#pragma once
#include <tuple>
#include <ATen/cuda/CUDAGeneratorImpl.h>
namespace at { namespace cuda { namespace philox {
__host__ __device__ inline std::tuple<uint64_t, uint64_t> unpack(const at::PhiloxCudaState &state) {
    return {state.seed, state.offset};
}
}}}
