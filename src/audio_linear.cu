#include "audio_linear.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <stdexcept>

namespace omni {
// Keep FP32 storage and accumulation. Three Tensor Core products recover the
// high/low components instead of rounding each operand to a single TF32 value.
using AudioGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 4, 4, false,
    cutlass::arch::OpMultiplyAddFastF32>;

bool audio_linear(const float *input, const float *weight, const float *bias, float *output,
                  size_t rows, size_t columns, size_t width, cudaStream_t stream) {
    AudioGemm::Arguments args{{int(rows), int(columns), int(width)},
                              {input, int(width)},
                              {weight, int(width)},
                              {bias, 0},
                              {output, int(columns)},
                              {1.f, 1.f}};
    if (AudioGemm::can_implement(args) != cutlass::Status::kSuccess)
        return false;
    AudioGemm operation;
    const auto status = operation(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(cutlassGetStatusString(status));
    return true;
}
} // namespace omni
