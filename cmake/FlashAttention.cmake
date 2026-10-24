include(FetchContent)
FetchContent_Declare(omni_fa2
  URL https://codeload.github.com/Dao-AILab/flash-attention/zip/060c9188beec3a8b62b33a3bfa6d5d2d44975fab
  URL_HASH SHA256=5e4ad0755cb4e049487c713bda78b3eeccef6cf288c3aaaa4f99b63da70afcf4
  SOURCE_SUBDIR omni-unused)
FetchContent_Declare(omni_cutlass
  URL https://codeload.github.com/NVIDIA/cutlass/zip/dc4817921edda44a549197ff3a9dcf5df0636e7b
  URL_HASH SHA256=9c0acc1f93ddcb734aaae7c277cfe0c1b340499ee156c394ae44fb070d7cdd48
  SOURCE_SUBDIR omni-unused)
FetchContent_MakeAvailable(omni_fa2 omni_cutlass)

# Compile only the BF16 forward kernel used by VQSigLIP. No Torch runtime is linked.
add_library(omni_flash_attention OBJECT ${PROJECT_SOURCE_DIR}/src/flash_attention.cu)
target_include_directories(omni_flash_attention PRIVATE
  ${PROJECT_SOURCE_DIR}/src/flash_compat
  ${omni_fa2_SOURCE_DIR}/csrc/flash_attn/src
  ${omni_cutlass_SOURCE_DIR}/include)
target_link_libraries(omni_flash_attention PRIVATE CUDA::cudart)
target_compile_definitions(omni_flash_attention PRIVATE
  FLASHATTENTION_DISABLE_DROPOUT FLASHATTENTION_DISABLE_ALIBI
  FLASHATTENTION_DISABLE_LOCAL FLASHATTENTION_DISABLE_SOFTCAP
  FLASHATTENTION_DISABLE_UNEVEN_K)
target_compile_options(omni_flash_attention PRIVATE
  --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
  --diag-suppress=177,221,550)
if(MSVC)
  target_compile_options(omni_flash_attention PRIVATE -Xcompiler=/Zc:__cplusplus)
  target_compile_definitions(omni_flash_attention PRIVATE _USE_MATH_DEFINES)
else()
  target_compile_options(omni_flash_attention PRIVATE -Xcompiler=-fvisibility=hidden)
endif()
set_target_properties(omni_flash_attention PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
  CUDA_RUNTIME_LIBRARY Shared
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
