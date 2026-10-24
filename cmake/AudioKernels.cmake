# Fetch only the standalone forward headers, with content hashes pinned.
set(OMNI_AUDIO_ATTENTION_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/omni_audio_attention-src"
  CACHE PATH "Directory containing the pinned memory-efficient attention headers")
set(_omni_audio_prefix "ATen/native/transformers/cuda/mem_eff_attention/")
set(_omni_audio_url "https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/aten/src")
foreach(_omni_entry IN ITEMS
  "debug_utils.h|0bc7029c77ea42c8144e92814a7417c7b49acb4dee002fba0763a35a7ee27c02"
  "epilogue/epilogue_pipelined.h|220dcb8e76c6f95339f9ef8630aae000a2c65fc003f7063d0c1d2da7df5ef8d8"
  "epilogue/epilogue_rescale_output.h|ce0b13ecf73c91c92e9f78c7b82851ee0fa4802d8594a3a60c73bb46e74dfc79"
  "epilogue/epilogue_thread_apply_logsumexp.h|c5ca7711ae4f94cd2b1c3cd8275114b5740b804748ca4d08c501bed9f42f6941"
  "gemm/custom_mma.h|5bdb4c6606d73a76fd740e762dcda90b175039d5e13cec18e1b074e1e9179397"
  "gemm/custom_mma_base.h|9281f07cebb654fb3107a4150a9b840a54a0318957ec8c4c9535586d6369b6c1"
  "gemm/custom_mma_multistage.h|4d1d0d844bf46786f8fe2f8c0d240c8f7b5428b1c1628e3ec1c1808c47f52021"
  "gemm/custom_mma_pipelined.h|c603283977dac8f7195b5e1c2a6ebab069e628eee0fdc525e53ae042a2f45768"
  "gemm/find_default_mma.h|223613991c96520b6dbeeefeb160b948483f80cb60800e81f184c22a0d335994"
  "gemm/mma_accum_lambda_iterator.h|ec666626f49dd4b6570a3f284735bf38bbbd5d56b53c49ffd3ff1f9345462784"
  "gemm/mma_from_smem.h|aa18d0eb6812037d128471464a20887192c84193e8f78396b8d3b1a3442abce8"
  "gemm_kernel_utils.h|ce914b76b79d92c487a44cfc3ed19f41da947d0354672ab716489091fb0a7cbf"
  "iterators/default_warp_iterator_from_smem.h|c46d8c01507a4a049f784545379b323194750f9b34b3400a020f2ccc2bb912d8"
  "iterators/make_residual_last.h|17f4e1b6e4b473e21e106d2d4f71a07196a6849d41be8769c16fd2db43a0e666"
  "iterators/predicated_tile_access_iterator_residual_last.h|d17ec394229e52e693cea853694f0278f813624dc7cd497e2a2ace515d1d9ff6"
  "iterators/predicated_tile_iterator_residual_last.h|880a3383d716b50f5527564d01e066a8eabeace93a534562065c55fa4faaf220"
  "iterators/transpose_warp_iterator.h|8eb466e7e3013da32d0a083a3e5df978d7983b83a2dd2a5e362c526892ae894e"
  "iterators/warp_iterator_from_smem.h|3e97f7b3eae7cb5c21fb9b18761a6472b0b1b317923508c6a0cfcf8240983258"
  "kernel_forward.h|fd75c66489fca2ddc07cdc22eedf2b15802684e908568ed845a8f01270b05b53"
  "transform/tile_smem_loader.h|798f55c4a47fadbaad11316d7a175f1063f63919069e034cf67debc946a2ecb0")
  string(REPLACE "|" ";" _omni_parts "${_omni_entry}")
  list(GET _omni_parts 0 _omni_name)
  list(GET _omni_parts 1 _omni_hash)
  set(_omni_relative "${_omni_audio_prefix}${_omni_name}")
  set(_omni_path "${OMNI_AUDIO_ATTENTION_SOURCE_DIR}/${_omni_relative}")
  set(_omni_actual "")
  if(EXISTS "${_omni_path}")
    file(SHA256 "${_omni_path}" _omni_actual)
  endif()
  if(NOT _omni_actual STREQUAL _omni_hash)
    get_filename_component(_omni_parent "${_omni_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${_omni_parent}")
    file(DOWNLOAD "${_omni_audio_url}/${_omni_relative}" "${_omni_path}"
      EXPECTED_HASH "SHA256=${_omni_hash}" TLS_VERIFY ON)
  endif()
endforeach()

add_library(omni_audio_kernels OBJECT
  ${PROJECT_SOURCE_DIR}/src/audio_attention.cu ${PROJECT_SOURCE_DIR}/src/audio_linear.cu)
target_include_directories(omni_audio_kernels PRIVATE
  ${PROJECT_SOURCE_DIR}/src/flash_compat
  ${OMNI_AUDIO_ATTENTION_SOURCE_DIR}
  ${omni_cutlass_SOURCE_DIR}/include)
target_link_libraries(omni_audio_kernels PRIVATE CUDA::cudart)
target_compile_definitions(omni_audio_kernels PRIVATE FLASHATTENTION_DISABLE_DROPOUT)
target_compile_options(omni_audio_kernels PRIVATE
  --expt-relaxed-constexpr --expt-extended-lambda --diag-suppress=177,221,550)
if(MSVC)
  target_compile_options(omni_audio_kernels PRIVATE -Xcompiler=/Zc:__cplusplus)
else()
  target_compile_options(omni_audio_kernels PRIVATE -Xcompiler=-fvisibility=hidden)
endif()
set_target_properties(omni_audio_kernels PROPERTIES
  POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
  CUDA_RUNTIME_LIBRARY Shared CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
