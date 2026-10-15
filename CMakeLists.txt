cmake_minimum_required(VERSION 3.18)
project(bonsai_p40 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Bonsai-P40: PQ2_0 ("q2_0") compute kernels for Ternary-Bonsai-2-27B on the
# NVIDIA Tesla P40.  Targets sm_61 only -- this is a single-model,
# single-GPU engine, there is deliberately no generic fallback.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(BONSAI_BUILD_CUDA "Build the sm_61 CUDA kernels" ON)
option(BONSAI_BUILD_TESTS "Build the host verification suite" ON)

# --- host verification suite (no GPU required) ------------------------------
if(BONSAI_BUILD_TESTS)
  add_executable(test_p40 tests/test_p40.cpp)
  target_include_directories(test_p40 PRIVATE src include)
  target_compile_options(test_p40 PRIVATE -O2)
  enable_testing()
  add_test(NAME host_verification COMMAND test_p40)
endif()

# --- CUDA kernels for sm_61 -------------------------------------------------
if(BONSAI_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 61)         # Tesla P40 / GP102 only

    add_library(bonsai_p40 STATIC
      src/kernels/gemv_p40.cu
      src/kernels/gemm_p40.cu
      src/kernels/elementwise_p40.cu
      src/kernels/attn_p40.cu
      src/kernels/kv_p40.cu)
    target_include_directories(bonsai_p40 PUBLIC include PRIVATE src)
    target_compile_options(bonsai_p40 PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xptxas -dlcm=ca>)
  else()
    message(STATUS "No CUDA compiler found; building host tests only.")
  endif()
endif()
