#ifndef GLEEMAN_UNIFORM_HPP
#define GLEEMAN_UNIFORM_HPP

#include <cstddef>

#include "gleeman/cuda_headers.hpp"

namespace gleeman {

enum APIType { Driver, CUDARuntime, NVML, Universal };

template<APIType>
struct API;

#ifdef USE_CUDA

template<>
struct API<Driver> {
  static constexpr size_t index = 0;
  using error_type = CUresult;
};

template<>
struct API<CUDARuntime> {
  static constexpr size_t index = 1;
  using error_type = cudaError_t;
};

#ifdef USE_NVML

template<>
struct API<NVML> {
  static constexpr size_t index = 2;
  using error_type = nvmlReturn_t;
};

#endif //USE_NVML
#endif //USE_CUDA

template<>
struct API<Universal> {
  static constexpr size_t index = 4;
  using error_type = int;
};

} //namespace gleeman

#endif //GLEEMAN_UNIFORM_HPP
