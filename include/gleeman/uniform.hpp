#ifndef GLEEMAN_UNIFORM_HPP
#define GLEEMAN_UNIFORM_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

namespace gleeman {

enum APIType {
  Driver, CUDARuntime, NVML
};

template<APIType>
struct API;

template<>
struct API<Driver> {
  static constexpr size_t index = 0;
  using error_type = CUresult;
};
constexpr size_t API<Driver>::index;

template<>
struct API<CUDARuntime> {
  static constexpr size_t index = 1;
  using error_type = cudaError_t;
};
constexpr size_t API<CUDARuntime>::index;

template<>
struct API<NVML> {
  static constexpr size_t index = 2;
  using error_type = nvmlReturn_t;
};
constexpr size_t API<NVML>::index;

[[noreturn]] void throw_no_nvml();

}

#endif //GLEEMAN_UNIFORM_HPP
