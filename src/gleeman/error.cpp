#include "gleeman/error.hpp"
#include "gleeman/uniform.hpp"

namespace gleeman {

const CUresult error_traits<CUresult>::success = CUDA_SUCCESS;
const cudaError_t error_traits<cudaError_t>::success = cudaSuccess;

std::string error_traits<CUresult>::what(const CUresult error) {
  const char *name, *what_;
  CUresult internal_error;
  internal_error = cuGetErrorName(error, &name);
  if (internal_error != CUDA_SUCCESS) {
    return what(internal_error);
  }
  internal_error = cuGetErrorString(error, &what_);
  if (internal_error != CUDA_SUCCESS) {
    return what(internal_error);
  }
  return std::string(name) + ": " + std::string(what_);
}

std::string error_traits<cudaError_t>::what(const cudaError_t error) {
  std::string name = cudaGetErrorName(error);
  std::string what = cudaGetErrorString(error);
  return name + ": " + what;
}

const nvmlReturn_t error_traits<nvmlReturn_t>::success = NVML_SUCCESS;
std::string error_traits<nvmlReturn_t>::what(const nvmlReturn_t error) {
#ifdef USE_NVML
  return nvmlErrorString(error);
#else
  throw_no_nvml();
#endif
}

}
