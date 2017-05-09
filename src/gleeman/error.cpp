#include <string>

#include "gleeman/error.hpp"

namespace gleeman {

UniversalErrorHandler defaultErrorHandler;

#ifdef USE_CUDA
const CUresult error_traits<CUresult>::success = CUDA_SUCCESS;
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

const cudaError_t error_traits<cudaError_t>::success = cudaSuccess;
std::string error_traits<cudaError_t>::what(const cudaError_t error) {
  std::string name = cudaGetErrorName(error);
  std::string what = cudaGetErrorString(error);
  return name + ": " + what;
}

#ifdef USE_NVML
const nvmlReturn_t error_traits<nvmlReturn_t>::success = nvml_success;
std::string error_traits<nvmlReturn_t>::what(const nvmlReturn_t error) {
  return nvmlErrorString(error);
}
#endif //USE_NVML
#endif //USE_CUDA

const int error_traits<API<Universal>::error_type>::success = 0;
std::string error_traits<API<Universal>::error_type>::what(
    const API<Universal>::error_type error) {
  return "ERROR CODE: " + std::to_string(error);
}

UniversalErrorHandler::UniversalErrorHandler() :
#ifdef USE_CUDA
      driver_error(CUDA_SUCCESS),
      runtime_error{cudaSuccess},
#ifdef USE_NVML
      nvml_error(nvml_success),
#endif //USE_CUDA
#endif //USE_NVML
      universal_error(0) {}

#define IMPL_FIELD_TPL(name)                                                 \
  template<>                                                                 \
  decltype(UniversalErrorHandler::name)                                      \
  UniversalErrorHandler::as<decltype(UniversalErrorHandler::name)>() const { \
    return UniversalErrorHandler::name;                                      \
  }                                                                          \
  template<>                                                                 \
  decltype(UniversalErrorHandler::name) UniversalErrorHandler::operator=(    \
      decltype(UniversalErrorHandler::name) error) {                         \
    return UniversalErrorHandler::name = error;                              \
  }

#ifdef USE_CUDA
IMPL_FIELD_TPL(driver_error)
IMPL_FIELD_TPL(runtime_error)
#ifdef USE_NVML
IMPL_FIELD_TPL(nvml_error)
#endif //USE_NVML
#endif //USE_CUDA
IMPL_FIELD_TPL(universal_error)
#undef IMPL_FIELD_TPL
}
