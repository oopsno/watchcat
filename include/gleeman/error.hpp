#ifndef GLEEMAN_ERROR_HPP
#define GLEEMAN_ERROR_HPP

#include <tuple>
#include <stdexcept>

#include "gleeman/universal.hpp"
#include "gleeman/exception.hpp"

namespace gleeman {

template<typename Error>
struct error_traits;

template<>
struct error_traits<CUresult> {
  static const CUresult success;
  static std::string what(const CUresult error);
};

template<>
struct error_traits<cudaError_t> {
  static const cudaError_t success;
  static std::string what(const cudaError_t error);
};

template<>
struct error_traits<nvmlReturn_t> {
  static const nvmlReturn_t success;
  static std::string what(const nvmlReturn_t error);
};

struct UniformedError {
  inline UniformedError() : driver_error(CUDA_SUCCESS), runtime_error(cudaSuccess), nvml_error(NVML_SUCCESS) {}

  inline CUresult operator=(CUresult error) {
    return driver_error = error;
  }

  inline cudaError_t operator=(cudaError_t error) {
    return runtime_error = error;
  }

  inline nvmlReturn_t operator=(nvmlReturn_t error) {
    return nvml_error = error;
  }

  bool operator==(const CUresult error) const {
    return driver_error == error;
  }

  bool operator==(const cudaError_t error) const {
    return runtime_error == error;
  }

  bool operator==(const nvmlReturn_t error) const {
    return nvml_error == error;
  }

  bool operator==(const UniformedError &error) const {
    return driver_error == error.driver_error && runtime_error == error.runtime_error && nvml_error == error.nvml_error;
  }

  template<typename Error, typename Traits=error_traits<Error>>
  inline UniformedError &handle(Error error) {
    *this = error;
    if (error != Traits::success) {
      throw GleemanError(Traits::what(error));
    }
    return *this;
  }

  template<typename Error, typename=error_traits<Error>>
  inline UniformedError &operator<<(Error error) {
    return handle(error);
  }

  CUresult driver_error;
  cudaError_t runtime_error;
  nvmlReturn_t nvml_error;
};

extern UniformedError defaultErrorHandler;

}

#endif //GLEEMAN_ERROR_HPP
