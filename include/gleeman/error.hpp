#ifndef GLEEMAN_ERROR_HPP
#define GLEEMAN_ERROR_HPP

#include "gleeman/universal.hpp"
#include "gleeman/exception.hpp"

namespace gleeman {

template<typename Error>
struct error_traits;

#define STUB_ERROR_TRAITS(api)                       \
  template<>                                         \
  struct error_traits<API<api>::error_type> {        \
    using error_type = API<api>::error_type;         \
    static const error_type success;                 \
    static std::string what(const error_type error); \
  };

#ifdef USE_CUDA
STUB_ERROR_TRAITS(Driver)
STUB_ERROR_TRAITS(CUDARuntime)
#ifdef USE_NVML
STUB_ERROR_TRAITS(NVML)
#endif //USE_NVML
#endif //USE_CUDA
STUB_ERROR_TRAITS(Universal)

#undef STUB_ERROR_TRAITS

struct UniversalErrorHandler {
  UniversalErrorHandler();

  template<typename Error>
  Error as();

  template<typename Error>
  Error operator=(Error error);

  template<typename Error>
  bool operator==(const Error error) const {
    return this->as<Error>() == error;
  }

  bool operator==(const UniversalErrorHandler &error) const = delete;

  template<typename Error, typename Traits = error_traits<Error>>
  inline UniversalErrorHandler &handle(Error error) {
    *this = error;
    if (error != Traits::success) {
      throw GleemanError(Traits::what(error));
    }
    return *this;
  }

  template<typename Error, typename = error_traits<Error>>
  inline UniversalErrorHandler &operator<<(Error error) {
    return handle(error);
  }

#define STUB_FIELD(api, name) API<api>::error_type name;
#ifdef USE_CUDA
  STUB_FIELD(Driver, driver_error)
  STUB_FIELD(CUDARuntime, runtime_error)
#ifdef USE_NVML
  STUB_FIELD(NVML, nvml_error)
#endif //USE_NVML
#endif //USE_CUDA
  STUB_FIELD(Universal, universal_error)
#undef STUB_FIELD
};

extern gleeman::UniversalErrorHandler defaultErrorHandler;

} //namespace gleeman

#endif //GLEEMAN_ERROR_HPP
