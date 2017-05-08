#include "gleeman/exception.hpp"

namespace gleeman {

GleemanError::GleemanError(std::string what) : std::runtime_error(what.c_str()) {}

NoNVMLError::NoNVMLError(std::string what) : GleemanError(what) {}

[[noreturn]] void throw_no_nvml() {
  throw NoNVMLError("Not compiled with USE_NVML");
}

}
