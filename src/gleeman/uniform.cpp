#include <exception>

#include "gleeman/exception.hpp"
#include "gleeman/uniform.hpp"

namespace gleeman {

[[noreturn]] void throw_no_nvml() {
  throw NoNVMLError("Not compiled with USE_NVML");
}

}
