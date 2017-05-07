#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include "gleeman/context.hpp"
#include "gleeman/error.hpp"

namespace gleeman {

void Context::initialize() {
  defaultErrorHandler << cuInit(0);
#ifdef USE_NVML
  nvmlInit();
#endif
}

void Context::finalize() {
#ifdef USE_NVML
  nvmlShutdown();
#endif
}

}
