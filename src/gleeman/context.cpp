#include "gleeman/cuda_headers.hpp"
#include "gleeman/context.hpp"
#include "gleeman/error.hpp"

namespace gleeman {

constexpr bool Context::use_nat;
constexpr bool Context::use_cuda;
constexpr bool Context::use_nvml;

void Context::initialize() {
#ifdef USE_CUDA
  defaultErrorHandler << cuInit(0);
#ifdef USE_NVML
  nvmlInit();
#endif //USE_NVML
#endif //USE_CUDA
}

void Context::finalize() {
#ifdef USE_NVML
  nvmlShutdown();
#endif //USE_NVML
}
}
