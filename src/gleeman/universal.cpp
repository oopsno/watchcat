#include "gleeman/universal.hpp"

namespace gleeman {

#ifdef USE_CUDA
constexpr size_t API<Driver>::index;
constexpr size_t API<CUDARuntime>::index;
#ifdef USE_NVML
constexpr size_t API<NVML>::index;
#endif //USE_CUDA
#endif //USE_NVML
constexpr size_t API<Universal>::index;
}