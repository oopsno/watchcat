#ifndef GLEEMAN_CUDA_HEADERS_HPP
#define GLEEMAN_CUDA_HEADERS_HPP

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef USE_NVML
#include <nvml.h>
#endif //USE_NVML
#endif //USE_CUDA

#endif //GLEEMAN_CUDA_HEADERS_HPP
