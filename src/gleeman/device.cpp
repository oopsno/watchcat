#include <tuple>
#include <type_traits>

#include "gleeman/universal.hpp"
#include "gleeman/traits.hpp"
#include "gleeman/device.hpp"

namespace gleeman {

Device::Device(uint16_t device_id) : device_id(device_id) {
#ifdef USE_CUDA
  CALL(cudaGetDeviceProperties, &property, device_id);
#ifdef USE_NVML
  CALL(nvmlDeviceGetHandleByIndex, device_id, &handle);
#endif //USE_NVML
#endif //USE_CUDA
}

std::tuple<size_t, size_t> Device::memory_information() const {
#ifdef USE_NVML
  auto mem_info = CVT_CALL_L(nvmlDeviceGetMemoryInfo, handle);
  return std::make_tuple(mem_info.free, mem_info.total);
#elif USE_CUDA
  activate();
  return CVT_CALL(cudaMemGetInfo);
#else
  throw_no_cuda();
  return std::make_tuple(0, 0);
#endif
}

Device Device::current() {
#ifdef USE_CUDA
  auto device_id = CVT_CALL(cudaGetDevice);
  return Device(device_id);
#else
  throw_no_cuda();
  return Device(0xFF);
#endif
}

void Device::activate() const {
#ifdef USE_CUDA
  CALL(cudaSetDevice, device_id);
#else
  throw_no_cuda();
#endif
}

size_t Devices::installed_devices() {
  size_t counter = 0;
#ifdef USE_NVML
  counter = CVT_CALL(nvmlDeviceGetCount);
#elif USE_CUDA
  counter = CVT_CALL(cudaGetDeviceCount);
#endif
  return counter;
}
}
