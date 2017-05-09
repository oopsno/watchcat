#include <type_traits>

#include "gleeman/traits.hpp"
#include "gleeman/device.hpp"
#include "gleeman/context.hpp"
#include "gleeman/exception.hpp"

namespace gleeman {

Device::Device(uint16_t device_id) : device_id(device_id) {
#ifdef USE_CUDA
  defaultErrorHandler << cudaGetDeviceProperties(&property, device_id);
#endif
}

std::tuple<size_t, size_t> Device::memory_information() const {
#ifdef USE_CUDA
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

size_t Devices::installed_devices() {
  size_t counter = 0;
#ifdef USE_CUDA
  counter = CVT_CALL(cudaGetDeviceCount);
  if (std::is_signed<decltype(counter)>::value) {
    counter = std::max(0, counter);
  }
#endif
  return counter;
}
}