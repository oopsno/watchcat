#include <type_traits>

#include "gleeman/traits.hpp"
#include "gleeman/device.hpp"

namespace gleeman {

Device::Device(uint16_t device_id) : device_id(device_id) {
  cudaGetDeviceProperties(&this->properties, device_id);
}

std::tuple<size_t, size_t> Device::memory_information() const {
  size_t total, free;
  UniformedError error;
  error << cudaMemGetInfo(&free, &total);
  return std::make_tuple(free, total);
}

Device Device::current() {
  auto device_id = CVT_CALL(cudaGetDevice);
  return Device(device_id);
}

size_t Devices::installed_devices() {
  auto counter = CVT_CALL(cudaGetDeviceCount);
  if (std::is_signed<decltype(counter)>::value) {
    counter = std::max(0, counter);
  }
  return counter;
}


}