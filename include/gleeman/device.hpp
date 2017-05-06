#ifndef GLEEMAN_DEVICE_HPP
#define GLEEMAN_DEVICE_HPP

#include <cstdint>
#include <algorithm>
#include <tuple>

#include <nvml.h>
#include <cuda_runtime.h>

#include "gleeman/traits.hpp"
#include "gleeman/error.hpp"

namespace gleeman {

struct Device {
  Device(uint16_t device_id);
  static Device current();
  std::tuple<size_t, size_t> memory_information() const;

  cudaDeviceProp properties;
  const uint16_t device_id;
};

struct Devices {
  static size_t installed_devices();
};

}

#endif //GLEEMAN_DEVICE_HPP
