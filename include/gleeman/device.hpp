#ifndef GLEEMAN_DEVICE_HPP
#define GLEEMAN_DEVICE_HPP

#include <cstdint>
#include <algorithm>
#include <tuple>

#include <nvml.h>
#include <cuda_runtime.h>

#include "gleeman/traits.hpp"
#include "gleeman/error.hpp"
#include "gleeman/nat.hpp"

namespace gleeman {

struct Device {
  Device(uint16_t device_id);
  static Device current();
#ifdef USE_NAT
  template <typename Nat, typename=std::enable_if_t<is_nat<Nat>::value>>
  Device(Nat nat) : device_id(nat.value) {
    defaultErrorHandler << cudaGetDeviceProperties(&property, device_id);
  }
#endif
  std::tuple<size_t, size_t> memory_information() const;

  const cudaDeviceProp &properties() const {
    return property;
  }

  const uint16_t device_id;
 private:
  cudaDeviceProp property;
};

struct Devices {
  static size_t installed_devices();
};

}

#endif //GLEEMAN_DEVICE_HPP
