#ifdef USE_CUDA
#include <algorithm>
#include <tuple>

#include "gleeman/placeholder.hpp"

namespace gleeman {

Placeholder::Placeholder(const Device &device)
    : device(device), allocated_memory_size(0ULL), address(nullptr) {}

Placeholder::~Placeholder() { release(); }

const Placeholder &Placeholder::memory(double rate, double step) {
  size_t free, total;
  rate = std::min(std::abs(rate), 1.0);
  std::tie(free, total) = device.memory_information();
  Device current_device = Device::current();
  device.activate();
  while (rate > 0) {
    size_t size = static_cast<size_t>(free * rate);
    auto state = cudaMalloc(&address, size);
    if (state != error_traits<decltype(state)>::success) {
      rate -= step;
    } else {
      allocated_memory_size = size;
      break;
    }
    if (rate <= 0) {
      address = nullptr;
      allocated_memory_size = 0;
      current_device.activate();
      defaultErrorHandler << state;
    }
  }
  return *this;
}

const Placeholder &Placeholder::release() {
  if (allocated_memory_size > 0 && address != nullptr) {
    defaultErrorHandler << cudaFree(address);
  }
  return *this;
}
}
#endif //USE_CUDA