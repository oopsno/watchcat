#ifndef GLEEMAN_PLACEHOLDER_HPP
#define GLEEMAN_PLACEHOLDER_HPP

#include <cstdint>

#include "gleeman/device.hpp"

namespace gleeman {

class Placeholder {
 public:
  Placeholder(const Device &device);
  ~Placeholder();
  const Placeholder &memory(double rate = 0.80, double step = 0.05);
  const Placeholder &calculation();
  const Placeholder &release();
 private:
  const Device &device;
  size_t allocated_memory_size;
  void *address;
};

}

#endif //GLEEMAN_PLACEHOLDER_HPP
