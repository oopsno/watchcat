#include <iostream>
#include "gleeman/device.hpp"

int main(int argc, char *argv[]) {
  std::cout << gleeman::Devices::installed_devices() << std::endl;
  return 0;
}
