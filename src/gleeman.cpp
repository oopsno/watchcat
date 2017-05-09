#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

#include "gleeman/device.hpp"
#include "gleeman/placeholder.hpp"

using namespace gleeman;
namespace po = boost::program_options;
using gpus_t = std::vector<uint16_t>;

auto build_options() {
  using namespace boost::program_options;
  options_description description;
  options_description command_devices("Hold options");
  command_devices.add_options()
      ("gpu,x", value<gpus_t>(), "GPU(s)");

  options_description command_generic("Generic options");
  command_generic.add_options()
      ("help", "print this message")
      ("hold,h", "hold device(s)")
      ("query,q", "query device(s)");

  description
      .add(command_generic)
      .add(command_devices);

  return description;
}

auto parse_options(int argc, const char *argv[]) {
  auto description = build_options();
  po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, description), vm);
  po::notify(vm);

  return vm;
}

void help() {
  auto description = build_options();

  std::cout << "gleeman usage:" << std::endl << std::endl
            << description << std::endl;
}

void query(const gpus_t &gpus) {
  const auto devices = Devices::installed_devices();
  std::cout << "Installed device(s): " << devices << std::endl;
  for (uint16_t device_id = 0; device_id < devices; ++device_id) {
    Device device(device_id);
    size_t free, total;
    std::tie(free, total) = device.memory_information();
    free  >>= 20;
    total >>= 20;
    std::cout << "\tDevice ID:     " << device_id << std::endl
              << "\tDevice name:   " << device.properties().name << std::endl
              << "\tDevice bus:    " << device.properties().pciBusID << std::endl
              << "\tDevice memory: " << free << "MB / " << total << "MB" << std::endl;
  }
}

void hold(const gpus_t &gpus) {
  const auto devices = Devices::installed_devices();
  std::cout << "Installed device(s): " << devices << std::endl;
  for (uint16_t device_id = 0; device_id < devices; ++device_id) {
    Device device(device_id);
    Placeholder placeholder(device);
    std::cout << "Holding device: " << device_id << std::endl;
    placeholder.memory();
  }
  std::cout << "Pending..." << std::endl;
  std::string what_ever;
  std::cin >> what_ever;
}

int main(int argc, const char *argv[]) {
  gpus_t gpus;
  auto vm = parse_options(argc, argv);
  if (vm.count("gpu")) {
    gpus = vm["gpu"].as<gpus_t>();
  }
  if (vm.count("hold")) {
    hold(gpus);
  } else if (vm.count("query")) {
    query(gpus);
  } else {
    help();
  }
  return 0;
}
