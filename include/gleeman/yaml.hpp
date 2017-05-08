#ifndef GLEEMAN_YAML_HPP
#define GLEEMAN_YAML_HPP

#include "yaml-cpp/yaml.h"

using size_t = std::size_t;

template<typename T>
T get(const YAML::Node &node, const std::string name, const T fallback) {
  auto cursor = node[name];
  if (cursor.IsDefined()) {
    return cursor.as<T>();
  } else {
    return fallback;
  }
}

auto parse_first_real(const std::string line);
size_t parse_time(const std::string text);
size_t parse_gram(const std::string text);

#endif //GLEEMAN_YAML_HPP
