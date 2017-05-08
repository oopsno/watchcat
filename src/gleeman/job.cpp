#include "gleeman/job.hpp"
#include "gleeman/yaml.hpp"

using size_t = std::size_t;

namespace gleeman {
Job::Job() {}

Job Job::from_yaml(const std::string yaml) {
  auto job = Job();
  const auto node = YAML::Load(yaml);
  job.name = get(node, "name", std::string("NewYAMLJob"));
  job.memory_requirement = parse_gram(node["gram"].as<std::string>());
  job.estimated_runtime = parse_time(node["time"].as<std::string>());
  job.gpus = get(node, "gpus", 1ULL);
  const auto scripts = node["scripts"].as<std::vector<std::string >>();
  job.executable = std::make_shared<ShellCommand>(scripts);
  return job;
}

}  // namespace gleeman


