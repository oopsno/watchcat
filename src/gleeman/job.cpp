#include "gleeman/job.hpp"
#include "gleeman/yaml.hpp"

using size_t = std::size_t;

namespace gleeman {
Job::Job(size_t gpus, size_t memory_requirement, size_t estimated_runtime, std::shared_ptr<Executable> executable)
    : gpus(gpus),
      memory_requirement(memory_requirement),
      estimated_runtime(estimated_runtime),
      executable(executable) {};

Job Job::from_yaml(const std::string yaml) {
  auto node = YAML::Load(yaml);
  auto name = get(node, "name", std::string("NewYAMLJob"));
  auto gram = parse_gram(node["gram"].as<std::string>());
  auto estime = parse_time(node["time"].as<std::string>());
  auto gpus = get(node, "gpus", 1ULL);
  auto scripts = node["scripts"].as<std::vector<std::string >>();
  auto exe = std::make_shared<ShellCommand>(scripts);
  return Job(gpus, gram, estime, exe);
}

}  // namespace gleeman


