#ifndef GLEEMAN_JOB_HPP
#define GLEEMAN_JOB_HPP

#include <memory>
#include <string>
#include <vector>

namespace gleeman {

struct Report {};

struct Exception {};

struct Executable {
  virtual void execute() noexcept = 0;
  virtual std::string inspect() noexcept = 0;
};

struct ShellCommand : Executable {
  ShellCommand(const std::vector<std::string> &command);
  ShellCommand(const std::string &command);
  void execute() noexcept;
  std::string inspect() noexcept;

  const std::string command;
  const std::vector<std::string> commands;
};

struct Job {
  Job(size_t gpus, size_t memory_requirement, size_t estimated_runtime,
      std::shared_ptr<Executable> executable);
  const std::shared_ptr<Executable> executable;
  const size_t memory_requirement;
  const size_t estimated_runtime;
  const size_t gpus;
  static Job from_yaml(const std::string yaml);
};

}  // name gleeman

#endif //GLEEMAN_JOB_HPP
