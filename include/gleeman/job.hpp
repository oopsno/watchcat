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
  std::string name;
  std::shared_ptr<Executable> executable;
  size_t memory_requirement;
  size_t estimated_runtime;
  size_t gpus;
  static Job from_yaml(const std::string yaml);
 private:
  Job();
};

}  // name gleeman

#endif //GLEEMAN_JOB_HPP
