#include <sstream>
#include <vector>
#include <iostream>

#include "gleeman/job.hpp"

static std::string concat(const std::vector<std::string> &lines, const std::string sep) {
  std::stringstream ss;
  for (auto i = 0; i < lines.size();) {
    ss << lines[i];
    i += 1;
    if (i != lines.size()) {
      ss << sep;
    }
  }
  return ss.str();
}

namespace gleeman {

ShellCommand::ShellCommand(const std::vector<std::string> &commands)
    : commands(commands), command(concat(commands, "; ")) {};

ShellCommand::ShellCommand(const std::string &command)
    : commands({command}), command(command) {};

void ShellCommand::execute() noexcept {
  int code = std::system(command.c_str());
  if (code != 0) {
    std::cerr << "Crashed with: " << std::to_string(code) << std::endl;
  } else {
    std::cout << "Finished with: " << std::to_string(code) << std::endl;
  }
}

std::string ShellCommand::inspect() noexcept { return command; }

}