#ifndef GLEEMAN_EXCEPTION_HPP
#define GLEEMAN_EXCEPTION_HPP

#include <exception>
#include <stdexcept>

namespace gleeman {

class GleemanError : public std::exception {
 public:
  GleemanError(std::string what);
};

class NoNVMLError : public std::runtime_error {
 public:
  NoNVMLError(std::string what);
};

}

#endif //GLEEMAN_EXCEPTION_HPP
