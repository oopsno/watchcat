#include "gleeman/exception.hpp"

namespace gleeman {

GleemanError::GleemanError(std::string what) : std::exception(what.c_str()) {}

NoNVMLError::NoNVMLError(std::string what) : std::runtime_error(what) {}

}
