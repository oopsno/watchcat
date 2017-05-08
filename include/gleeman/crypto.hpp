#ifndef GLEEMAN_CRYPTO_HPP
#define GLEEMAN_CRYPTO_HPP

#include <string>

namespace gleeman {

std::string sha1(const std::string text);
std::string sha224(const std::string text);
std::string sha256(const std::string text);
std::string sha384(const std::string text);
std::string sha512(const std::string text);

}

#endif  // GLEEMAN_CRYPTO_HPP