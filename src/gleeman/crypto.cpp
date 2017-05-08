#include <hex.h>
#include <eccrypto.h>

#include "gleeman/crypto.hpp"

namespace gleeman {

template<typename Algorithm>
struct SHACrypto {
  const SHACrypto<Algorithm> &feed(const std::string &text) {
    algorithm.CalculateDigest(digests, (const byte *) text.c_str(), text.size());
    return *this;
  }

  std::string encode() const {
    std::string cryptogram;
    CryptoPP::HexEncoder encoder;
    encoder.Attach(new CryptoPP::StringSink(cryptogram));
    encoder.Put(digests, Algorithm::DIGESTSIZE);
    encoder.MessageEnd();
    return cryptogram;
  }

  Algorithm algorithm;
  byte digests[Algorithm::DIGESTSIZE];
};

#define EXPORT_ALGORITHM(name, cls)        \
std::string name(const std::string text) { \
  SHACrypto<cls> crypto;                   \
  return crypto.feed(text).encode();       \
}

EXPORT_ALGORITHM(sha1,   CryptoPP::SHA1)
EXPORT_ALGORITHM(sha224, CryptoPP::SHA224)
EXPORT_ALGORITHM(sha256, CryptoPP::SHA256)
EXPORT_ALGORITHM(sha384, CryptoPP::SHA384)
EXPORT_ALGORITHM(sha512, CryptoPP::SHA512)

#undef EXPORT_ALGORITHM

}