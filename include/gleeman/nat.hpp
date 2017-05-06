#ifndef GLEEMAN_NAT_HPP
#define GLEEMAN_NAT_HPP

#include <type_traits>

namespace gleeman {
namespace nat {

struct Z {
  static constexpr size_t value = 0;
};

template<typename Nat>
struct S : Nat {
  static constexpr size_t value = 1 + Nat::value;
};

template<size_t N>
struct construct {
  using type = S<typename construct<N - 1>::type>;
};

template<>
struct construct<0> {
  using type = Z;
};

}
}

#endif //GLEEMAN_NAT_HPP
