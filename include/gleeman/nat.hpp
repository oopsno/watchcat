#ifndef GLEEMAN_NAT_HPP
#define GLEEMAN_NAT_HPP

#include <type_traits>

namespace gleeman {
namespace nat {

struct Z {};

template<typename Nat>
struct S : Nat {};

template<size_t N>
struct construct {
  using type = S<typename construct<N - 1>::type>;
};

template<>
struct construct<0> {
  using type = Z;
};

template<typename Nat>
struct serialize;

template<>
struct serialize<Z> : std::integral_constant<size_t, 0> {};

template<typename Pred>
struct serialize<S<Pred>> : std::integral_constant<size_t, 1 + serialize<Pred>::value> {};

}
}

#endif //GLEEMAN_NAT_HPP
