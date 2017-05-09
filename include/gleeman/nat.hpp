#ifndef GLEEMAN_NAT_HPP
#define GLEEMAN_NAT_HPP

#include <type_traits>

#include "gleeman/universal.hpp"

namespace gleeman {
namespace nat {

struct Z {
  static constexpr size_t value = 0;
};

template<typename Nat>
struct S : Nat {
  static constexpr size_t value = 1 + Nat::value;
};
template<typename Nat>
constexpr size_t S<Nat>::value;

template<typename T>
struct is_nat : std::false_type {};

template<>
struct is_nat<Z> : std::true_type {};

template<typename Next>
struct is_nat<S<Next>> : is_nat<Next> {};

template<size_t N>
struct construct {
  using type = S<typename construct<N - 1>::type>;
};

template<>
struct construct<0> {
  using type = Z;
};

template<typename LHS, typename RHS>
struct add;

template<typename Result>
struct add<Result, Z> {
  using type = Result;
};

template<typename LHS, typename Next>
struct add<LHS, S<Next>> {
  using type = typename add<S<LHS>, Next>::type;
};

template<typename LHS, typename RHS>
struct mul;

template<typename Result>
struct mul<Result, Z> {
  using type = Z;
};

template<typename LHS, typename Next>
struct mul<LHS, S<Next>> {
  using type = typename add<LHS, typename mul<LHS, Next>::type>::type;
};
}
}

#endif //GLEEMAN_NAT_HPP
