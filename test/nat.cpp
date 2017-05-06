#include <catch.hpp>
#include <iostream>
#include "gleeman/nat.hpp"
#include <typeinfo>

TEST_CASE("gleeman::nat::construct") {
  using namespace gleeman::nat;
  constexpr size_t I4 = 4;
  using N4 = S<S<S<S<Z>>>>;
  auto  same = std::is_same<construct<I4>::type, N4>::value;

  REQUIRE( same );
}

TEST_CASE("gleeman::nat::serialize") {
  using namespace gleeman::nat;
  constexpr size_t I4 = 4;
  using N4 = S<S<S<S<Z>>>>;
  auto same = serialize<N4>::value == I4;

  REQUIRE( same );
}
