#include <catch.hpp>
#include <iostream>
#include "gleeman/nat.hpp"
#include <typeinfo>

TEST_CASE("gleeman::nat::construct") {
  using namespace gleeman::nat;
  constexpr size_t I4 = 4;
  using N4 = S<S<S<S<Z>>>>;
  auto  same = std::is_same<construct<I4>::type, N4>::value;
  N4 n4;

  REQUIRE( same );
  REQUIRE( N4::value == I4);
  REQUIRE( n4.value  == I4);
}
