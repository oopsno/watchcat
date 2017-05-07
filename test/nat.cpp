#include <catch.hpp>
#include <iostream>
#include "gleeman/nat.hpp"
#include <typeinfo>

using namespace gleeman::nat;

TEST_CASE("gleeman::nat::construct") {
  constexpr size_t I4 = 4;
  using N4 = S<S<S<S<Z>>>>;
  auto same = std::is_same<construct<I4>::type, N4>::value;

  REQUIRE(same);
  REQUIRE(N4::value == I4);
  REQUIRE(N4().value == I4);
}

TEST_CASE("gleeman::nat::is_nat") {
  using namespace gleeman::nat;
  using N0 = Z;
  using N4 = S<S<S<S<Z>>>>;
  constexpr bool n0_is_nat = is_nat<N0>::value;
  constexpr bool n4_is_nat = is_nat<N4>::value;
  constexpr bool void_is_not_nat = !is_nat<void>::value;

  REQUIRE( n0_is_nat );
  REQUIRE( n4_is_nat );
  REQUIRE( void_is_not_nat );
}

TEST_CASE("gleeman::nat::mul") {
  constexpr size_t I3 = 3, I7 = 7;
  using N3  = construct<I3>::type;
  using N7  = construct<I7>::type;
  using N21 = mul<N3, N7>::type;

  REQUIRE(N21::value == I3 * I7);
}

TEST_CASE("gleeman::nat::add") {
  constexpr size_t I3 = 3, I7 = 7;
  using N3  = construct<I3>::type;
  using N7  = construct<I7>::type;
  using N10 = add<N3, N7>::type;

  REQUIRE(N10::value == I3 + I7);
}

TEST_CASE("gleeman::nat::limits") {
  using N002 = S<S<Z>>;
  using N004 = S<S<N002>>;
  using N005 = S<N004>;
  using N008 = mul<N002, N004>::type;
  using N016 = mul<N004, N004>::type;
  using N025 = mul<N005, N005>::type;
  using N200 = mul<N008, N025>::type;
  using N256 = mul<N016, N016>::type;
  using N400 = mul<N002, N200>::type;

  REQUIRE(N002::value ==   2);
  REQUIRE(N004::value ==   4);
  REQUIRE(N008::value ==   8);
  REQUIRE(N016::value ==  16);
  REQUIRE(N025::value ==  25);
  REQUIRE(N200::value == 200);
  REQUIRE(N256::value == 256);
  REQUIRE(N400::value == 400);
}
