#ifndef GLEEMAN_TRAITS_HPP
#define GLEEMAN_TRAITS_HPP

#include <type_traits>
#include <string>

#include "gleeman/error.hpp"

namespace gleeman {

template<typename T, typename...>
using _1st = T;

template<typename, typename T, typename...>
using _2nd = T;

template<typename, typename, typename T, typename...>
using _3rd = T;

template<typename, typename, typename, typename T, typename...>
using _4th = T;

template<typename T>
struct unary_function_traits;
template<typename R, typename P0>
struct unary_function_traits<R(P0)> {
  static constexpr size_t parameters = 1;
  using return_type = R;
  using parameter_type_0 = P0;
  typedef R function_t(P0);

  static auto convert(function_t fn) {
    std::remove_pointer_t<P0> value;
    defaultErrorHandler << fn(&value);
    return value;
  }
};
template<typename R, typename P0>
const size_t unary_function_traits<R(P0)>::parameters;

template<typename T>
struct binary_function_traits;
template<typename R, typename P0, typename P1>
struct binary_function_traits<R(P0, P1)> {
  static constexpr size_t parameters = 2;
  using return_type = R;
  using parameter_type_0 = P0;
  using parameter_type_1 = P1;
  typedef R function_t(P0, P1);

  static auto convert(function_t fn) {
    std::remove_pointer_t<P0> fst;
    std::remove_pointer_t<P1> snd;
    defaultErrorHandler << fn(&fst, &snd);
    return std::make_tuple(fst, snd);
  }

  static auto convert_r(function_t fn, P1 snd) {
    std::remove_pointer_t<P0> fst;
    defaultErrorHandler << fn(&fst, snd);
    return fst;
  }

  static auto convert_l(function_t fn, P0 fst) {
    std::remove_pointer_t<P1> snd;
    defaultErrorHandler << fn(fst, &snd);
    return snd;
  }
};

template<typename R, typename P0, typename P1>
const size_t binary_function_traits<R(P0, P1)>::parameters;

template<typename...>
struct ternary_function_traits;
template<typename R, typename P0, typename P1, typename P2>
struct ternary_function_traits<R(P0, P1, P2)> {
  static constexpr size_t parameters = 3;
  using return_type = R;
  using parameter_type_0 = P0;
  using parameter_type_1 = P1;
  using parameter_type_2 = P2;
  typedef R function_t(P0, P1, P2);

  static auto convert(function_t function) {
    std::remove_pointer_t<P0> fst;
    std::remove_pointer_t<P1> snd;
    std::remove_pointer_t<P2> trd;
    defaultErrorHandler << function(&fst, &snd, &trd);
    return std::make_tuple(fst, snd, trd);
  }

  static auto convert_l(function_t function, P0 fst) {
    std::remove_pointer_t<P1> snd;
    std::remove_pointer_t<P2> trd;
    defaultErrorHandler << function(fst, &snd, &trd);
    return std::make_tuple(snd, trd);
  }

  static auto convert_l(function_t function, P0 fst, P1 snd) {
    std::remove_pointer_t<P2> trd;
    defaultErrorHandler << function(fst, snd, &trd);
    return trd;
  }

  static auto convert_r(function_t function, P2 trd) {
    std::remove_pointer_t<P0> fst;
    std::remove_pointer_t<P1> snd;
    defaultErrorHandler << function(&fst, &snd, trd);
    return std::make_tuple(fst, snd);
  }

  static auto convert_r(function_t function, P1 snd, P2 trd) {
    std::remove_pointer_t<P0> fst;
    defaultErrorHandler << function(&fst, snd, trd);
    return fst;
  }
};
template<typename R, typename P0, typename P1, typename P2>
const size_t ternary_function_traits<R(P0, P1, P2)>::parameters;

template<typename...>
struct quaternary_function_traits;
template<typename R, typename P0, typename P1, typename P2, typename P3>
struct quaternary_function_traits<R(P0, P1, P2, P3)> {
  static constexpr size_t parameters = 4;
  using return_type = R;
  using parameter_type_0 = P0;
  using parameter_type_1 = P1;
  using parameter_type_2 = P2;
  using parameter_type_3 = P3;
  typedef R function_t(P0, P1, P2, P3);
};
template<typename R, typename P0, typename P1, typename P2, typename P3>
const size_t quaternary_function_traits<R(P0, P1, P2, P3)>::parameters;

template<size_t N>
struct function_traits_impl;

#define STUB_FUNCTION_TRAITS(N, TRAITS) \
template<>                              \
struct function_traits_impl<N> {        \
  template <typename Function>          \
  using traits = TRAITS<Function>;      \
}

STUB_FUNCTION_TRAITS(1, unary_function_traits);
STUB_FUNCTION_TRAITS(2, binary_function_traits);
STUB_FUNCTION_TRAITS(3, ternary_function_traits);
STUB_FUNCTION_TRAITS(4, quaternary_function_traits);

#undef STUB_FUNCTION_TRAITS

template<typename>
struct function_traits;
template<typename Return, typename...Parameters>
struct function_traits<Return(Parameters...)> :
    function_traits_impl<sizeof...(Parameters)>::template traits<Return(Parameters...)> {
  typedef Return function_t(Parameters...);
  void call(function_t function, Parameters ...param) {
    defaultErrorHandler << function(param...);
  }
};

}

#define CALL(fn, ...)       function_traits<decltype(fn)>::call(fn, ##__VA_ARGS__)
#define CVT_CALL(fn, ...)   function_traits<decltype(fn)>::convert(fn, ##__VA_ARGS__)
#define CVT_CALL_L(fn, ...) function_traits<decltype(fn)>::convert_l(fn, ##__VA_ARGS__)
#define CVT_CALL_R(fn, ...) function_traits<decltype(fn)>::convert_r(fn, ##__VA_ARGS__)

#endif //GLEEMAN_TRAITS_HPP
