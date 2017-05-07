#include <catch.hpp>
#include <nvml.h>

#include "gleeman/traits.hpp"

TEST_CASE("gleeman::_nth") {
  using namespace gleeman;
  constexpr bool get_1st = std::is_same<_1st<void, char, int, float>, void>::value;
  constexpr bool get_2nd = std::is_same<_2nd<void, char, int, float>, char>::value;
  constexpr bool get_3rd = std::is_same<_3rd<void, char, int, float>, int>::value;
  constexpr bool get_4th = std::is_same<_4th<void, char, int, float>, float>::value;

  REQUIRE(get_1st);
  REQUIRE(get_2nd);
  REQUIRE(get_3rd);
  REQUIRE(get_4th);
}

TEST_CASE("gleeman::unary_function_traits") {
  using traits = gleeman::unary_function_traits<decltype(nvmlDeviceGetCount)>;
  bool require_uint32p = std::is_same<traits::parameter_type_0, unsigned int *>::value;
  bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;

  REQUIRE(require_uint32p);
  REQUIRE(returns_state);
}

TEST_CASE("gleeman::binary_function_traits") {
  using traits = gleeman::binary_function_traits<decltype(nvmlDeviceGetBrand)>;
  bool require_device = std::is_same<traits::parameter_type_0, nvmlDevice_t>::value;
  bool require_brandp = std::is_same<traits::parameter_type_1, nvmlBrandType_t *>::value;
  bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;

  REQUIRE(traits::parameters == 2);
  REQUIRE(require_device);
  REQUIRE(require_brandp);
  REQUIRE(returns_state);
}

TEST_CASE("gleeman::ternary_function_traits") {
  using traits = gleeman::ternary_function_traits<decltype(nvmlDeviceSetApplicationsClocks)>;
  bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;
  bool require_device = std::is_same<traits::parameter_type_0, nvmlDevice_t>::value;
  bool require_mem_clock = std::is_same<traits::parameter_type_1, unsigned int>::value;
  bool require_gpu_clock = std::is_same<traits::parameter_type_2, unsigned int>::value;

  REQUIRE(traits::parameters == 3);
  REQUIRE(returns_state);
  REQUIRE(require_device);
  REQUIRE(require_mem_clock);
  REQUIRE(require_gpu_clock);
}

TEST_CASE("gleeman::quaternary_function_traits") {
  using traits = gleeman::quaternary_function_traits<decltype(nvmlDeviceGetDetailedEccErrors)>;
  bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;
  bool require_device = std::is_same<traits::parameter_type_0, nvmlDevice_t>::value;
  bool require_error_t = std::is_same<traits::parameter_type_1, nvmlMemoryErrorType_t>::value;
  bool require_counter_t = std::is_same<traits::parameter_type_2, nvmlEccCounterType_t>::value;
  bool require_counter = std::is_same<traits::parameter_type_3, nvmlEccErrorCounts_t *>::value;

  REQUIRE(traits::parameters == 4);
  REQUIRE(returns_state);
  REQUIRE(require_device);
  REQUIRE(require_error_t);
  REQUIRE(require_counter_t);
  REQUIRE(require_counter);
}

TEST_CASE("gleeman::function_traits") {
  SECTION("unary") {
    using traits = gleeman::function_traits<decltype(nvmlDeviceGetCount)>;
    bool require_uint32p = std::is_same<traits::parameter_type_0, unsigned int *>::value;
    bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;

    REQUIRE(require_uint32p);
    REQUIRE(returns_state);
  }
  SECTION("binary") {
    using traits = gleeman::function_traits<decltype(nvmlDeviceGetBrand)>;
    bool require_device = std::is_same<traits::parameter_type_0, nvmlDevice_t>::value;
    bool require_brandp = std::is_same<traits::parameter_type_1, nvmlBrandType_t *>::value;
    bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;

    REQUIRE(traits::parameters == 2);
    REQUIRE(require_device);
    REQUIRE(require_brandp);
    REQUIRE(returns_state);
  }

  SECTION("ternary") {
    using traits = gleeman::function_traits<decltype(nvmlDeviceSetApplicationsClocks)>;
    bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;
    bool require_device = std::is_same<traits::parameter_type_0, nvmlDevice_t>::value;
    bool require_mem_clock = std::is_same<traits::parameter_type_1, unsigned int>::value;
    bool require_gpu_clock = std::is_same<traits::parameter_type_2, unsigned int>::value;

    REQUIRE(traits::parameters == 3);
    REQUIRE(returns_state);
    REQUIRE(require_device);
    REQUIRE(require_mem_clock);
    REQUIRE(require_gpu_clock);
  }

  SECTION("quaternary") {
    using traits = gleeman::function_traits<decltype(nvmlDeviceGetDetailedEccErrors)>;
    bool returns_state = std::is_same<traits::return_type, nvmlReturn_t>::value;
    bool require_device = std::is_same<traits::parameter_type_0, nvmlDevice_t>::value;
    bool require_error_t = std::is_same<traits::parameter_type_1, nvmlMemoryErrorType_t>::value;
    bool require_counter_t = std::is_same<traits::parameter_type_2, nvmlEccCounterType_t>::value;
    bool require_counter = std::is_same<traits::parameter_type_3, nvmlEccErrorCounts_t *>::value;

    REQUIRE(traits::parameters == 4);
    REQUIRE(returns_state);
    REQUIRE(require_device);
    REQUIRE(require_error_t);
    REQUIRE(require_counter_t);
    REQUIRE(require_counter);
  }
}
