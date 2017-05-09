#include <catch.hpp>

#include "gleeman/cuda_headers.hpp"
#include "gleeman/exception.hpp"
#include "gleeman/error.hpp"

TEST_CASE("gleeman::error_traits") {
  using namespace gleeman;
  #define SUCCESS_OF_API(api) error_traits<API<api>::error_type>::success
#ifdef USE_CUDA
  auto driver_success    = SUCCESS_OF_API(Driver);
  auto runtime_success   = SUCCESS_OF_API(CUDARuntime);
  REQUIRE(driver_success == cudaSuccess);
  REQUIRE(nvml_success   == NVML_SUCCESS);
#ifdef USE_NVML
  auto nvml_success      = SUCCESS_OF_API(NVML);
  REQUIRE(nvml_success   == NVML_SUCCESS);
#endif //USE_NVML
#endif //USE_CUDA
  auto universal_success = SUCCESS_OF_API(Universal);
  REQUIRE(universal_success == 0);
}

TEST_CASE("gleeman::UniveralErrorHandler") {
  gleeman::UniversalErrorHandler error;

  SECTION("default constructor") {
#ifdef USE_CUDA
    REQUIRE(error == CUDA_SUCCESS);
    REQUIRE(error == cudaSuccess);
#ifdef USE_NVML
    REQUIRE(error == NVML_SUCCESS);
#endif
#endif
  }

  SECTION("operator=") {
#ifdef USE_CUDA
    CHECK_THROWS_AS(error.handle(CUDA_ERROR_OUT_OF_MEMORY),
                    gleeman::GleemanError);
    REQUIRE(error == CUDA_ERROR_OUT_OF_MEMORY);

    CHECK_THROWS_AS(error.handle(cudaErrorMemoryAllocation),
                    gleeman::GleemanError);
    REQUIRE(error == cudaErrorMemoryAllocation);

#ifdef USE_NVML
    CHECK_THROWS_AS(error.handle(NVML_ERROR_TIMEOUT), gleeman::NoNVMLError);
    REQUIRE(error == NVML_ERROR_TIMEOUT);
#endif //USE_NVML
#endif //USE_CUDA
  }
}
