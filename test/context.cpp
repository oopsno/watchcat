#include <catch.hpp>
#include "gleeman/context.hpp"

TEST_CASE("gleeman::Context") {
  using namespace gleeman;
  auto use_nat = Context::use_nat;
  auto use_cuda = Context::use_cuda;
  auto use_nvml = Context::use_nvml;

#ifdef USE_NAT
  CHECK(use_nat);
#else
  CHECK(!use_nat);
#endif

#ifdef USE_CUDA
  CHECK(use_cuda);
#else
  CHECK(!use_cuda);
#endif

#ifdef USE_NVML
  CHECK(use_nvml);
#else
  CHECK(!use_nvml);
#endif
}
