#ifndef GLEEMAN_CONTEXT_HPP
#define GLEEMAN_CONTEXT_HPP

namespace gleeman {

#ifdef USE_NAT
#define BOOL_USE_NAT true
#else
#define BOOL_USE_NAT false
#endif

#ifdef USE_CUDA
#define BOOL_USE_CUDA true
#else
#define BOOL_USE_CUDA false
#endif

#ifdef USE_NVML
#define BOOL_USE_NVML true
#else
#define BOOL_USE_NVML false
#endif

class Context {
 public:
  void initialize();
  void finalize();
  static constexpr bool use_nat = BOOL_USE_NAT;
  static constexpr bool use_cuda = BOOL_USE_CUDA;
  static constexpr bool use_nvml = BOOL_USE_NVML;
};
}

#undef BOOL_USE_NAT
#undef BOOL_USE_CUDA
#undef BOOL_USE_NVML

#endif //GLEEMAN_CONTEXT_HPP
