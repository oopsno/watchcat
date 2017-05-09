#include <cmath>
#include <iterator>

#include "gleeman/yaml.hpp"

using ull_t = unsigned long long int;

ull_t operator"" _B(ull_t base) { return base; }
ull_t operator"" _KB(ull_t base) { return base * 1024_B; }
ull_t operator"" _MB(ull_t base) { return base * 1024_KB; };
ull_t operator"" _GB(ull_t base) { return base * 1024_MB; };
ull_t operator"" _sec(ull_t base) { return base; }
ull_t operator"" _min(ull_t base) { return base * 60_sec; }
ull_t operator"" _hr(ull_t base) { return base * 60_min; }
ull_t operator"" _day(ull_t base) { return base * 24_hr; }
ull_t operator"" _week(ull_t base) { return base * 7_day; }

auto parse_first_real(const std::string line) {
  auto begin = line.c_str();
  auto end = begin + line.size();
  char *stop;
  double real = std::strtod(begin, &stop);
  std::string remains = line.substr(std::distance((const char *)(stop), end));
  return std::make_tuple(real, remains);
}

#define INPUT(x)          \
  double base, rate = -1; \
  std::string suffix;     \
  std::tie(base, suffix) = parse_first_real(x)

#define PROCESS_SUFFIX if (suffix.size())

#define TRY_MATCH(x)                   \
  if (suffix == #x) {                  \
    rate = static_cast<double>(1_##x); \
  }

#define RESULT() static_cast<size_t>(std::ceil(base * rate))

#define THROW_OTHERWISE                                             \
  if (rate < 0) {                                                   \
    throw std::invalid_argument(("cannot parse: " + text).c_str()); \
  }

size_t parse_gram(const std::string text) {
  INPUT(text);
  PROCESS_SUFFIX {
    TRY_MATCH(GB);
    TRY_MATCH(MB);
    TRY_MATCH(KB);
    THROW_OTHERWISE;
  }
  return RESULT();
}

size_t parse_time(const std::string text) {
  INPUT(text);
  PROCESS_SUFFIX {
    TRY_MATCH(week);
    TRY_MATCH(day);
    TRY_MATCH(hr);
    TRY_MATCH(min);
    TRY_MATCH(sec);
    THROW_OTHERWISE;
  }
  return RESULT();
}

#undef INPUT
#undef PROCESS_SUFFIX
#undef MATCH
#undef THROW_OTHERWISE
#undef RESULT
