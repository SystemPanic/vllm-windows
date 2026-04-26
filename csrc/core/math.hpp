#pragma once

#include <climits>
#include <iostream>

#ifdef _MSC_VER
// MSVC constexpr-friendly implementation (no __builtin_clz).
inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  uint32_t v = num - 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}
#else
inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
#ifdef _WIN32
  return 1 << (CHAR_BIT * sizeof(num) - __lzcnt(num - 1));
#else
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
#endif
}
#endif

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// Round a down to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

// Round a up to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}
