#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
#if defined(_WIN32) && defined(_M_ARM64)
  // __lzcnt is x86/x64-only; ARM64 MSVC exposes CLZ as _CountLeadingZeros
  return 1 << (CHAR_BIT * sizeof(num) - _CountLeadingZeros(num - 1));
#elif defined(_WIN32)
  return 1 << (CHAR_BIT * sizeof(num) - __lzcnt(num - 1));
#else
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
#endif
}

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

namespace vllm {

// Returns true if env VLLM_BATCH_INVARIANT=1
inline bool vllm_is_batch_invariant() {
  static bool cached = []() {
    const char* val = std::getenv("VLLM_BATCH_INVARIANT");
    return val && std::atoi(val) != 0;
  }();
  return cached;
}

}  // namespace vllm
