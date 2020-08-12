#include "complex.h"

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> m(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result -= rhs;
  return result;
}

template<typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_assign_complex() {
  using namespace c10::complex_literals;
  constexpr c10::complex<scalar_t> y3 = m(scalar_t(2), scalar_t(2), 1.0_id);
  static_assert(y3.real() == scalar_t(2), "");
#if !defined(__CUDACC__)
  // The following is flaky on nvcc
  static_assert(y3.imag() == scalar_t(1), "");
#endif
}

__global__ void test_arithmetic_assign() {
  test_arithmetic_assign_complex<float>();
  test_arithmetic_assign_complex<double>();
}

int main() {}