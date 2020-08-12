#include <type_traits>
#include <tuple>
#include <sstream>
#include "complex.h"

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define MAYBE_GLOBAL __global__
#else
#define MAYBE_GLOBAL
#endif

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> p(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result += rhs;
  return result;
}

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> m(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result -= rhs;
  return result;
}

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> t(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result *= rhs;
  return result;
}

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> d(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result /= rhs;
  return result;
}

template<typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_assign_complex() {
  using namespace c10::complex_literals;
  constexpr c10::complex<scalar_t> x2 = p(scalar_t(2), scalar_t(2), 1.0_if);
  static_assert(x2.real() == scalar_t(2), "");
  static_assert(x2.imag() == scalar_t(3), "");
  constexpr c10::complex<scalar_t> x3 = p(scalar_t(2), scalar_t(2), 1.0_id);
  static_assert(x3.real() == scalar_t(2), "");
#if !defined(__CUDACC__)
  // The following is flaky on nvcc
  static_assert(x3.imag() == scalar_t(3), "");
#endif

  constexpr c10::complex<scalar_t> y2 = m(scalar_t(2), scalar_t(2), 1.0_if);
  static_assert(y2.real() == scalar_t(2), "");
  static_assert(y2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> y3 = m(scalar_t(2), scalar_t(2), 1.0_id);
  static_assert(y3.real() == scalar_t(2), "");
#if !defined(__CUDACC__)
  // The following is flaky on nvcc
  static_assert(y3.imag() == scalar_t(1), "");
#endif

  constexpr c10::complex<scalar_t> z2 = t(scalar_t(1), scalar_t(-2), 1.0_if);
  static_assert(z2.real() == scalar_t(2), "");
  static_assert(z2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> z3 = t(scalar_t(1), scalar_t(-2), 1.0_id);
  static_assert(z3.real() == scalar_t(2), "");
  static_assert(z3.imag() == scalar_t(1), "");

  constexpr c10::complex<scalar_t> t2 = d(scalar_t(-1), scalar_t(2), 1.0_if);
  static_assert(t2.real() == scalar_t(2), "");
  static_assert(t2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> t3 = d(scalar_t(-1), scalar_t(2), 1.0_id);
  static_assert(t3.real() == scalar_t(2), "");
  static_assert(t3.imag() == scalar_t(1), "");
}

MAYBE_GLOBAL void test_arithmetic_assign() {
  test_arithmetic_assign_complex<float>();
  test_arithmetic_assign_complex<double>();
}