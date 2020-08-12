#pragma once

#include <complex>

#define C10_HOST_DEVICE __host__ __device__

namespace c10 {

template<typename T>
struct alignas(sizeof(T) * 2) complex {
  using value_type = T;

  T real_ = T(0);
  T imag_ = T(0);

  constexpr complex() = default;
  constexpr complex(const T& re, const T& im = T()): real_(re), imag_(im) {}
  template<typename U>
  explicit constexpr complex(const std::complex<U> &other): complex(other.real(), other.imag()) {}


  // Use SFINAE to specialize casting constructor for c10::complex<float> and c10::complex<double>
  template<typename U = T>
  explicit constexpr complex(const std::enable_if_t<std::is_same<U, float>::value, complex<double>> &other):
    real_(other.real_), imag_(other.imag_) {}
  template<typename U = T>
  constexpr complex(const std::enable_if_t<std::is_same<U, double>::value, complex<float>> &other):
    real_(other.real_), imag_(other.imag_) {}

  template<typename U>
  constexpr complex<T> &operator =(const complex<U> &rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

  template<typename U>
  constexpr complex<T> &operator -=(const complex<U> &rhs) {
    real_ -= rhs.real();
    imag_ -= rhs.imag();
    return *this;
  }

  constexpr T real() const {
    return real_;
  }
  constexpr void real(T value) {
    real_ = value;
  }
  constexpr T imag() const {
    return imag_;
  }
  constexpr void imag(T value) {
    imag_ = value;
  }
};

namespace complex_literals {

constexpr complex<float> operator"" _if(long double imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator"" _id(long double imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

constexpr complex<float> operator"" _if(unsigned long long imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator"" _id(unsigned long long imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

} // namespace complex_literals

} // namespace c10