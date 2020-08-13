template<typename T>
struct alignas(sizeof(T) * 2) complex {
  using value_type = T;

  T real_ = T(0);
  T imag_ = T(0);

  constexpr complex() = default;
  constexpr complex(const T& re, const T& im = T()): real_(re), imag_(im) {}

  template<typename U>
  constexpr complex<T> &operator -=(const complex<U> &rhs) {
    real_ -= rhs.real();
    imag_ -= rhs.imag();
    return *this;
  }

  constexpr T real() const {
    return real_;
  }

  constexpr T imag() const {
    return imag_;
  }
};

constexpr complex<double> operator"" _id(long double imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

template<typename scalar_t, typename rhs_t>
constexpr complex<scalar_t> m(scalar_t real, scalar_t imag, complex<rhs_t> rhs) {
  complex<scalar_t> result(real, imag);
  result -= rhs;
  return result;
}

__global__ void test_arithmetic_assign() {
  constexpr complex<float> y3 = m(float(2), float(2), 1.0_id);
  static_assert(y3.real() == float(2), "");
}

int main() {}