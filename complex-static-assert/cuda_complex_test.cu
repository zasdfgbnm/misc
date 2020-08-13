template<typename T>
struct complex {
  T real_ = T(0);

  constexpr complex(const T& re): real_(re) {}

  constexpr complex<T> &operator -=(double rhs) {
    real_ -= rhs;
    return *this;
  }
};

template<typename scalar_t>
constexpr complex<scalar_t> m(scalar_t real, double rhs) {
  complex<scalar_t> result(real);
  result -= rhs;
  return result;
}

__global__ void test_arithmetic_assign() {
  constexpr complex<float> y3 = m(float(2), 0.0);
  static_assert(y3.real_ == float(2), "");
}

int main() {}