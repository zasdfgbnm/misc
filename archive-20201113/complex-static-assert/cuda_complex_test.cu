struct complex {
  float real_ = 0;

  constexpr complex(const float& re): real_(re) {}

  constexpr complex &operator -=(double rhs) {
    real_ -= rhs;
    return *this;
  }
};

constexpr complex m(float real, double rhs) {
  complex result(real);
  result -= rhs;
  return result;
}

__global__ void test_arithmetic_assign() {
  constexpr complex y3 = m(float(2), 0.0);
  static_assert(y3.real_ == float(2), "");
}

int main() {}