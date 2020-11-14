struct Two {
  float t = 2;
  constexpr float operator-(double rhs) {
    return t -= rhs;
  }
};

static_assert(Two() - 0.0 == 2.0f, "2 - 0 == 2 !!!");  // error: static assertion failed with "2 - 0 == 2 !!!"

int main() {}