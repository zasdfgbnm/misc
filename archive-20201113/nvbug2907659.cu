struct Two {
  float t = 2;
  constexpr float operator+(double rhs) {
    return t += rhs;
  }
};
static_assert(Two() + 1.0 != 2.0f, "2 + 1 == 3 !!!");  // error: static assertion failed with "2 + 1 == 3 !!!"
int main() {}