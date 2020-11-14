struct A {
  int x = 1;
};
constexpr A a = (A() = A());  // error: expression must have a constant value
int main() {}