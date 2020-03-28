struct T {
  int a, b;
  T(int a = 0, int b = 0): a(a), b(b) {}
};

struct U: public T {
  using T::T;
  U(float _) {}
};

int main() {
  U u;
}
