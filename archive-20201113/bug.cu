namespace c10 {
struct complex{};
}

void operator+(c10::complex lhs, c10::complex rhs) {}

namespace at {
struct Tensor {};
void operator +(Tensor x, Tensor y) {}
}

namespace at { namespace native {

void add(c10::complex a) {
  a + a; // error: no operator "+" matches these operands
}

}}
