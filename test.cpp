namespace c10 {
struct complex{};
void operator+(complex lhs, complex rhs) {}
}


namespace at {
struct Tensor {};
void operator +(Tensor x, Tensor y) {}
}

namespace at { namespace native {

void add(c10::complex a) {
  a + a; // error: no operator "+" matches these operands
}

}}

int main() {}
