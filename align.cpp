#include <iostream>
#include <tuple>

struct alignas(16) A {
  double data[2];
};

int main() {
    using tup = std::tuple<bool, A>;
    std::cout << alignof(tup) << std::endl;
}