#include <iostream>

template<bool use>
void f() {
    int a = 1;
    if constexpr(use) {
        std::cout << a << std::endl;
    }
}

void g() {
    int a = 1;
}

int main() {
    f<true>();
    f<false>();
}
