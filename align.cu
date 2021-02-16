#include <iostream>
#include <tuple>
#include <cuda_runtime.h>

struct alignas(16) A {
    double data[2];
};

__global__ void align() {
    using tup = std::tuple<bool, A>;
    printf("%d", (int)alignof(tup));
}

int main() {
    using tup = std::tuple<bool, A>;
    std::cout << alignof(tup) << std::endl;
    align <<<1, 1>>> ();
    cudaDeviceSynchronize();
}