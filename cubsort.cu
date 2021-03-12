#include <iostream>
#include <cub/cub.cuh>
#include <chrono>

using namespace std::chrono;

constexpr size_t N = 1000000;

int main() {
    int *h1, *h2, *d1, *d2, *d3, *d4, *tmp;
    h1 = new int[N];
    h2 = new int[N];
    for(size_t i = 0; i < N; i++) {
        h1[i] = N - i;
    }
    cudaMalloc(&d1, sizeof(int) * N);
    cudaMalloc(&d2, sizeof(int) * N);
    cudaMalloc(&d3, sizeof(int) * N);
    cudaMalloc(&d4, sizeof(int) * N);
    cudaMemcpy(d1, h1, sizeof(int) * N, cudaMemcpyDefault);

    size_t temp_storage_bytes;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d1, d2, d3, d4, N);
    cudaMalloc(&tmp, temp_storage_bytes);

    cudaDeviceSynchronize();
    auto start = high_resolution_clock::now();
    cub::DeviceRadixSort::SortPairs(tmp, temp_storage_bytes, d1, d2, d3, d4, N);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << duration.count() << std::endl;

    cudaMemcpy(h2, d2, sizeof(int) * N, cudaMemcpyDefault);
    for(size_t i = 0; i < 10; i++) {
        std::cout << h2[i] << ", ";
    }
}