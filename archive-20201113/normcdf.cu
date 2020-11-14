#include <random>
#include <cassert>

__managed__ float x[100000][512];
__managed__ float out1[100000][512];
__managed__ float out2[100000][512];

__global__ void warmup() {
    out1[blockIdx.x][threadIdx.x] = x[blockIdx.x][threadIdx.x];
    out2[blockIdx.x][threadIdx.x] = out1[blockIdx.x][threadIdx.x];
    out1[blockIdx.x][threadIdx.x] = out2[blockIdx.x][threadIdx.x];
}

__global__ void normcdf_kernel() {
    out1[blockIdx.x][threadIdx.x] = normcdf(x[blockIdx.x][threadIdx.x]);
}

__global__ void normcdf_kernel2() {
    constexpr float invsqrt2 = 0.7071067811865475244008443621048490392848359376884740365883398689f;
    out2[blockIdx.x][threadIdx.x] = 0.5f * (1.0f + erf(invsqrt2 * x[blockIdx.x][threadIdx.x]));
}

int main() {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < 100000; i++) {
        for (int j = 0; j < 512; j++) {
            x[i][j] = distribution(generator);
        }
    }
    cudaDeviceSynchronize();
    warmup<<<100000, 512>>>();
    cudaDeviceSynchronize();
    normcdf_kernel2<<<100000, 512>>>();
    cudaDeviceSynchronize();
    normcdf_kernel<<<100000, 512>>>();
    cudaDeviceSynchronize();
    for (int i = 0; i < 100000; i++) {
        for (int j = 0; j < 512; j++) {
            assert(abs(out1[i][j] - out2[i][j]) < 1e-5);
        }
    }
}