#include <stdexcept>
#include <iostream>
#include <initializer_list>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

struct array4 {
    int64_t data_[4];
    __device__ __host__ int64_t &operator[](int64_t i) {
        return data_[i];
    }
    int64_t *data() {
        return data_;
    }
};

int64_t input_numel = 2 * 8 * 4 * 4;
array4 input_sizes = { 2, 8, 4, 4 };
array4 input_strides_nchw = { 128, 16, 4, 1 };
array4 input_strides_nhwc = { 128, 1, 32, 8 };

int64_t output_numel = 2 * 4 * 2 * 2;
array4 output_sizes = { 2, 4, 2, 2 };
array4 output_strides_nchw = { 16, 4, 2, 1 };
array4 output_strides_nhwc = { 16, 1, 8, 4 };

int64_t filter_numel = 4 * 8 * 3 * 3;
array4 filter_sizes = { 4, 8, 3, 3 };
array4 filter_strides_nchw = { 72, 9, 3, 1 };
array4 filter_strides_nhwc = { 72, 1, 24, 8 };

int64_t ones[5] = {1, 1, 1, 1, 1};
int64_t zeros[5] = {0, 0, 0, 0, 0};

__global__ void transpose(
    array4 sizes,
    float *destPtr,
    array4 destStrides,
    float *srcPtr,
    array4 srcStrides
) {
    for (int64_t i = 0; i < sizes[0]; i++) {
        for (int64_t j = 0; j < sizes[1]; j++) {
            for (int64_t k = 0; k < sizes[2]; k++) {
                for (int64_t l = 0; l < sizes[3]; l++) {
                    int64_t srcOffset = i * srcStrides[0] + j * srcStrides[1] + k * srcStrides[2] + l * srcStrides[3];
                    int64_t destOffset = i * destStrides[0] + j * destStrides[1] + k * destStrides[2] + l * destStrides[3];
                    printf("%ld, %ld, %ld, %ld, %ld, %ld, %f\n", i, j, k, l, srcOffset, destOffset, srcPtr[srcOffset]);
                    destPtr[destOffset] = srcPtr[srcOffset];
                }
            }
        }
    }
}

__global__ void assertEqual(
    array4 sizes,
    float *destPtr,
    array4 destStrides,
    float *srcPtr,
    array4 srcStrides
) {
    for (int64_t i = 0; i < sizes[0]; i++) {
        for (int64_t j = 0; j < sizes[1]; j++) {
            for (int64_t k = 0; k < sizes[2]; k++) {
                for (int64_t l = 0; l < sizes[3]; l++) {
                    int64_t srcOffset = i * srcStrides[0] + j * srcStrides[1] + k * srcStrides[2] + l * srcStrides[3];
                    int64_t destOffset = i * destStrides[0] + j * destStrides[1] + k * destStrides[2] + l * destStrides[3];
                    assert(destPtr[destOffset] == srcPtr[srcOffset]);
                }
            }
        }
    }
}

__global__ void initialize(int64_t size, float *ptr) {
    curandStatePhilox4_32_10_t state;
    curand_init(0, 0, 0, &state);
    for (int64_t i = 0; i < size; i++) {
        *ptr = curand_normal(&state);
    }
}
__global__ void read(int64_t size, float *ptr) {
    for (int64_t i = 0; i < size; i++) {
        printf("%f\n", *ptr);
    }
}
int main() {
    float* nchwPtrX = NULL;
    float* nchwPtrW = NULL;
    float* nchwPtrY = NULL;
    float* nhwcPtrX = NULL;
    float* nhwcPtrW = NULL;
    float* nhwcPtrY = NULL;
    cudaMalloc((void**)&(nchwPtrX), sizeof(float) * input_numel);
    cudaMalloc((void**)&(nchwPtrW), sizeof(float) * filter_numel);
    cudaMalloc((void**)&(nchwPtrY), sizeof(float) * output_numel);
    cudaMalloc((void**)&(nhwcPtrX), sizeof(float) * input_numel);
    cudaMalloc((void**)&(nhwcPtrW), sizeof(float) * filter_numel);
    cudaMalloc((void**)&(nhwcPtrY), sizeof(float) * output_numel);
    cudaDeviceSynchronize();

    initialize<<<1,1>>>(input_numel, nchwPtrX);
    initialize<<<1,1>>>(filter_numel, nchwPtrW);
    read<<<1,1>>>(input_numel, nchwPtrX);
    read<<<1,1>>>(filter_numel, nchwPtrW);
    cudaDeviceSynchronize();
    transpose<<<1,1>>>(input_sizes, nhwcPtrX, input_strides_nhwc, nchwPtrX, input_strides_nchw);
    transpose<<<1,1>>>(filter_sizes, nhwcPtrW, filter_strides_nhwc, nchwPtrW, filter_strides_nchw);
    assertEqual<<<1,1>>>(input_sizes, nhwcPtrX, input_strides_nhwc, nchwPtrX, input_strides_nchw);
    assertEqual<<<1,1>>>(filter_sizes, nhwcPtrW, filter_strides_nhwc, nchwPtrW, filter_strides_nchw);
    cudaDeviceSynchronize();
}
