#include <stdexcept>
#include <iostream>
#include <cudnn_frontend.h>

int64_t ones[5] = {1, 1, 1, 1, 1};
int64_t zeros[5] = {0, 0, 0, 0, 0};

int64_t input_shape[5] = {2, 2, 2, 6, 6};
int64_t input_stride[5] = {144, 72, 36, 6, 1};

int64_t output_shape[5] = {2, 2, 2, 4, 4};
int64_t output_stride[5] = {64, 32, 16, 4, 1};

int64_t filter_shape[5] = {2, 2, 2, 3, 3};
int64_t filter_stride[5] = {36, 18, 9, 3, 1};

int main() {
    auto type = CUDNN_DATA_DOUBLE;

    auto x = cudnn_frontend::TensorBuilder()
        .setDim(5, input_shape)
        .setStrides(5, input_stride)
        .setId('x')
        .setAlignment(8)
        .setDataType(type)
        .build();
}
