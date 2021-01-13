#include <stdexcept>
#include <iostream>
#include <cudnn_frontend.h>

const int64_t batch = 4096;  // 4095 works fine
int64_t ones[5] = {1, 1, 1, 1, 1};
int64_t zeros[5] = {0, 0, 0, 0, 0};
int64_t input_shape[5] = {batch, 2, 512, 512};
int64_t input_stride[5] = {524288, 262144, 512, 1};
int64_t output_shape[5] = {batch, 2, 64, 64};
int64_t output_stride[5] = {8192, 4096, 64, 1};
int64_t filter_shape[5] = {2, 2, 8, 8};
int64_t filter_stride[5] = {128, 64, 8, 1};
int64_t stride[2] = {8, 8};

void checkCudnnErr(cudnnStatus_t code) {
    if (code) {
        throw std::runtime_error("error");
    }
}

int main() {
    cudnnHandle_t handle; checkCudnnErr(cudnnCreate(&handle));
    uint64_t convDim = 2;
    auto type = CUDNN_DATA_FLOAT;
    auto x = cudnn_frontend::TensorBuilder()
        .setDim(4, input_shape)
        .setStrides(4, input_stride)
        .setId('x')
        .setAlignment(4)
        .setDataType(type)
        .build();
    auto y = cudnn_frontend::TensorBuilder()
        .setDim(4, output_shape)
        .setStrides(4, output_stride)
        .setId('y')
        .setAlignment(4)
        .setDataType(type)
        .build();
    auto w = cudnn_frontend::TensorBuilder()
        .setDim(4, filter_shape)
        .setStrides(4, filter_stride)
        .setId('w')
        .setAlignment(4)
        .setDataType(type)
        .build();
    auto conv_descriptor = cudnn_frontend::ConvDescBuilder()
        .setDataType(type)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setNDims(convDim)
        .setStrides(convDim, stride)
        .setPrePadding(convDim, zeros)
        .setPostPadding(convDim, zeros)
        .setDilation(convDim, ones)
        .build();
    std::cout << conv_descriptor.describe() << std::endl;
    auto op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .setxDesc(x)
        .setyDesc(y)
        .setwDesc(w)
        .setcDesc(conv_descriptor)
        .setAlpha(1.0f)
        .setBeta(0.0f)
        .build();
}