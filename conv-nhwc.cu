#include <stdexcept>
#include <iostream>
#include <initializer_list>
#include <cudnn_frontend.h>
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
                    // printf("%ld, %ld, %ld, %ld, %ld, %ld, %f\n", i, j, k, l, srcOffset, destOffset, srcPtr[srcOffset]);
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
                    float diff = std::abs(destPtr[destOffset] - srcPtr[srcOffset]);
                    printf("%f\n", diff);
                    assert(diff < 1e-3);
                }
            }
        }
    }
}

__device__ float to3(float r01) {
    if (r01 < 1.0 / 7) {
        return -3;
    }
    if (r01 < 2.0 / 7) {
        return -2;
    }
    if (r01 < 3.0 / 7) {
        return -1;
    }
    if (r01 < 4.0 / 7) {
        return -0;
    }
    if (r01 < 5.0 / 7) {
        return 1;
    }
    if (r01 < 6.0 / 7) {
        return 2;
    }
    return 3;
}

__global__ void initialize(int64_t size, float *ptr) {
    curandStatePhilox4_32_10_t state;
    curand_init(0, 0, 0, &state);
    for (int64_t i = 0; i < size; i++) {
        ptr[i] = to3(curand_uniform(&state));
    }
}

void checkCudnnErr(cudnnStatus_t code) {
    if (code) {
        throw std::runtime_error("error");
    }
}

cudnn_frontend::Tensor getTensorDescriptor(
    int64_t id,
    array4 sizes,
    array4 strides
) {
    auto tensor = cudnn_frontend::TensorBuilder()
        .setDim(4, sizes.data())
        .setStrides(4, strides.data())
        .setId(id)
        .setAlignment(8)
        .setDataType(CUDNN_DATA_FLOAT)
        .build();
    std::cout << tensor.describe() << std::endl;
    return tensor;
}

void compute(
    float* devPtrX,
    array4 sizesX,
    array4 stridesX,
    float* devPtrW,
    array4 sizesW,
    array4 stridesW,
    float* devPtrY,
    array4 sizesY,
    array4 stridesY
) {
    cudaDeviceSynchronize();
    uint64_t convDim = 2;
    cudnnHandle_t handle; checkCudnnErr(cudnnCreate(&handle));
    auto conv_descriptor = cudnn_frontend::ConvDescBuilder()
        .setDataType(CUDNN_DATA_FLOAT)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setNDims(convDim)
        .setStrides(convDim, ones)
        .setPrePadding(convDim, zeros)
        .setPostPadding(convDim, zeros)
        .setDilation(convDim, ones)
        .build();
    std::cout << conv_descriptor.describe() << std::endl;

    auto op = cudnn_frontend::OperationBuilder()
        .setOpMode(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .setxDesc(getTensorDescriptor('x', sizesX, stridesX))
        .setyDesc(getTensorDescriptor('y', sizesY, stridesY))
        .setwDesc(getTensorDescriptor('w', sizesW, stridesW))
        .setcDesc(conv_descriptor)
        .setAlpha(1.0f)
        .setBeta(0.0f)
        .build();
    std::cout << op.describe() << std::endl;

    std::array<cudnn_frontend::Operation const *, 1> ops = {&op};

    auto opGraph = cudnn_frontend::OperationGraphBuilder()
        .setHandle(handle)
        .setOperationGraph(1, ops.data())
        .build();

    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
        .setOperationGraph(opGraph)
        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
        .build();

    auto &engine_config = heuristics.getEngineConfig(100000);

    auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(engine_config[0])
        .build();

    void * data_ptrs[] = {devPtrX, devPtrY, devPtrW};
    int64_t uids[] = {'x', 'y', 'w'};
    void * workspace_ptr;
    cudaMalloc(&workspace_ptr, 100000);
    auto variantPack = cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace_ptr)
        .setDataPointers(3, data_ptrs)
        .setUids(3, uids)
        .build();

    checkCudnnErr(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
    transpose<<<1,1>>>(input_sizes, nhwcPtrX, input_strides_nhwc, nchwPtrX, input_strides_nchw);
    transpose<<<1,1>>>(filter_sizes, nhwcPtrW, filter_strides_nhwc, nchwPtrW, filter_strides_nchw);
    assertEqual<<<1,1>>>(input_sizes, nhwcPtrX, input_strides_nhwc, nchwPtrX, input_strides_nchw);
    assertEqual<<<1,1>>>(filter_sizes, nhwcPtrW, filter_strides_nhwc, nchwPtrW, filter_strides_nchw);

    compute(
        nhwcPtrX, input_sizes, input_strides_nhwc,
        nhwcPtrW, filter_sizes, filter_strides_nhwc,
        nhwcPtrY, output_sizes, output_strides_nhwc
    );
    compute(
        nchwPtrX, input_sizes, input_strides_nchw,
        nchwPtrW, filter_sizes, filter_strides_nchw,
        nchwPtrY, output_sizes, output_strides_nchw
    );

    assertEqual<<<1,1>>>(output_sizes, nhwcPtrY, output_strides_nhwc, nchwPtrY, output_strides_nchw);

    cudaDeviceSynchronize();
    std::cout << "match" << std::endl;
}
