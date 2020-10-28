#include <stdexcept>
#include <iostream>
#include <initializer_list>
#include <cudnn_frontend.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

using scalar_t = float;
auto type = CUDNN_DATA_FLOAT;

// using scalar_t = double;
// auto type = CUDNN_DATA_DOUBLE;

struct array4 {
    int64_t data_[4];
    __device__ __host__ int64_t &operator[](int64_t i) {
        return data_[i];
    }
    int64_t *data() {
        return data_;
    }
};

int64_t N = 4;
int64_t C = 2;
int64_t H = 8;
int64_t W = 8;
int64_t K = 4;
int64_t R = 2;
int64_t S = 2;

array4 input_sizes = { N, C, H, W };
int64_t input_numel = N * C * H * W;
array4 input_strides_nchw = { C * H * W, H * W, W, 1 };
array4 input_strides_nhwc = { C * H * W, 1, C * W, C };

array4 filter_sizes = { K, C, R, S };
int64_t filter_numel = K * C * R * S;
array4 filter_strides_nchw = { C * R * S, R * S, S, 1 };
array4 filter_strides_nhwc = { C * R * S, 1, C * S, C };

array4 output_sizes = { N, K, H - R + 1, W - S + 1 };
int64_t output_numel = output_sizes[0] * output_sizes[1] * output_sizes[2] * output_sizes[3];
array4 output_strides_nchw = { output_sizes[1] * output_sizes[2] * output_sizes[3], output_sizes[2] * output_sizes[3], output_sizes[3], 1 };
array4 output_strides_nhwc = { output_sizes[1] * output_sizes[2] * output_sizes[3], 1, output_sizes[1] * output_sizes[3], output_sizes[1] };

int64_t ones[5] = {1, 1, 1, 1, 1};
int64_t zeros[5] = {0, 0, 0, 0, 0};

__global__ void transpose(
    array4 sizes,
    scalar_t *destPtr,
    array4 destStrides,
    scalar_t *srcPtr,
    array4 srcStrides
) {
    for (int64_t i = 0; i < sizes[0]; i++) {
        for (int64_t j = 0; j < sizes[1]; j++) {
            for (int64_t k = 0; k < sizes[2]; k++) {
                for (int64_t l = 0; l < sizes[3]; l++) {
                    int64_t srcOffset = i * srcStrides[0] + j * srcStrides[1] + k * srcStrides[2] + l * srcStrides[3];
                    int64_t destOffset = i * destStrides[0] + j * destStrides[1] + k * destStrides[2] + l * destStrides[3];
                    printf("%ld, %ld, %ld, %ld, %ld, %ld, %f\n", i, j, k, l, srcOffset, destOffset, (float)srcPtr[srcOffset]);
                    destPtr[destOffset] = srcPtr[srcOffset];
                }
            }
        }
    }
}

__global__ void assertEqual(
    array4 sizes,
    scalar_t *destPtr,
    array4 destStrides,
    scalar_t *srcPtr,
    array4 srcStrides
) {
    for (int64_t i = 0; i < sizes[0]; i++) {
        for (int64_t j = 0; j < sizes[1]; j++) {
            for (int64_t k = 0; k < sizes[2]; k++) {
                for (int64_t l = 0; l < sizes[3]; l++) {
                    int64_t srcOffset = i * srcStrides[0] + j * srcStrides[1] + k * srcStrides[2] + l * srcStrides[3];
                    int64_t destOffset = i * destStrides[0] + j * destStrides[1] + k * destStrides[2] + l * destStrides[3];
                    scalar_t diff = std::abs(destPtr[destOffset] - srcPtr[srcOffset]);
                    // printf("%f\n", diff);
                    assert(diff < 1e-3);
                }
            }
        }
    }
}

__global__ void initialize(int64_t size, scalar_t *ptr) {
    curandStatePhilox4_32_10_t state;
    curand_init(0, 0, 0, &state);
    for (int64_t i = 0; i < size; i++) {
        ptr[i] = curand_uniform(&state);
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
        .setDataType(type)
        .build();
    std::cout << tensor.describe() << std::endl;
    return tensor;
}

void compute(
    scalar_t* devPtrX,
    array4 sizesX,
    array4 stridesX,
    scalar_t* devPtrW,
    array4 sizesW,
    array4 stridesW,
    scalar_t* devPtrY,
    array4 sizesY,
    array4 stridesY
) {
    cudaDeviceSynchronize();
    uint64_t convDim = 2;
    cudnnHandle_t handle; checkCudnnErr(cudnnCreate(&handle));
    auto conv_descriptor = cudnn_frontend::ConvDescBuilder()
        .setDataType(type)
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

    void * workspace_ptr;
    cudaMalloc(&workspace_ptr, 100000);

    for (auto &cfg : engine_config) {
        try {
            auto plan = cudnn_frontend::ExecutionPlanBuilder()
                .setHandle(handle)
                .setEngineConfig(cfg)
                .build();

            void * data_ptrs[] = {devPtrX, devPtrY, devPtrW};
            int64_t uids[] = {'x', 'y', 'w'};
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                .setDataPointers(3, data_ptrs)
                .setUids(3, uids)
                .build();

            checkCudnnErr(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
            cudaDeviceSynchronize();
        } catch(...) {}
    }
    throw "no algo found";
}

int main() {
    scalar_t* nchwPtrX = NULL;
    scalar_t* nchwPtrW = NULL;
    scalar_t* nchwPtrY = NULL;
    scalar_t* nhwcPtrX = NULL;
    scalar_t* nhwcPtrW = NULL;
    scalar_t* nhwcPtrY = NULL;
    cudaMalloc((void**)&(nchwPtrX), sizeof(scalar_t) * input_numel);
    cudaMalloc((void**)&(nchwPtrW), sizeof(scalar_t) * filter_numel);
    cudaMalloc((void**)&(nchwPtrY), sizeof(scalar_t) * output_numel);
    cudaMalloc((void**)&(nhwcPtrX), sizeof(scalar_t) * input_numel);
    cudaMalloc((void**)&(nhwcPtrW), sizeof(scalar_t) * filter_numel);
    cudaMalloc((void**)&(nhwcPtrY), sizeof(scalar_t) * output_numel);
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
