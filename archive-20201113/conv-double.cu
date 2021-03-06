#include <stdexcept>
#include <iostream>
#include <initializer_list>
#include <cudnn_frontend.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

// using scalar_t = float;
// auto type = CUDNN_DATA_FLOAT;

// using scalar_t = double;
auto type = CUDNN_DATA_DOUBLE;

int64_t input_numel = 2 * 8 * 4 * 4;
std::array<int64_t, 4> input_sizes = { 2, 8, 4, 4 };
std::array<int64_t, 4> input_strides = { 128, 16, 4, 1 };

int64_t output_numel = 2 * 4 * 2 * 2;
std::array<int64_t, 4> output_sizes = { 2, 4, 2, 2 };
std::array<int64_t, 4> output_strides = { 16, 4, 2, 1 };

int64_t filter_numel = 4 * 8 * 3 * 3;
std::array<int64_t, 4> filter_sizes = { 4, 8, 3, 3 };
std::array<int64_t, 4> filter_strides = { 72, 9, 3, 1 };

int64_t ones[5] = {1, 1, 1, 1, 1};
int64_t zeros[5] = {0, 0, 0, 0, 0};

__global__ void assertEqual(
    int64_t size,
    float *destPtr,
    double *srcPtr
) {
    for (int64_t i = 0; i < sizes; i++) {
        scalar_t diff = std::abs(destPtr[i] - srcPtr[i]);
        printf("%f\n", diff);
        assert(diff < 1e-3);
    }
}

template<typename scalar_t>
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
    std::array<int64_t, 4> sizes,
    std::array<int64_t, 4> strides,

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

template<typename scalar_t>
void compute(
    scalar_t* devPtrX,
    scalar_t* devPtrW,
    scalar_t* devPtrY
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
        .setxDesc(getTensorDescriptor('x', input_sizes.data(), input_strides.data()))
        .setyDesc(getTensorDescriptor('y', output_sizes.data(), output_sizes.data()))
        .setwDesc(getTensorDescriptor('w', filter_sizes.data(), filter_strides.data()))
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
