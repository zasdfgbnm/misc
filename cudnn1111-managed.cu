#include <stdexcept>
#include <iostream>
#include <cudnn_frontend.h>

float x[1] = {0.12f};
float w[1] = {3.4f};
float y[1] = {0.99f};
float workspace[100000000];

int64_t ones[4] = {1, 1, 1, 1};
int64_t zeros[4] = {0, 0, 0, 0};

void checkCudnnErr(cudnnStatus_t code) {
    if (code) {
        throw std::runtime_error("error");
    }
}

cudnn_frontend::Tensor getTensorDescriptor(int64_t id) {
    auto tensor = cudnn_frontend::TensorBuilder()
    .setDim(4, ones)
    .setStrides(4, ones)
    .setId(id)
    .setAlignment(4)
    .setDataType(CUDNN_DATA_FLOAT)
    .build();
    std::cout << tensor.describe() << std::endl;
    return tensor;
}

int main() {
    cudnnHandle_t handle; checkCudnnErr(cudnnCreate(&handle));

    try {
    uint64_t convDim = 2;
    float* devPtrX = NULL;
    float* devPtrW = NULL;
    float* devPtrY = NULL;

    cudaMalloc((void**)&(devPtrX), sizeof(float));
    cudaMalloc((void**)&(devPtrW), sizeof(float));
    cudaMalloc((void**)&(devPtrY), sizeof(float));
    cudaMemcpy(devPtrX, x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrW, w, sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

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
        .setxDesc(getTensorDescriptor('x'))
        .setyDesc(getTensorDescriptor('y'))
        .setwDesc(getTensorDescriptor('w'))
        .setcDesc(conv_descriptor)
        .setAlpha(1.0)
        .setBeta(0.0)
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

    auto &engine_config = heuristics.getEngineConfig();

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
    cudaMemcpy(y, devPtrY, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "expect: " << x[0] * w[0] << std::endl
            << "get: " << y[0] << std::endl;
    } catch (cudnn_frontend::cudnnException e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }
}
