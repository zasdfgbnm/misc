#include <stdexcept>
#include <iostream>
#include <cudnn_frontend.h>

__managed__ float x[1] = {0.12};
__managed__ float w[1] = {3.4};
__managed__ float y[1] = {0};
__managed__ float workspace[100000000];

int64_t ones[] = {1, 1, 1, 1, 1};
int64_t zeros[] = {0, 0, 0, 0, 0};

void checkCudnnErr(cudnnStatus_t code) {
    if (code) {
        throw std::runtime_error("error");
    }
}

cudnn_frontend::Tensor getTensorDescriptor(int64_t id) {
    int64_t nDim = 5;
    return cudnn_frontend::TensorBuilder()
      .setDim(nDim, ones)
      .setStrides(nDim, ones)
      .setId(id)
      .setAlignment(4)
      .setDataType(CUDNN_DATA_FLOAT)
      .build();
  }

int main() {
    cudnnHandle_t handle; checkCudnnErr(cudnnCreate(&handle));

    uint64_t convDim = 2;
    auto conv_descriptor = cudnn_frontend::ConvDescBuilder()
        .setDataType(CUDNN_DATA_FLOAT)
        .setMathMode(CUDNN_CROSS_CORRELATION)
        .setNDims(convDim)
        .setStrides(convDim, ones)
        .setPrePadding(convDim, zeros)
        .setPostPadding(convDim, zeros)
        .setDilation(convDim, ones)
        .build();
  
    auto op = cudnn_frontend::OperationBuilder()
        .setxDesc(getTensorDescriptor('x'))
        .setyDesc(getTensorDescriptor('y'))
        .setwDesc(getTensorDescriptor('w'))
        .setcDesc(conv_descriptor)
        .setAlpha(1.0)
        .setBeta(0.0)
        .setOpMode(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .build();
  
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
  
    void * data_ptrs[] = {x, y, w};
    int64_t uids[] = {'x', 'y', 'w'};
    auto variantPack = cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace)
        .setDataPointers(3, data_ptrs)
        .setUids(3, uids)
        .build();

    checkCudnnErr(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));

    cudaDeviceSynchronize();
    std::cout << "expect: " << x[0] * w[0] << std::endl
              << "get: " << y[0] << std::endl;
}