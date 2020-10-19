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

void checkCudnnErr(cudnnStatus_t code) {
    if (code) {
        throw std::runtime_error("error");
    }
}

int main() {
    cudnnHandle_t handle; checkCudnnErr(cudnnCreate(&handle));
    uint64_t convDim = 2;

    // auto type = CUDNN_DATA_FLOAT;
    auto type = CUDNN_DATA_HALF;

    auto x = cudnn_frontend::TensorBuilder()
        .setDim(5, input_shape)
        .setStrides(5, input_stride)
        .setId('x')
        .setAlignment(4)
        .setDataType(type)
        .build();
    auto y = cudnn_frontend::TensorBuilder()
        .setDim(5, output_shape)
        .setStrides(5, output_stride)
        .setId('y')
        .setAlignment(4)
        .setDataType(type)
        .build();
    auto w = cudnn_frontend::TensorBuilder()
        .setDim(5, filter_shape)
        .setStrides(5, filter_stride)
        .setId('w')
        .setAlignment(4)
        .setDataType(type)
        .build();
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
        .setxDesc(x)
        .setyDesc(y)
        .setwDesc(w)
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
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
        .setOperationGraph(opGraph)
        .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .build();

    auto &engine_config = heuristics.getEngineConfig();
    auto &fallback_list = fallback.getFallbackList();

    std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
    std::cout << "Fallback List has " << fallback_list.size() << " configurations " << std::endl;

    for (auto cfg_list : {&engine_config, &fallback_list}) {
        for (auto &cfg : *cfg_list) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(cfg)
                    .build();
                return 0;
            } catch (cudnn_frontend::cudnnException) {}
        }
    }
    std::cerr << "No engine found to do this computation" << std::endl;
    throw std::runtime_error("No engine found to do this computation");
}
