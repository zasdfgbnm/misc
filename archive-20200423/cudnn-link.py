import torch
import torch.nn as nn
from torch.utils import cpp_extension
torch.backends.cudnn.deterministic = True
cpp_source = """
    #include <cudnn.h>
    #include <ATen/cuda/Exceptions.h> // for CUDNN_CHECK
    #include <ATen/cudnn/Descriptors.h> // for TensorDescriptor
    #include <ATen/cudnn/Handle.h> // for getCudnnHandle
    #include <limits>
    #include <vector>
    #include <functional>
    #include <ATen/ATen.h>
    #include <ATen/NativeFunctions.h>
    #include <ATen/Config.h>
    #include <ATen/cuda/CUDAConfig.h>
    #include <ATen/cuda/Exceptions.h>
    #include <ATen/native/ConvUtils.h>
    void my_fun(const torch::Tensor& grad_weight, const torch::Tensor& grad_output, const torch::Tensor& input,
    at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    auto handle = torch::native::getCudnnHandle();
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;//torch::native::getCudnnDataType(input);
    at::native::Constant one(dataType, 1);
    at::native::Constant zero(dataType, 0);
    torch::Tensor workspace = at::empty({static_cast<int64_t>(100*1024*1024*4)}, input.options().dtype(at::kByte));
    torch::native::TensorDescriptor idesc(input, 4);
    torch::native::FilterDescriptor wdesc;
    wdesc.set(grad_weight, 0, false);
    torch::native::TensorDescriptor odesc(grad_output, 4);
    torch::native::ConvolutionDescriptor cdesc;
    int pad[2] = {1, 1};
    int str[2] = {1, 1};
    int dil[2] = {1, 1};
    cdesc.set(dataType, input.dim() - 2, pad, str, dil, groups);
    AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        handle,
        &one, idesc.desc(), input.data_ptr(),
        odesc.desc(), grad_output.data_ptr(),
        cdesc.desc(), CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, workspace.data_ptr(), 100*1024*1024*4,
        &zero, wdesc.desc(), grad_weight.data_ptr()));
    return;
    }
"""
if __name__=="__main__":
    module = torch.utils.cpp_extension.load_inline(
        name="cuda_test_extension",
        cpp_sources=cpp_source,
        #cuda_sources=cuda_source,
        with_cuda=True,
        extra_ldflags=["-lcudnn"],
        functions="my_fun",
        verbose=True,
    )
    grad_output = torch.load("grad_output.zip").cuda()
    input = torch.load("input.zip").cuda()
    grad_weight = torch.zeros_like(grad_output);
    out = module.my_fun(grad_weight, grad_output, input, {1, 1}, {1, 1}, {1, 1}, 1, False, True)
    print(out)