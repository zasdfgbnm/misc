// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <type_traits>
#include <tuple>

#include "FunctionTraits.h"

// Marks a lambda as executable on both the host and device. The __host__
// attribute is important so that we can access static type information from
// the host, even if the function is typically only executed on the device.
#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

#define C10_HOST_DEVICE __host__ __device__
#define C10_RESTRICT __restrict__

static constexpr int launch_size_1d = 512;

using ScalarType = int;

// NOTE: @zasdfgbnm is currently working on rewriting the gpu loops.
// Some of the old codes has been moved to namespace legacy, and
// new codes will be put into namespace modern. These two namespaces
// will coexists for a while until the rewrite is done. Once the rewrite
// is done, we will remove the legacy and modern namespace and everything
// will be in at::native directly.
namespace legacy {

template<int nt, int vt, typename func_t>
__global__ void elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_kernel(int64_t N, func_t f) {
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  elementwise_kernel<nt, vt, func_t><<<grid, block>>>(N, f);
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type
invoke_impl(func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i,
            std::index_sequence<INDEX...>) {
  return f();
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type
invoke(func_t &f, char *const C10_RESTRICT data[], const index_t strides[], int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

} // namespace legacy


template <typename func_t>
void gpu_kernel_impl(func_t f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  char *const data[ntensors] = {nullptr};

  int64_t numel = 0;
  if (true) {
    int strides[ntensors];
    for (int i = 0; i < ntensors; i++) {
      strides[i] = 0;
    }

    legacy::launch_kernel<launch_size_1d, 1>(numel, [=]GPU_LAMBDA(int idx) mutable {
        arg0_t* out = (arg0_t*)(data[0] + strides[0] * idx);
        *out = legacy::invoke<func_t>(f, &data.data[1], &strides.data[1], idx);
    });
  }
}

int main() {
    float thread_work_index_ = 0;
    gpu_kernel_impl([thread_work_index_]() mutable -> float {
      return thread_work_index_++;
    });
}
