#include <iostream>
#include <algorithm>
#include <cuda.h>
#include "FunctionTraits.h"

// template <typename func_t, typename array_t, std::size_t... I>
// __device__ inline constexpr decltype(auto) apply_impl(func_t f, array_t t, std::index_sequence<I...>)
// {
//     return f(t[I]...);
// }

// template <typename func_t, typename array_t>
// __device__ inline constexpr decltype(auto) array_apply(func_t f, array_t a) {
//   return apply_impl(f, a, std::make_index_sequence<1>{});
// }

template <typename func_t, typename array_t, std::enable_if_t<(function_traits<func_t>::arity == 0), int> = 0>
__device__ inline constexpr typename function_traits<func_t>::result_type array_apply(func_t f, array_t a) {
  return f();
}

template <typename func_t, typename array_t, std::enable_if_t<(function_traits<func_t>::arity == 1), int> = 0>
__device__ inline constexpr typename function_traits<func_t>::result_type array_apply(func_t f, array_t a) {
  return f(a[0]);
}

template <typename func_t, typename array_t, std::enable_if_t<(function_traits<func_t>::arity == 2), int> = 0>
__device__ inline constexpr typename function_traits<func_t>::result_type array_apply(func_t f, array_t a) {
  return f(a[0], a[1]);
}

template<typename func_t>
__global__ void elementwise_kernel(func_t f, float *output, float *input) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  float args[1] = { input[idx] };
  output[idx] = array_apply(f, args);
}

std::ostream &operator<<(std::ostream &o, float v[3]) {
  return o << v[0] << ", " << v[1] << ", " << v[2];
}

__managed__ float input[3] = { 1.0, 2.0, 3.0 };
__managed__ float output[3];
float expected_output[3];

int main() {
    cudaDeviceSynchronize();

    float a = 2.0;
    float b = 1.0;
    float c = 1.0;

    auto lambda = [=] __device__ __host__ (float x) -> float {
        // Uncomment the line below to get the correct result
        // for(int i=0; i < 100000; i++);
        return x <= 0 ? (x * c - 1) * a : x * b;
    };
    elementwise_kernel<<<1, 3>>>(lambda, output, input);

    cudaDeviceSynchronize();

    std::transform(input, input + 3, expected_output, lambda);
    std::cout << "output: " << output << std::endl;
    std::cout << "expect: " << expected_output << std::endl;
}