template<typename func_t>
__global__ void elementwise_kernel(func_t f) {
  f();
}

template<typename func_t>
void launch_kernel(func_t f) {
  elementwise_kernel<func_t><<<1, 1>>>(f);
}

template<typename func_t>
__host__ __device__ float invoke_impl(func_t &f) {
  return f();
}

template<typename func_t>
__host__ __device__ float invoke(func_t &f) {
  return invoke_impl(f);
}

template<typename func_t>
void gpu_kernel_impl(func_t f) {

  launch_kernel([=]__host__ __device__() mutable {
    float* out = nullptr;
    *out = invoke<func_t>(f);
  });
}

int main() {
    float thread_work_index_ = 0;
    gpu_kernel_impl([thread_work_index_]() mutable -> float {
      return thread_work_index_++;
    });
}
