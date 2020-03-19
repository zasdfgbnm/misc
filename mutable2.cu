template<typename func_t>
__host__ __device__ void invoke_impl(func_t &f) {
    f();
}

template<typename func_t>
__host__ __device__ void invoke(func_t &f) {
    invoke_impl(f);
}

template<typename func_t>
__global__ void kernel(func_t g) {
    g();
}

template<typename func_t>
void launch_kernel(func_t g) {
    kernel<<<1, 1>>>(g);
}

template<typename func_t>
void gpu_kernel_impl(func_t f) {
    launch_kernel([=]__host__ __device__() mutable {
        invoke<func_t>(f);
    });
}

int main() {
    int a = 0;
    gpu_kernel_impl([=]__host__ __device__() mutable { return a++; });
}