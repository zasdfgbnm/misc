int main() {
    void *p[10];
    cusolverDnHandle_t handle;
    cudaMalloc(&p[0], 100000);
    cudaMalloc(&p[1], 100000);
    cusolverDnCreate(handle);
    cudaMalloc(&p[2], 100000);
    cudaMalloc(&p[3], 100000);
}
