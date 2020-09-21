#include <cuda_bf16.h>

__global__ void kernel(__nv_bfloat16 *p) {
    __ldg(p);
}

