template<typename scalar_t, typename mask_t>
__global__ masked_select_kernel(
      scalar_t* result_ptr,
      scalar_t* self_ptr,
      mask_t* mask_ptr,
      int64_t* mask_inclusive_scan_ptr,
      int64_t num_input_elements
) {
    mask_t mask[thread_work_size];
    int64_t base_index = block_work_size * blockIdx.x;
    int remaining = std::min<int64_t>(num_input_elements - base_index, block_work_size);

    // load data into registers
    int thread_idx = threadIdx.x;
    #pragma unroll
    for(int i = 0; i < thread_work_size; i++) {
        if (thread_idx >= remaining) {
          break;
        }
        int64_t input_idx = thread_idx + base_index;
        mask[i] = mask_ptr[input_idx];
        thread_idx += num_threads;
    }

    // compute and store
    int thread_idx = threadIdx.x;
    #pragma unroll
    for(int i = 0; i < thread_work_size; i++) {
        if (thread_idx >= remaining) {
            break;
        }
        if (mask[i]) {
            int64_t input_idx = thread_idx + base_index;
            int64_t result_idx = mask_inclusive_scan_ptr[input_idx] - 1;
            result_ptr[result_idx] = self_ptr[input_idx];
        }
        thread_idx += num_threads;
    }
}

auto stream = at::cuda::getCurrentCUDAStream();
int64_t grid = (N + block_work_size - 1) / block_work_size;
masked_select_kernel<<<grid, num_threads, 0, stream>>>(.....);
