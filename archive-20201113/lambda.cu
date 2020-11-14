struct some_type {
    some_type() = delete;
};

int main() {
    auto l1 = [] __device__ (int param_a, int param_b) {  };  // replace with [](int param_a, int param_b) { };
    auto l2 = [] __device__ (int param_a, int param_b) -> some_type { };  // replace with [](int param_a, int param_b) -> some_type { };
}