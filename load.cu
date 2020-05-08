template<typename func_t>
struct LoadArgsWithDynamicCast {
  using traits = function_traits<func_t>;
  ScalarType args_types[traits::arity];  // this is a zero array when func_t is a nullary function!!!

  void load(void *addresses[]) {
    #pragma unroll
    for(int i = 0; i < traits::arity; i++) {
      switch (args_types[i]) {
      case kFloat:
        static_cast<traits::arg<i>::type>(*static_cast<float *>(addresses[i]));
        break;
      case kInt:
        static_cast<traits::arg<i>::type>(*static_cast<int *>(addresses[i]));
        break;
      case ....
    }
  }
};

template<typename load_policy_t>
__global__ some_kernel(void *addresses[], load_policy_t policy) {
  policy.load(addresses);
  // do something else...
}