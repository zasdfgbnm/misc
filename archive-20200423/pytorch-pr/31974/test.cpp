#include "FunctionTraits.h"

template <typename func_t, std::size_t arity>
struct arg_type_helper {
  using type = typename function_traits<func_t>::template arg<0>::type;
};

template <typename func_t>
struct arg_type_helper<func_t, 0> {
  using type = decltype(nullptr);
};

template <typename func_t>
using arg_type = typename arg_type_helper<func_t, function_traits<func_t>::arity>::type;

int main() {
  auto l0 = []() {return 0;};
  auto l1 = [](int) {return 0;};
  using t0 = arg_type<decltype(l0)>;
  using t1 = arg_type<decltype(l1)>;
}