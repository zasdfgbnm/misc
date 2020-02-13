#include <functional>

// `drop_first_arg` takes a function type as template argument
// and returns the function type with first argument dropped
// The implementation is copied from ATen/detail/FunctionTraits.h

template <typename T>
struct drop_first_arg_helper : public drop_first_arg_helper<decltype(&T::operator())> {
};

template <typename ClassType, typename T>
struct drop_first_arg_helper<T ClassType::*> : public drop_first_arg_helper<T> {
};

template <typename ClassType, typename ReturnType, typename... Args>
struct drop_first_arg_helper<ReturnType(ClassType::*)(Args...) const> : public drop_first_arg_helper<ReturnType(Args...)> {
};

template<typename return_type, typename first, typename... arg_types>
struct drop_first_arg_helper<return_type (first, arg_types...)> {
  using type = return_type (arg_types...);
};

template <typename func_t>
using drop_first_arg = typename drop_first_arg_helper<func_t>::type;

int fff(int, int);

int main() {
  int a;
  auto f = [&](int, int) { return a; };
  using ff = drop_first_arg<decltype(f)>;
}