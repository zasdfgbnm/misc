#include <tuple>

// This helper converts std::tuple<T1, T2, ....> to std::tuple<T1*, T2*, ....>
template <typename T>
struct pointers_helper {};

template <typename... types>
struct pointers_helper<std::tuple<types...>> {
  using type = std::tuple<types *...>;
};

template <typename T>
using pointers = typename pointers_helper<T>::type;

using t = pointers<std::tuple<int, int, bool>>;

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static void with_args(Args... args) {
    func<current>::apply(args...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static void with_args(Args... args) {}
};

template<int i>
struct func {
  static void apply(t &tt) {
    std::get<i>(tt) = 0;
  }
};

int main() {
  t tt;
  static_unroll<func, 3>::with_args(tt);
}