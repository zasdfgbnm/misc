#include <type_traits>
#include <iostream>


template <typename scalar_t>
struct has_builtin_vector_type : public std::false_type {};

template <> struct has_builtin_vector_type<uint8_t> : public std::true_type {};
template <> struct has_builtin_vector_type<int8_t>  : public std::true_type {};
template <> struct has_builtin_vector_type<int16_t> : public std::true_type {};
template <> struct has_builtin_vector_type<int>     : public std::true_type {};
template <> struct has_builtin_vector_type<int64_t> : public std::true_type {};
template <> struct has_builtin_vector_type<float>   : public std::true_type {};
template <> struct has_builtin_vector_type<double>  : public std::true_type {};

template<typename scalar_t, bool>
struct can_vectorize_up_to_impl;

template<typename scalar_t>
struct can_vectorize_up_to_impl<scalar_t, false> {
  static constexpr inline int get(char *pointer) {
    std::cout << "mindlessly return 1" << std::endl;
    return 1;
  }
};

template<typename scalar_t>
struct can_vectorize_up_to_impl<scalar_t, true> {
  static constexpr inline int get(char *pointer) {
    uint64_t address = reinterpret_cast<uint64_t>(pointer);
    std::cout << "address = " << address << std::endl;
    return 1;
  }
};

template<typename scalar_t>
inline int can_vectorize_up_to(char *pointer) {
  std::cout << "has_builtin_vector_type<scalar_t>::value = " << has_builtin_vector_type<scalar_t>::value << std::endl;
  return can_vectorize_up_to_impl<scalar_t, has_builtin_vector_type<scalar_t>::value>::get(pointer);
}

char p;

int main() {
  std::cout << can_vectorize_up_to<uint8_t>(&p) << std::endl;
}