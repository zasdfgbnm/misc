#include <complex>
#include <iostream>
#include <chrono>

template<typename T>
inline std::complex<T> log1p_v1(const std::complex<T> &z) {
  // log1p(z) = log(1+z)
  // Let's define 1 + z = r*e^(i*a), then we have
  // log(r*e^(i*a)) = log(r) + i*a
  // but log(r) could have precision issue when |z| << 1, so we should really
  // be using log1p(r-1), where the r-1 should be computed in high precision.
  // to do so, we are doing the following transformation: (assuming z = x+iy)
  // r-1 = (r-1)*(r+1)/(r+1) = (r^2-1) / (r+1)
  //     = ((x+1)^2 + y^2 - 1) / (r+1)
  //     = (x^2 + y^2 + 2x) / (r+1)
  T x = z.real();
  T y = z.imag();
  std::complex<T> p1 = z + T(1);
  T r = std::abs(p1);
  T a = std::arg(p1);
  T rm1 = (r * r + x * T(2)) / (r + 1);
  return {std::log1p(rm1), a};
}

template<typename T>
inline std::complex<T> log1p_v2(const std::complex<T> &z) {
  return std::log(T(1) + z);
}

int main() {
  std::complex<float> input(0.5, 2.0);

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 10000; i++) {
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
    log1p_v1(input);
  }
  auto end = std::chrono::system_clock::now();
  auto elapsed = end - start;
  std::cout << "time for v1: " << elapsed.count() << '\n';


  start = std::chrono::system_clock::now();
  for (int i = 0; i < 10000; i++) {
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
    log1p_v2(input);
  }
  end = std::chrono::system_clock::now();
  elapsed = end - start;
  std::cout << "time for v2: " << elapsed.count() << '\n';
}
