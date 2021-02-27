template <typename T>
__global__ void kernel(T *ptr) {
    *ptr = ::sqrt(*ptr);
}

int main() {
    kernel<<<1, 1>>>((char *)nullptr);
    kernel<<<1, 1>>>((unsigned char *)nullptr);
    kernel<<<1, 1>>>((short *)nullptr);
    kernel<<<1, 1>>>((int *)nullptr);
    kernel<<<1, 1>>>((long long *)nullptr);
    kernel<<<1, 1>>>((float *)nullptr);
    kernel<<<1, 1>>>((double *)nullptr);
}

/*
sqrt.cu(3): error: calling a __host__ function("double  ::sqrt<char, (int)0> (T1)") from a __global__ function("kernel<char> ") is not allowed

sqrt.cu(3): error: identifier "sqrt<char, (int)0> " is undefined in device code

sqrt.cu(3): error: calling a __host__ function("double  ::sqrt<unsigned char, (int)0> (T1)") from a __global__ function("kernel<unsigned char> ") is not allowed

sqrt.cu(3): error: identifier "sqrt<unsigned char, (int)0> " is undefined in device code

sqrt.cu(3): error: calling a __host__ function("double  ::sqrt<short, (int)0> (T1)") from a __global__ function("kernel<short> ") is not allowed

sqrt.cu(3): error: identifier "sqrt<short, (int)0> " is undefined in device code

sqrt.cu(3): error: calling a __host__ function("double  ::sqrt<int, (int)0> (T1)") from a __global__ function("kernel<int> ") is not allowed

sqrt.cu(3): error: identifier "sqrt<int, (int)0> " is undefined in device code

sqrt.cu(3): error: calling a __host__ function("double  ::sqrt<long long, (int)0> (T1)") from a __global__ function("kernel<long long> ") is not allowed

sqrt.cu(3): error: identifier "sqrt<long long, (int)0> " is undefined in device code

10 errors detected in the compilation of "sqrt.cu".
*/