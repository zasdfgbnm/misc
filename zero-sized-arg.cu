
struct A {
  float data[0];
};

struct B {
  int x;
  int y;
};

__global__ void unrolled_elementwise_kernel(int, int, A, B b)
{
  if (b.x == 0) {
    *(float *)nullptr = 0.0;
  }
}

void fill_kernel_cuda() {
  unrolled_elementwise_kernel<<<1, 1>>>(0, 0, A(), B());
}

