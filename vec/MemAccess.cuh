#pragma once

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace mem {

template <typename scalar_t, int size>
struct Info;

#define DEFINE_VECTOR_INFO(TYPE, SIZE, VECTYPE, ALIGNMENT)    \
  template <>                                                 \
  struct Info<TYPE, SIZE> {                                   \
    static constexpr int alignment = ALIGNMENT;               \
    using vector_type = VECTYPE;                              \
  }


// Alignment data could be found at:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types

//                           TYPE,   SIZE,    VECTYPE, ALIGNMENT
DEFINE_VECTOR_INFO(          char,      2,      char2,         2);
DEFINE_VECTOR_INFO(unsigned  char,      2,     uchar2,         2);
DEFINE_VECTOR_INFO(          char,      4,      char4,         4);
DEFINE_VECTOR_INFO(unsigned  char,      4,     uchar4,         4);

DEFINE_VECTOR_INFO(         short,      2,     short2,         4);
DEFINE_VECTOR_INFO(unsigned short,      2,    ushort2,         4);
DEFINE_VECTOR_INFO(         short,      4,     short4,         8);
DEFINE_VECTOR_INFO(unsigned short,      4,    ushort4,         8);

DEFINE_VECTOR_INFO(           int,      2,       int2,         8);
DEFINE_VECTOR_INFO(unsigned   int,      2,      uint2,         8);
DEFINE_VECTOR_INFO(           int,      4,       int4,        16);
DEFINE_VECTOR_INFO(unsigned   int,      4,      uint4,        16);

DEFINE_VECTOR_INFO(          long,      2,      long2,        16);
DEFINE_VECTOR_INFO(unsigned  long,      2,     ulong2,        16);
DEFINE_VECTOR_INFO(          long,      4,      long4,        16);
DEFINE_VECTOR_INFO(unsigned  long,      4,     ulong4,        16);

DEFINE_VECTOR_INFO(         float,      2,     float2,         8);
DEFINE_VECTOR_INFO(         float,      4,     float4,        16);

DEFINE_VECTOR_INFO(        double,      2,    double2,        16);
DEFINE_VECTOR_INFO(        double,      4,    double4,        16);

#undef DEFINE_VECTOR_INFO

namespace policy {

template <
  typename scalar_t,  // type of data
  int num_threads,    // number of threads in a block
  int size            // number of elements each thread needs to handle if it is not on boundary
>
struct Policies {
  static_assert(size % 4 == 0, "load size for each block has to be a multiple of number of threads in a block");
  const expr

  struct Unroll {

    // load data from 
    __device__ inline void load(scalar_t to[size], scalar_t *from, int real_size) {
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < size; i++) {
        int index = thread_idx + i * blockDim.x;
        if (index < boundary) {
          to[index] = from[index];
        }
      }
    }

    __device__ inline void store(scalar_t *to, scalar_t from[size], int real_size) {
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < size; i++) {
        int index = thread_idx + i * blockDim.x;
        if (index < boundary) {
          to[i] = from[i];
        }
      }
    }
  };

  template <int vec_size>
  struct UnrollWithVec {

    __device__ inline void load(scalar_t to[size], scalar_t *from, int real_size) {
      using vec_t = typename Info<scalar_t, vec_size>::vector_type;
      vec_t *from_ = reinterpret_cast<vec_t *>(from);
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < size / vec_size; i++) {
        int index = thread_idx + i * blockDim.x;
        (vec_t [size / vec_size])to[i] = from_[i];
      }
    }

    __device__ void store(scalar_t *to, scalar_t from[size], int real_size) {
      using vec_t = typename Info<scalar_t, vec_size>::vector_type;
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < size / vec_size; i++) {
        (vec_t *)to[i] = (vec_t [size / vec_size])from[i];
      }
    }
  };
};

template <typename func_t, typename ptr_array_t>
void dispatch(func_t f, ptr_array_t data_ptrs) {

}

}  // namespace policy


}}}} // namespace at::native::mem
