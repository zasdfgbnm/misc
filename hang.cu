#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA(op)                                  \
  {                                                     \
    auto code = (op);                                   \
    if (code != cudaSuccess) {                          \
      throw std::runtime_error(cudaGetErrorName(code)); \
    }                                                   \
  }


const int N = 1000;

// auto FLAG = cudaEventDisableTiming | cudaEventInterprocess;
auto FLAG = cudaEventDisableTiming;

void code1() {
  CHECK_CUDA(cudaSetDevice(0));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int i = 0; i < N; i++) {
    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreateWithFlags(&event, FLAG));
    CHECK_CUDA(cudaEventRecord(event, stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream, event, 0));
    CHECK_CUDA(cudaEventDestroy(event));
  }
  CHECK_CUDA(cudaStreamDestroy(stream));
}

void code2() {
  CHECK_CUDA(cudaSetDevice(0));
  for (int i = 0; i < N; i++) {
    cudaEvent_t myEvent;
    CHECK_CUDA(cudaEventCreateWithFlags(&myEvent, FLAG));
    CHECK_CUDA(cudaEventDestroy(myEvent));
  }
}

int main() {
  std::thread thread1(code1);
  std::thread thread2(code2);
  thread1.join();
  thread2.join();
}
