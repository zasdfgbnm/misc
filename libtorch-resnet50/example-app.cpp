#include <torch/script.h>

#include <pthread.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

struct inferStruct {
  at::Tensor output_;
  torch::jit::script::Module module_;
  std::vector<torch::jit::IValue> inputs_;
};

// Run model 5 times
void*
NForwardPass(void* run)
{
  // Model loaded in main thread
  torch::Device device_ = torch::Device(torch::kCUDA, 0);
  std::string model_path = "resnet50_libtorch.pt";
  torch::jit::script::Module module1 = torch::jit::load(model_path, device_);
  std::vector<torch::jit::IValue> inputs_(1);
  inputs_[0] = torch::zeros({1, 3, 224, 224}).to(device_);
  // pre-warm model
  at::Tensor output = module1.forward(inputs_).toTensor();

  inferStruct run1;
  pthread_t thread_id1;  //, thread_id2;

  run1.module_ = module1;
  run1.inputs_ = inputs_;

  auto run_tmp = &run1;
  run_tmp->output_ = run_tmp->module_.forward(run_tmp->inputs_).toTensor();

  for (int i = 0; i < 5; i++) {
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    run_tmp->output_ = run_tmp->module_.forward(run_tmp->inputs_).toTensor();
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << duration / 1000.0
              << " ms to run model in process thread (NForwardPass)"
              << std::endl;
  }
}

// Run model 1 time
void*
ForwardPass(void* run)
{
  // Model loaded in main thread
  torch::Device device_ = torch::Device(torch::kCUDA, 0);
  std::string model_path = "resnet50_libtorch.pt";
  torch::jit::script::Module module1 = torch::jit::load(model_path, device_);
  std::vector<torch::jit::IValue> inputs_(1);
  inputs_[0] = torch::zeros({1, 3, 224, 224}).to(device_);
  // pre-warm model
  at::Tensor output = module1.forward(inputs_).toTensor();

  inferStruct run1;
  pthread_t thread_id1;  //, thread_id2;

  run1.module_ = module1;
  run1.inputs_ = inputs_;

  auto run_tmp = &run1;
  run_tmp->output_ = run_tmp->module_.forward(run_tmp->inputs_).toTensor();

  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  run_tmp->output_ = run_tmp->module_.forward(run_tmp->inputs_).toTensor();
  cudaDeviceSynchronize();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << duration / 1000.0
            << " ms to run model in process thread (ForwardPass)" << std::endl;
}

int
main()
{
  // Model loaded in main thread
  torch::Device device_ = torch::Device(torch::kCUDA, 0);
  std::string model_path = "resnet50_libtorch.pt";
  torch::jit::script::Module module1 = torch::jit::load(model_path, device_);
  std::vector<torch::jit::IValue> inputs_(1);
  inputs_[0] = torch::zeros({1, 3, 224, 224}).to(device_);
  // pre-warm model
  at::Tensor output = module1.forward(inputs_).toTensor();

  inferStruct run1;
  pthread_t thread_id1;  //, thread_id2;

  run1.module_ = module1;
  run1.inputs_ = inputs_;
  run1.output_ = run1.module_.forward(run1.inputs_).toTensor();

  // Run model in main thread N times and report runtime each time
  for (int i = 0; i < 5; i++) {
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    output = module1.forward(inputs_).toTensor();
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << duration / 1000.0 << " ms to run model in main thread"
              << std::endl;
  }
  sleep(1);

  // Run N threads and in each run model once and report runtime each time
  std::cout << std::endl;
  for (int i = 0; i < 5; i++) {
    pthread_create(&thread_id1, NULL, ForwardPass, &run1);
    pthread_join(thread_id1, NULL);
  }
  sleep(1);

  // Run 1 thread and in it run the model N and report runtime each time
  std::cout << std::endl;
  pthread_create(&thread_id1, NULL, NForwardPass, &run1);
  pthread_join(thread_id1, NULL);

  return 0;
}