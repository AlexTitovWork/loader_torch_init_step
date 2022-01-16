// coded by Alex. 28.10.2021
// alexeytitovwork@gmail.com
// Torch to CUDA data transfer test.
#include <torch/script.h>
#include <torch/torch.h>

#include <memory>
#include <time.h>
//----------------------------
//async transfer
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#define DEFAULT_TORCH_SCRIPT ""
#define PATCH_WIDTH (400)
#define PATCH_HEIGHT (400)
//----------------------------
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/core.hpp"
#define LOG_FLAG false
#define TIMERS_FLAG true

using namespace std;

//-----------------------
/*
 * cmd line build:
 # For the Docker build

 sudo cmake -DCMAKE_PREFIX_PATH=../libtorch .
 sudo cmake --build . --config Release
 ./loader_torch_init_step test_data/style2.png

 # Time testing
time ./loader_torch_init_step ./test_data/structure2.png

  Processing result:
  
  root@3ec54f35ea02:/home/loader_torch_init_step# cmake --build . --config Release
  Scanning dependencies of target loader_torch_init_step
  [ 50%] Building CXX object CMakeFiles/loader_torch_init_step.dir/loader_torch_init_step.cpp.o
  [100%] Linking CXX executable loader_torch_init_step
  [100%] Built target loader_torch_init_step
  root@3ec54f35ea02:/home/loader_torch_init_step# time ./loader_torch_init_step ./test_data/structure2.png
  Pre-load, 
  Mem allocation and pining.   Time taken: 2.65s

  ROI set.                   Time taken: 0.00s
  CV loader 1.               Time taken: 0.04s
  Fit data in to memory. Time taken: 0.00s
  CPU - GPU transfer/reassign. Time taken: 0.01s

  Processing. Time taken: 0.09s
  ok!

  real	0m3.160s
  user	0m1.593s
  sys	0m1.609s

*/
//------------------------------------------------------------------------------


int main(int argc, const char *argv[]){
  clock_t tTotal = clock();
  clock_t tPreLoad = clock();
  //-----------------------------------------------------------------------------
  /**
   * @brief    Approach 2. Pre-allocate and pin memory before of trancfering.
   * Module::to(at::Device device, at::ScalarType dtype, bool non_blocking)
   * Module::to(at::ScalarType dtype, bool non_blocking)
   * Tested and worked.
   */
  int height =400;
  int width = 400;
  torch::Tensor tensor_image = torch::zeros({1,height,width,3});
  // torch::Tensor tensor_image = torch::rand({1,400,400,3});
  tensor_image = tensor_image.pin_memory();
  //-----------------------------------------------------------------------------
  /**
   * @brief    Approach 3. Pre-allocate and pin memory before of trancfering.
   * Module::to(at::Device device, at::ScalarType dtype, bool non_blocking)
   * Module::to(at::ScalarType dtype, bool non_blocking)
   * Tested and worked.
   */
  // int height =400;
  // int width = 400;
  // std::vector<int64_t> dims = { 1, height, width, 3 };
  // auto options = torch::TensorOptions().dtype(torch::kUInt8).device({ torch::kCUDA }).requires_grad(false);
  // torch::Tensor tensor_image = torch::zeros(dims, options);

  //--------------------------------------------------------------------------------

  printf("Pre-load, \nMem allocation and pining.   Time taken: %.2fs\n\n", (double)(clock() - tPreLoad) / CLOCKS_PER_SEC);

  if (LOG_FLAG){
    printf("OpenCV: %s", cv::getBuildInformation().c_str());
  }
  
  if (LOG_FLAG){

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()){
      std::cout << "CUDA is available! Training on GPU." << std::endl;
      device = torch::kCUDA;
    }
    std::cout << "PyTorch version: "
              << TORCH_VERSION_MAJOR << "."
              << TORCH_VERSION_MINOR << "."
              << TORCH_VERSION_PATCH << std::endl;
  }

  clock_t tROIset = clock();

  cv::Rect myROI(30, 10, 400, 400);
  printf("ROI set.                   Time taken: %.2fs\n", (double)(clock() - tROIset) / CLOCKS_PER_SEC);
 
  clock_t tImgOneConversion = clock();
  clock_t tLoadOpenCV = clock();

  cv::Mat img = cv::imread(argv[1]); // 600x900
  // cv::Mat croppedImage = cv::imread(argv[1]); // 600x900

  if (TIMERS_FLAG){
    printf("CV loader 1.               Time taken: %.2fs\n", (double)(clock() - tLoadOpenCV) / CLOCKS_PER_SEC);
    // std::cout<<"tensor_image_style2 Loaded  and converted to Tensor. OK."<<std::endl;
  }

  cv::Mat croppedImage = img(myROI);
  cv::Mat input; 
  cv::cvtColor(croppedImage, input, cv::COLOR_BGR2RGB);
  //-------------------------------------------------------------------------
  /**
     * @brief Approach 1. Base approach, blob and data transfering here, witout preallocation 
     * All time spended here. There are no preallocation and warming GPU. 
     */
    // torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte);
    // tensor_image = tensor_image.permute({0, 3, 1, 2});
    // tensor_image = tensor_image.toType(torch::kFloat);
    // tensor_image = tensor_image.div(255);
    // tensor_image = tensor_image.to(torch::kCUDA);
  //-------------------------------------------------------------------------

  clock_t tFitData = clock();
  {
    // bool non_blocking = true;
    // torch::NoGradGuard no_grad;
  //-------------------------------------------------------------------------
  /**
   * @brief    Approach 2. Pre-allocate and pin memory before of trancfering.
   * Module::to(at::Device device, at::ScalarType dtype, bool non_blocking)
   * Module::to(at::ScalarType dtype, bool non_blocking)
   * Tested and worked.
   * 
   * Pin CPU memory, various form
   * torch::Tensor tensor_pinned = at::empty(gpu.sizes(), device(at::kCPU).pinned_memory(true));
   * torch::Tensor tensor_image = at::empty(gpu.sizes(), device(at::kCPU).pinned_memory(true));
   * torch::Tensor tensor_pinned = torch::empty(tensor_image.sizes(), device(torch::kCPU).pin_memory(true));
   * torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte );
   * torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte).pin_memory(torch::kCPU);
   */
  
  tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte );
  tensor_image = tensor_image.permute({0, 3, 1, 2});
  tensor_image = tensor_image.toType(torch::kFloat);
  tensor_image = tensor_image.div(255);

    if (TIMERS_FLAG){
      printf("Fit data in to memory. Time taken: %.2fs\n", (double)(clock() - tFitData) / CLOCKS_PER_SEC);
      // std::cout<<"tensor_image_style2 Loaded  and converted to Tensor. OK."<<std::endl;
    }

  //-------------------------------------------------------------------------
      clock_t tTransferData = clock();
  /**
     * @brief Approach 3. CUDA Asynch approach, data transfering here in prepinned and preallocated memory.
     * Data transfer directly cuda copy, without using BLOB data
     * Worked but with similar perfomance as BLOB Approach 3.
     *  
     * Manual for CUDA asynch copy
     * cudaMemcpy2DAsync(
     * dst: *mut c_void, 
     * dpitch: usize, 
     * src: *const c_void, 
     * spitch: usize, 
     * width: usize, 
     * height: usize, 
     * kind: cudaMemcpyKind, 
     * stream: *mut CUstream_st)
     */
     
    // 
    // auto data = tensor_image.data_ptr<uint8_t>() ;
    // auto pitch = width * sizeof(uint8_t) * 3;
    // // uint8_t* data = new uint8_t[PATCH_HEIGHT * PATCH_WIDTH * 3];
    // auto stream = at::cuda::getStreamFromPool(true, 0);
    // cudaMemcpy2DAsync(data,
    //     pitch,
    //     input.data,
    //     pitch,
    //     pitch,
    //     height,
    //     cudaMemcpyHostToDevice,
    //     stream.stream());

  	// cudaStreamSynchronize(stream.stream());

    // tensor_image = tensor_image.permute({0, 3, 1, 2});
    // tensor_image = tensor_image.toType(torch::kFloat);
    // tensor_image = tensor_image.div(255);
    //-------------------------------------------------------------------------
    /**
     * @brief 
     * void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking)
     */
    tensor_image = tensor_image.to(torch::kCUDA);
    // tensor_image = tensor_image.to(torch::kCUDA, torch::kFloat, non_blocking);
    //-------------------------------------------------------------------------
    // Check Tensor in CUDA memory
    // std::cout<<tensor_image<<std::endl;
    if (TIMERS_FLAG){
      printf("CPU - GPU transfer/reassign. Time taken: %.2fs\n\n", (double)(clock() - tTransferData) / CLOCKS_PER_SEC);
    }
  }

  if (LOG_FLAG)
    std::cout << "Loaded ok." << std::endl;

  if (TIMERS_FLAG){
    printf("Processing. Time taken: %.2fs\n", (double)(clock() - tImgOneConversion) / CLOCKS_PER_SEC);
  }
  std::cout << "ok!\n";
}
