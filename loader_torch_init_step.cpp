// coded by Alex. 28.10.2021
// alexeytitovwork@gmail.com
// Torch to CUDA data transfer test.
#include <torch/script.h>
#include <torch/torch.h>

#include <memory>
#include <time.h>

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
*/

int main(int argc, const char *argv[])
{

  clock_t tTotal = clock();
  clock_t tPreLoad = clock();


  // torch::Tensor tensor_image = torch::rand({1,400,400,3});
  // tensor_image = tensor_image.pin_memory();

  printf("Pre-load, \nMem allocation and pining.   Time taken: %.2fs\n\n", (double)(clock() - tPreLoad) / CLOCKS_PER_SEC);


  if (LOG_FLAG)
  {
    printf("OpenCV: %s", cv::getBuildInformation().c_str());
  }
  

  // TODO INIT device and memory
  if (LOG_FLAG)
  {

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available())
    {
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

  if (TIMERS_FLAG)
  {
    printf("CV loader 1.               Time taken: %.2fs\n", (double)(clock() - tLoadOpenCV) / CLOCKS_PER_SEC);
    // std::cout<<"tensor_image_style2 Loaded  and converted to Tensor. OK."<<std::endl;
  }

  cv::Mat croppedImage = img(myROI);
  cv::Mat input; 
  cv::cvtColor(croppedImage, input, cv::COLOR_BGR2RGB);


  // torch::Tensor tensor_pinned = at::empty(gpu.sizes(), device(at::kCPU).pinned_memory(true));
  // torch::Tensor tensor_image = at::empty(gpu.sizes(), device(at::kCPU).pinned_memory(true));
  // torch::Tensor tensor_pinned = torch::empty(tensor_image.sizes(), device(torch::kCPU).pin_memory(true));
  clock_t tFitData = clock();
  bool non_blocking = true;


  torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte );
  // tensor_image = tensor_image.pin_memory();
  // tensor_image = tensor_image.pin_memory(torch::kCUDA);

  if (TIMERS_FLAG)
  {
    printf("Fit data in to memory. Time taken: %.2fs\n", (double)(clock() - tFitData) / CLOCKS_PER_SEC);
    // std::cout<<"tensor_image_style2 Loaded  and converted to Tensor. OK."<<std::endl;
  }

  // torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte).pin_memory(torch::kCPU);

  tensor_image = tensor_image.permute({0, 3, 1, 2});
  tensor_image = tensor_image.toType(torch::kFloat);
  tensor_image = tensor_image.div(255);

  clock_t tTransferData = clock();
  //void Module::to(at::Device device, at::ScalarType dtype, bool non_blocking)
  tensor_image = tensor_image.to(torch::kCUDA,torch::kFloat, non_blocking);

  // tensor_image = tensor_image.to(torch::kCUDA, 1);
  // tensor_image = tensor_image.cuda();
  
//TODO  
//   //test low level code
// cudaErr = cudaMemcpyAsync(data,
// 		pitch,
// 		inputData,
// 		pitch,
// 		pitch,
// 		height,
// 		cudaMemcpyHostToDevice,
// 		stream.stream());

// 	if (cudaErr != cudaSuccess)
// 	{
// 		std::cerr << "Error copying data" << std::endl;
// 		return false;
// 	}
// 	cudaStreamSynchronize(stream.stream());


  //end test 


  if (TIMERS_FLAG)
  {
    printf("CPU - GPU transfer/reassign. Time taken: %.2fs\n\n", (double)(clock() - tTransferData) / CLOCKS_PER_SEC);
  }


  if (LOG_FLAG)
    std::cout << "Loaded ok." << std::endl;

  if (TIMERS_FLAG)
  {
    printf("Processing. Time taken: %.2fs\n", (double)(clock() - tImgOneConversion) / CLOCKS_PER_SEC);
  }

  
  std::cout << "ok!\n";
}
