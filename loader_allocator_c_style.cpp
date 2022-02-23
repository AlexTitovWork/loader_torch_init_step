/**
 * @file loader_torch_init_step.cpp
 * @author Alex Titov (alexeytitovwork@gmail.com)
 * @brief   Perfomance test. 
 * Image lodaing with Open CV 4.5.4 in to Torch Tensor.
 * Torch tensor to CUDA memory data transfer.
 * @version 0.1
 * @date 2022-02-03
 * @copyright Copyright (c) 2022
 */
#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <time.h>
//----------------------------
// For *.cu files
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda_runtime.h>

//async transfer
// #include <c10/cuda/CUDAStream.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <cuda_runtime_api.h>
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

cmake -DCMAKE_PREFIX_PATH=../libtorch .
cmake --build . --config Release
./loader_torch_init_step test_data/style2.png


// ISSUE ld -1 --------------------------------------------------------
interceptor@interceptor-N750JK:~/Документы/Git_Medium_repo/Binary_search_engine_CUDA/loader_torch_init_step$ sudo cmake --build . --config Release 
-- TORCH_LIBRARIES = torch;torch_library;/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/libtorch/lib/libc10.so;/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/libtorch/lib/libkineto.a
-- OpenCV_LIBS = opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_gapi;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_stitching;opencv_video;opencv_videoio
-- Configuring done
-- Generating done
-- Build files have been written to: /home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/loader_torch_init_step
Consolidate compiler generated dependencies of target loader_torch_init_step
[ 50%] Building CXX object CMakeFiles/loader_torch_init_step.dir/loader_torch_init_step.cpp.o
[100%] Linking CXX executable loader_torch_init_step
CMakeFiles/loader_torch_init_step.dir/loader_torch_init_step.cpp.o: In function `main':
loader_torch_init_step.cpp:(.text+0x731): undefined reference to `cv::imread(std::string const&, int)'
collect2: error: ld returned 1 exit status
CMakeFiles/loader_torch_init_step.dir/build.make:115: recipe for target 'loader_torch_init_step' failed
make[2]: *** [loader_torch_init_step] Error 1
CMakeFiles/Makefile2:838: recipe for target 'CMakeFiles/loader_torch_init_step.dir/all' failed
make[1]: *** [CMakeFiles/loader_torch_init_step.dir/all] Error 2
Makefile:100: recipe for target 'all' failed
make: *** [all] Error 2

SOLVED 
1. reinstall libtorch from other link !
2. sometimes after delete all cache and cmake temp files use:
sudo cmake -DCMAKE_PREFIX_PATH=/usr/local/lib . 
# in local installation of Open CV 
sudo cmake -DCMAKE_PREFIX_PATH=/home/alex/local/lib . 

//---------------------------------------------------------------------

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

Sun Jan 16 14:58:54 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:01:00.0 Off |                  N/A |
| 25%   42C    P2    53W / 257W |    988MiB / 11016MiB |     10%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 208...  Off  | 00000000:02:00.0 Off |                  N/A |
|  7%   44C    P0    62W / 250W |      3MiB / 11019MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    944573      C   ./loader_torch_init_step          985MiB |
+-----------------------------------------------------------------------------+
*/
//------------------------------------------------------------------------------
// std::array<int64_t,4> tensor_dim = {4, 32, 32, 32};

  int64_t tensor_dim = 1234567;

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

  torch::globalContext().setUserEnabledCuDNN(false);
  std::cout <<"torch::cuda::cudnn_is_available() " << torch::cuda::cudnn_is_available() << std::endl;
  std::cout << "Tensor creation comparison" << std::endl;
  std::cout << cudnnCnnInferVersionCheck()<< std::endl;

  // Ligt pre inittвывсывсываеttt
  // torch::Tensor init_mem_tensor = torch::zeros({1,1,1,3});
  // init_mem_tensor = init_mem_tensor.pin_memory();

  torch::manual_seed(0);
  torch::NoGradGuard guard;
  
  std::array<int64_t,4> tensor_str = {32768, 1024, 32, 1}; // NCHW format
  cudnnDataType_t data_type        = CUDNN_DATA_FLOAT;
  int64_t alignment                = sizeof(float);
  int64_t id                       = 0xD0D0CACA; // Some magic number
// std::array<int64_t,4> tensor_dim = {4, 32, 32, 32};
  // int64_t tensor_dim = 1234567;
  
  printf("tensor_dim %p \n",&tensor_dim);


  // int64_t *p=(int64_t  *)0x5612312e5020;
  // printf("%ld \n",*tensor_dim);
  // printf("%p \n", tensor_dim);

//------------------------------------------



  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()){
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
    // torch::Backend::MkldnnCPU = false;

    if (LOG_FLAG){
      std::cout << "PyTorch version: "
                << TORCH_VERSION_MAJOR << "."
                << TORCH_VERSION_MINOR << "."
                << TORCH_VERSION_PATCH << std::endl;
    }

    printf("Device select     Time taken: %.2fs\n", (double)(clock() - tPreLoad) / CLOCKS_PER_SEC);

    //-------------------------------------------------------
    int height =400;
    int width = 400;
    // // torch::cuda::synchronize(-1);
    /**
     * @brief directly CUDA memory allocation using device tag
     * 
     */
    // torch::Tensor tensor_image = torch::zeros({1,height,width,3}, device);
    /**
     * @brief Allocation into HOST and pin it. Than redefine this memory for GPU
     * 
     */
    // torch::Tensor tensor_image = torch::rand({1,400,400,3});
    clock_t tAllocTensor = clock();

    torch::Tensor tensor_image = torch::zeros({1,height,width,3});


    printf("Tensor zeros alloc on CPU   Time taken: %.2fs\n\n", (double)(clock() - tAllocTensor) / CLOCKS_PER_SEC);
    clock_t tPin = clock();
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

    printf("Mem pining.   Time taken: %.2fs\n\n", (double)(clock() - tPin) / CLOCKS_PER_SEC);

    if (LOG_FLAG){
      printf("OpenCV: %s", cv::getBuildInformation().c_str());
    }
    
    clock_t tROIset = clock();
    cv::Rect myROI(30, 10, 400, 400);
    printf("ROI set.                   Time taken: %.2fs\n", (double)(clock() - tROIset) / CLOCKS_PER_SEC);
  
    clock_t tImgOneConversion = clock();
    clock_t tLoadOpenCV = clock();

    cv::Mat img = cv::imread(argv[1]); // 600x900
    // cv::Mat croppedImage = cv::imread(argv[1]); // 600x900

    if (TIMERS_FLAG){
      printf("CV loader 1.          Time taken: %.2fs\n", (double)(clock() - tLoadOpenCV) / CLOCKS_PER_SEC);
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
    
    // torch::cuda::synchronize(-1);
    tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte );
    // torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte );

    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);

      if (TIMERS_FLAG){
        printf("Fit data in to memory. Time taken: %.2fs\n", (double)(clock() - tFitData) / CLOCKS_PER_SEC);
        // std::cout<<"tensor_image_style2 Loaded  and converted to Tensor. OK."<<std::endl;
      }

    //-------------------------------------------------------------------------
    clock_t tTransferData = clock();
    // torch::cuda::synchronize();

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
      // torch::cuda::synchronize(-1);
      // tensor_image = tensor_image.to(torch::kCUDA);
      tensor_image = tensor_image.to(device);


      height = tensor_image.size(0);
      width = tensor_image.size(1);
      int depth = tensor_image.size(2);
      
      std::cout<<   "tensor_image Tensor size:"<<std::endl;
      std::cout<<   height << "x"<< width <<"x"<<depth <<std::endl;
      // tensor_image = tensor_image.to(torch::kCUDA, torch::kFloat, non_blocking);
      //-------------------------------------------------------------------------
      // Check Tensor in CUDA memory

      // torch::cuda::synchronize();

      if (TIMERS_FLAG){

        printf("CPU - GPU transfer/reassign. Time taken: %.2fs\n\n", (double)(clock() - tTransferData) / CLOCKS_PER_SEC);
      }
      
      // std::cout<<tensor_image<<std::endl;


    }

    if (LOG_FLAG)
      std::cout << "Loaded ok." << std::endl;

    if (TIMERS_FLAG){
      printf("Processing. Time taken: %.2fs\n", (double)(clock() - tImgOneConversion) / CLOCKS_PER_SEC);
    }


    // //---------------
    system("pause");
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    std::string dummy;
    std::cout << "Press any key to continue . . .";
    std::getline(std::cin, dummy);
    // //---------------
    
    std::cout << "ok!\n";
  }
}
