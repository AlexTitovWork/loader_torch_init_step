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

cmake -DCMAKE_PREFIX_PATH=../libtorch .
cmake --build . --config Release
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
__global__
void kernel(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}


int main(int argc, const char *argv[]){

    int N = 2048*2;
    float *host_a, *device_a;        // Define host-specific and device-specific arrays.
    int size = sizeof(float) * N;

    clock_t tTotal = clock();

    //----------------------------------------------------------------------------------
    clock_t tAlloc = clock();

    cudaMalloc(&device_a, size);   // `device_a` is immediately available on the GPU.
    cudaMallocHost(&host_a, size); // `host_a` is immediately available on CPU, and is page-locked, or pinned.
    // Create a stream for this segment's worth of copy and work.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t threadsPerBlock;
    size_t numberOfBlocks;
    size_t numberOfSMs;

    // numberOfSMs = 1;
    threadsPerBlock = 256;
    numberOfBlocks = N /256;

    printf("Alloc. Time taken: %.2fs\n\n", (double)(clock() - tAlloc) / CLOCKS_PER_SEC);
    //----------------------------------------------------------------------------------
    clock_t tInit = clock();

    for(int i = 0 ; i<N; i++)
      host_a[i] = i*1.2;

    printf("Init. Time taken: %.2fs\n\n", (double)(clock() - tInit) / CLOCKS_PER_SEC);
    //----------------------------------------------------------------------------------
    clock_t tTrans = clock();

    // `cudaMemcpy` takes the destination, source, size, and a CUDA-provided variable for the direction of the copy.
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    printf("tTransfer. Time taken: %.2fs\n\n", (double)(clock() - tTrans) / CLOCKS_PER_SEC);

    kernel<<<numberOfBlocks, threadsPerBlock, 0, stream>>>(33.0, device_a, N);
    
    clock_t tBTrans = clock();
    // `cudaMemcpy` can also copy data from device to host.
    cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);


    printf("tBack-Trans. Time taken: %.2fs\n\n", (double)(clock() - tBTrans) / CLOCKS_PER_SEC);

    // for(int i = 0 ; i<N; i++)
    //   std::cout<< host_a[i] << std::endl;
    std::cout<< host_a[1] << std::endl;
    std::cout<< host_a[N] << std::endl;


    cudaFree(device_a);
    cudaFreeHost(host_a);          // Free pinned memory like this.
  
    printf("tTotal. Time taken: %.2fs\n\n", (double)(clock() - tTotal) / CLOCKS_PER_SEC);

  std::cout << "ok!\n";
}
