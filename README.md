# loader_torch_init_step<br>
Tuning init step of torch. Tuning image loader in to torch tensor format<br>
<br>
//-----------------------<br>
/*<br>
 * cmd line build:<br>

 download libtorch lib with CUDA support in to ./home directory:<br>
  wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.10.1%2Bcu102.zip<br>
 
 download project in to ./home directory:<br>
  git clone https://gitlab.com/pingpong_alex_optimization/swapping_optimized<br>

 # For local side

 sudo cmake -DCMAKE_PREFIX_PATH=../libtorch .<br>
 sudo cmake --build . --config Release<br>
 ./loader_torch_init_step test_data/style2.png<br>

 # For the Docker build<br>

 cmake -DCMAKE_PREFIX_PATH=../libtorch .<br>
 cmake --build . --config Release<br>
 ./loader_torch_init_step test_data/style2.png<br>


1. For test Libtorch loader use "loader_torch_init_step.cpp" and rename "CMakeLists_torch_cv_cuda.txt" in to 
"CMakeLists.txt" repeat cmake and build instructions.

2. For test pure CUDA loader use "loader_direct_cuda.cu" and rename "CMakeLists_cuda.txt" in to 
"CMakeLists.txt" repeat cmake and build instructions.

 # Libtorch image loader - time test<br>
time ./loader_torch_init_step ./test_data/structure2.png<br>

  Processing result after libtorch test:<br>

  root@3ec54f35ea02:/home/loader_torch_init_step# cmake --build . --config Release <br>
  Scanning dependencies of target loader_torch_init_step<br>
  [ 50%] Building CXX object CMakeFiles/loader_torch_init_step.dir/loader_torch_init_step.cpp.o<br>
  [100%] Linking CXX executable loader_torch_init_step<br>
  [100%] Built target loader_torch_init_step<br>
  root@3ec54f35ea02:/home/loader_torch_init_step# time ./loader_torch_init_step ./test_data/structure2.png<br>
  Pre-load, <br>
  Mem allocation and pining.   Time taken: 2.65s<br>

  ROI set.                   Time taken: 0.00s<br>
  CV loader 1.               Time taken: 0.04s<br>
  Fit data in to memory. Time taken: 0.00s<br>
  CPU - GPU transfer/reassign. Time taken: 0.01s<br>
<br>
  Processing. Time taken: 0.09s<br>
  ok!<br>

  real	0m3.160s<br>
  user	0m1.593s<br>
  sys	0m1.609s<br>

Sun Jan 16 14:58:54 2022       <br>
+-----------------------------------------------------------------------------+<br>
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |<br>
|-------------------------------+----------------------+----------------------+<br>
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |<br>
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |<br>
|                               |                      |               MIG M. |<br>
|===============================+======================+======================|<br>
|   0  GeForce RTX 208...  Off  | 00000000:01:00.0 Off |                  N/A |<br>
| 25%   42C    P2    53W / 257W |    988MiB / 11016MiB |     10%      Default |<br>
|                               |                      |                  N/A |<br>
+-------------------------------+----------------------+----------------------+<br>
|   1  GeForce RTX 208...  Off  | 00000000:02:00.0 Off |                  N/A |<br>
|  7%   44C    P0    62W / 250W |      3MiB / 11019MiB |      1%      Default |<br>
|                               |                      |                  N/A |<br>
+-------------------------------+----------------------+----------------------+<br>
                                                                               <br>
+-----------------------------------------------------------------------------+<br>
| Processes:                                                                  |<br>
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |<br>
|        ID   ID                                                   Usage      |<br>
|=============================================================================|<br>
|    0   N/A  N/A    944573      C   ./loader_torch_init_step          985MiB |<br>
+-----------------------------------------------------------------------------+<br>
<br>
*/<br>
//------------------------------------------------------------------------------<br>
