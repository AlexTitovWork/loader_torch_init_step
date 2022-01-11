cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
SET(CUDA_HOME /usr/local/cuda)
SET(CUDA_LAUNCH_BLOCKING 1)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
# SET(CUDA_CUBLAS_LIBRARIES /usr/local/cuda/lib64)
SET(CUDA_cublas_device_LIBRARY /usr/local/cuda/lib64)
# SET(CMAKE_CUDA_COMPILER_ENV_VAR /usr/local/cuda/bin/nvcc)
SET(CUDA_INCLUDE_DIRS /usr/local/cuda/include)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
# set(OpenCV_DIR /usr/local/lib)


project(cmake_and_cuda LANGUAGES CXX CUDA)
 
include(CTest)

find_package(CUDA REQUIRED)

# libtorch with OpenCV support , with fixed Target BUG !!!!! 
list(APPEND CMAKE_PREFIX_PATH ./libtorch)
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")

# set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
# list(APPEND CMAKE_PREFIX_PATH  ./opencv4)

find_package(OpenCV REQUIRED)
# include_directories( ${OpenCV_INCLUDE_DIRS} )
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

  
 
add_executable(loader_torch_init_step loader_torch_init_step.cpp)
 
set_property(TARGET loader_torch_init_step 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(loader_torch_init_step PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(particle_cuda_test PRIVATE particles_new_lib)

# LINK LIBRARIES

# target_link_libraries(particle_cuda_test PRIVATE "${OpenCV_LIBS}"  )

# target_link_libraries(particle_cuda_test PRIVATE "${TORCH_LIBRARIES}")

# target_link_libraries(particle_cuda_test PRIVATE particles_new_lib) # Duplicate link, use if not use ${TORCH_LIBRARIES} ${OpenCV_LIBS} libs common.
set_property(TARGET loader_torch_init_step PROPERTY CXX_STANDARD 14)