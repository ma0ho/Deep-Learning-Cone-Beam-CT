cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(userops)

enable_language(CXX)

find_package(CUDA QUIET REQUIRED)

# locate tf
execute_process(
   COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
   OUTPUT_VARIABLE TF_INCLUDE
)
execute_process(
   COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
   OUTPUT_VARIABLE TF_LIB
)

include_directories(SYSTEM
   ${TF_INC}
   "${TF_INC}/external/nsync/public"
)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-DGOOGLE_CUDA=1;-fPIC;-Xcompiler")

set (CMAKE_CXX_STANDARD 11)

cuda_add_library(
   backproject
   backproject.cu backproject.cc
)

target_link_libraries(backproject ${TF_LIB})

