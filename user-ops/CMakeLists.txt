include_directories(SYSTEM ${TF_INCLUDE} "${TF_INCLUDE}/external/nsync/public")

list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-DGOOGLE_CUDA=1;-Xcompiler")

# add backproject ops
cuda_add_library(backproject backproject.cu backproject.cc SHARED)

# add dennerlein writer
add_library(write_dennerlein SHARED write_dennerlein.cc)

install(TARGETS backproject write_dennerlein
   LIBRARY DESTINATION lib)

