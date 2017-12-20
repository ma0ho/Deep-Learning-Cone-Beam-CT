include_directories(SYSTEM ${TF_INCLUDE} "${TF_INCLUDE}/external/nsync/public")
link_directories(${TF_LIB})

list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-DGOOGLE_CUDA=1;-Xcompiler")

# add backproject ops
cuda_add_library(backproject backproject.cu backproject.cc SHARED)
target_link_libraries(backproject tensorflow_framework)

# add dennerlein writer
add_library(write_dennerlein SHARED write_dennerlein.cc)
target_link_libraries(write_dennerlein tensorflow_framework)

install(TARGETS backproject write_dennerlein
   LIBRARY DESTINATION lib)

