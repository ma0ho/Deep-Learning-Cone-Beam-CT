# DeepImaging

CUDA-enabled CT reconstruction using Tensorflow.

## Dependencies
* CUDA (tested on versions 8 and 9)
* Tensorflow >= 1.2
* CMake >= 3.1
* GCC (tested on 4.9 and 7.2)
* Python >= 3.5

## How to build
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=[...] ..
make && make install
```
