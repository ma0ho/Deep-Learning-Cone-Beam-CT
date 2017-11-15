TF_INC = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_LIBS = /usr/local/cuda/lib64
CC = g++
CUCC = nvcc
CFLAGS = -std=c++11 -Wall -shared -O2 -fPIC
# -D_FORCE_INLINES is a workaround for Ubuntu 16.04
# see https://github.com/BVLC/caffe/issues/4046
CUFLAGS = -std=c++11 -O2 -D_FORCE_INLINES -ccbin=gcc-4.9
SRC = $(wildcard *.cc)
OUT = $(SRC:%.cc=%.so)

backproject.so: backproject.cu.o

%.cu.o: %.cu
	$(CUCC) $(CUFLAGS) -c -o $@ $< \
	    -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

%.so: %.cc %.cu.o
	$(CC) $(CFLAGS) $^ -o $@ $1 -I $(TF_INC) -L $(CUDA_LIBS) -lcudart

%.so: %.cc
	$(CC) $(CFLAGS) $< -o $@ -I $(TF_INC)

all: $(OUT)

.PHONY: clean
clean:
	rm *.so *.o
