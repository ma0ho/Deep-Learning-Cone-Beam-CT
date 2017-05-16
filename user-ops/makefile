TF_INC = $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CC = g++
CFLAGS = -std=c++11 -Wall -shared -O2 -fPIC
SRC = $(wildcard *.cc)
OUT = $(SRC:%.cc=%.so)

%.so: %.cc
	$(CC) $(CFLAGS) $< -o $@ -I $(TF_INC)

all: $(OUT)

.PHONY: clean
clean:
	rm *.so
