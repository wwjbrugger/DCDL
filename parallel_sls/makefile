PROJECT_NAME = Parallel_SLS

# Compiler +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
NVCC = nvcc
CC = g++

# Paths ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CUDAPATH = /usr/local/cuda
BUILD_DIR = bld
SOURCE_DIR = src
INCLUDE_DIR = inc
srcfiles := src/main.cpp

# FLAGS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CFLAGS = -std=c++11 -c -fopenmp -I$(CUDAPATH)/include -O0
NVCCFLAGS = -c -I$(CUDAPATH)/include
LFLAGS = -L$(CUDAPATH)/lib -L$(INCLUDE_DIR) -fopenmp
SHARED_L = -shared
SHARED_CC = -fpic
DEBUG = -g

# Targets ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
all: build clean shared clean

build: build_dir cpu
	$(CC) $(LFLAGS) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o

shared: cpu_shared
	$(CC) $(LFLAGS) $(SHARED_L) -o $(BUILD_DIR)/$(PROJECT_NAME)_shared *.o

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) $(DEBUG) $(SOURCE_DIR)/*.cu

cpu:
	$(CC) $(CFLAGS) $(DEBUG) $(SOURCE_DIR)/*.cpp

cpu_shared:
	$(CC) $(CFLAGS) $(SHARED_CC) $(DEBUG) $(SOURCE_DIR)/*.cpp

clean:
	rm *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)
