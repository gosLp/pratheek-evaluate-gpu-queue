# Makefile for MS Queue Test

CXX = g++
CXXFLAGS = -std=c++14 -Wall -O3
TARGET = ms_queue_test
SOURCE = main.cpp

# OpenCL library flags - adjust based on your system
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    # Linux
    OPENCL_LIBS = -lOpenCL
    OPENCL_INCLUDE = 
endif
ifeq ($(UNAME_S),Darwin)
    # macOS
    OPENCL_LIBS = -framework OpenCL
    OPENCL_INCLUDE = 
endif

# Windows (if using MinGW)
ifeq ($(OS),Windows_NT)
    OPENCL_LIBS = -lOpenCL
    OPENCL_INCLUDE = -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/include"
endif

LIBS = $(OPENCL_LIBS)
INCLUDES = $(OPENCL_INCLUDE)

.PHONY: all clean setup

all: setup $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

setup:
	@mkdir -p kernels
	@echo "Make sure your .cl and .h files are in the kernels/ directory"

clean:
	rm -f $(TARGET)

install-deps-ubuntu:
	sudo apt-get update
	sudo apt-get install build-essential opencl-headers ocl-icd-opencl-dev

install-deps-fedora:
	sudo dnf install gcc-c++ opencl-headers ocl-icd-devel

help:
	@echo "Available targets:"
	@echo "  all          - Build the executable"
	@echo "  clean        - Remove built files"
	@echo "  setup        - Create necessary directories"
	@echo "  install-deps-ubuntu - Install OpenCL dependencies on Ubuntu/Debian"
	@echo "  install-deps-fedora - Install OpenCL dependencies on Fedora/RHEL"