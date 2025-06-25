#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "kernels/lcrqueue32_host.h"

#define __CL_ENABLE_EXTENSIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

int main (int argc, char **argv) {
    cl_int err;
    
    // Read the file queue_dispatch source cl kernel
    std::ifstream srcFile("kernels/queue_dispatch.cl");
    if (!srcFile) {
        std::cerr << "Error: no file queue_dispatch.cl found!" << std::endl;
        return 1;
    }
    std::string src(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));


    // Get platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);

    // Find GPU device
    cl_device_id gpu_device = NULL;
    for (cl_platform_id platform : platforms) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);
            gpu_device = devices[0];
            break;
        }
    }

    if (gpu_device == NULL) {
        std::cerr << "No GPU devices found!" << std::endl;
        return 1;
    }

    // Get device name
    char device_name[256];
    clGetDeviceInfo(gpu_device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::cout << "Using GPU device: " << device_name << std::endl;

    // Create context
    cl_context context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context!" << std::endl;
        return 1;
    }

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, gpu_device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue!" << std::endl;
        return 1;
    }

    // Build program
    const char* src_ptr = src.c_str();
    size_t src_size = src.length();
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_size, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program!" << std::endl;
        return 1;
    }

    // std::string buildOpts = "-I./kernels -DMY_QUEUE_LENGTH=4096 -DGROUPS=256 -DWORK=100";
    std::string buildOpts = "-I./kernels -DMY_QUEUE_LENGTH=4096 -DMY_QUEUE_FACTOR=12 -DGROUPS=256 -DWORK=100";

    // Auto-detect which queue is being used by reading queue_dispatch.cl
    std::ifstream dispatchFile("kernels/queue_dispatch.cl");
    std::string dispatchContent(std::istreambuf_iterator<char>(dispatchFile), (std::istreambuf_iterator<char>()));
    
    std::string queueType = "UNKNOWN";
    size_t queueSize = 0;  // Fixed: was missing declaration
    std::vector<uint32_t> queueInitData;


    // Test parameters
    const int num_elements = 1000;
    const int queuelen = 4096;
    
    const size_t barrier_size = 1000 * sizeof(uint32_t);
    const size_t array_size = sizeof(int) * num_elements;
    
    if (dispatchContent.find("#include \"queue_ms.cl\"") != std::string::npos && 
        dispatchContent.find("// #include \"queue_ms.cl\"") == std::string::npos) {
        buildOpts += " -DUSE_MS_QUEUE";
        queueType = "MS";
        queueSize = queuelen * 3 * sizeof(uint32_t);  // MS queue sizing
        queueInitData.resize(queuelen * 3, 0);
        queueInitData[0] = 65536; // head
        queueInitData[1] = 65536; // tail
    }
    else if (dispatchContent.find("#include \"queue_sfq.cl\"") != std::string::npos && 
             dispatchContent.find("// #include \"queue_sfq.cl\"") == std::string::npos) {
        buildOpts += " -DUSE_SFQ_QUEUE";
        queueType = "SFQ";
        queueSize = (queuelen * 3) * sizeof(uint32_t);  // SFQ queue sizing
        queueInitData.resize(queuelen * 3, 0);
        // SFQ specific initialization if needed
    }
    else if (dispatchContent.find("#include \"queue_tz.cl\"") != std::string::npos && 
             dispatchContent.find("// #include \"queue_tz.cl\"") == std::string::npos) {
        buildOpts += " -DUSE_TZ_QUEUE";
        queueType = "TZ";
        queueSize = (queuelen + 5) * sizeof(uint32_t);  // TZ queue sizing
        queueInitData.resize(queuelen + 5, 0);
        queueInitData[0] = 0;  // head
        queueInitData[1] = 1;  // tail
        queueInitData[2] = 4294967295;  // vnull to null_1
        // Initialize rest to null_0
        for (int i = 3; i < queuelen + 5; i++) {
            queueInitData[i] = 4294967294;  // null_0
        }
        queueInitData[3] = 4294967295;  // first to null_1
    }
    else if (dispatchContent.find("#include \"queue_lcrq32.cl\"") != std::string::npos && 
             dispatchContent.find("// #include \"queue_lcrq32.cl\"") == std::string::npos) {
        buildOpts += " -DUSE_LCRQ_QUEUE ";
        queueType = "LCRQ";
        // For the reduced test size:
        // CRQ_LEN = 256, NUM_BASE_CRQS = 4
        // const size_t crq32_size = 64 + (256 * 16) + (1500 * 4) + 4; // ~10KB per crq32
        // const size_t lcrq32_size = 64 + (4 * crq32_size); // ~40KB total
        // queueSize = queuelen * 10 * sizeof(uint32_t);  // LCRQ queue sizing (estimate)
        queueSize = sizeof(lcrq32_host); 
        // queueInitData.resize(queuelen * 10, 0);
        // LCRQ specific initialization if needed
    }
    
    std::cout << "Detected queue type: " << queueType << std::endl;

    // Get device vendor for platform-specific options
    char vendor[256];
    clGetDeviceInfo(gpu_device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    std::string vendor_str(vendor);

    if (vendor_str.find("AMD") != std::string::npos) {
        buildOpts += " -DAMD -DWARP=64 -DFAILSAFE=100000";
    } else if (vendor_str.find("NVIDIA") != std::string::npos) {
        buildOpts += " -DNVIDIA -DNOFAILSAFE -DWARP=32";
    } else if (vendor_str.find("Intel") != std::string::npos) {
        buildOpts += " -DINTEL -DWARP=16 -DNOFAILSAFE";
    }

    err = clBuildProgram(program, 1, &gpu_device, buildOpts.c_str(), NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Build failed!" << std::endl;
        size_t log_size;
        clGetProgramBuildInfo(program, gpu_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, gpu_device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        std::cerr << "Build log: " << log.data() << std::endl;
        return 1;
    }

    

    // Create buffers
    cl_mem barrier_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, barrier_size, NULL, &err);
    cl_mem queue_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, queueSize, NULL, &err);  // Fixed: queueSize instead of queue_size
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);
    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, array_size, NULL, &err);

    // Initialize data
    std::vector<int> input_data(num_elements);
    for (int i = 0; i < num_elements; i++) {
        input_data[i] = i + 1;
    }

    std::vector<uint32_t> barrier_data(1000, 0);

    // Write data to buffers
    clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0, array_size, input_data.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, barrier_buf, CL_TRUE, 0, barrier_size, barrier_data.data(), 0, NULL, NULL);
    // clEnqueueWriteBuffer(queue, queue_buf, CL_TRUE, 0, queueSize, queueInitData.data(), 0, NULL, NULL);
    // Only write initial data for non-LCRQ queues
if (queueType != "LCRQ") {
    clEnqueueWriteBuffer(queue, queue_buf, CL_TRUE, 0, queueSize, queueInitData.data(), 0, NULL, NULL);
}

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "generic_queue_copy_test", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel!" << std::endl;
        return 1;
    }

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &barrier_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &queue_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_buf);
    clSetKernelArg(kernel, 4, sizeof(int), &num_elements);

    // Launch kernel
    size_t global[] = {256, 4};  // 4 work-groups of 256 threads each
    size_t local[] = {256, 1};   // Local work-group size

    std::cout << "Testing " << queueType << " queue with " << global[0] * global[1] << " threads in " 
              << (global[0] * global[1]) / (local[0] * local[1]) << " work-groups" << std::endl;

    // Initialize LCRQ if needed
if (queueType == "LCRQ") {
    cl_kernel init_kernel = clCreateKernel(program, "lcrq_init", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create init kernel!" << std::endl;
        return 1;
    }
    
    clSetKernelArg(init_kernel, 0, sizeof(cl_mem), &queue_buf);
    
    size_t one = 1;
    err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &one, &one, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to launch init kernel!" << std::endl;
        return 1;
    }
    
    clFinish(queue);  // Wait for initialization to complete
    clReleaseKernel(init_kernel);
    
    std::cout << "LCRQ queue initialized successfully" << std::endl;
}

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to launch kernel! Error: " << err << std::endl;
        return 1;
    }

    clFinish(queue);

    // Read back results
    std::vector<int> output_data(num_elements);
    clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, array_size, output_data.data(), 0, NULL, NULL);

    // Verify results
    int correct = 0;
    for (int i = 0; i < num_elements; i++) {
        if (output_data[i] == input_data[i]) {
            correct++;
        }
    }

    std::cout << "Test completed. Correct elements: " << correct << "/" << num_elements << std::endl;
    
    if (correct == num_elements) {
        std::cout << "SUCCESS: All elements processed correctly!" << std::endl;
    } else {
        std::cout << "PARTIAL SUCCESS: " << correct << " elements processed correctly" << std::endl;
    }

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseMemObject(barrier_buf);
    clReleaseMemObject(queue_buf);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}