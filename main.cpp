#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <climits>

#define __CL_ENABLE_EXTENSIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

// MS Queue layout calculation
struct ms_pointer_t {
    unsigned short count;
    unsigned short ptr;
};

struct ms_node_t {
    unsigned value;
    ms_pointer_t next;
    unsigned int free;
};

struct ms_queue_layout {
    ms_pointer_t head;
    ms_pointer_t tail;
    ms_node_t nodes[4097]; // MY_QUEUE_LENGTH + 1
    unsigned hazard1[1500];
    unsigned hazard2[1500];
    unsigned base_spin;
};

std::string getGPUName(cl_device_id device) {
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::string name(device_name);
    std::replace(name.begin(), name.end(), ' ', '_');
    return name;
}

std::string getVendorName(cl_device_id device) {
    char vendor[256];
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    return std::string(vendor);
}

// Forward declaration
void runThroughputTest(cl_context context, cl_command_queue command_queue, cl_program program, 
                      const std::string& queue_type, size_t queue_size, cl_device_id device);

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <queue_type>" << std::endl;
        std::cout << "queue_type: sfq, ms, tz" << std::endl;
        return 1;
    }
    
    std::string queue_type = argv[1];
    if (queue_type != "sfq" && queue_type != "ms" && queue_type != "tz") {
        std::cerr << "Error: queue_type must be sfq, ms, or tz" << std::endl;
        return 1;
    }
    
    cl_int err;
    
    // Get GPU device
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return 1;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), NULL);
    
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
    
    std::string gpu_name = getGPUName(gpu_device);
    std::string vendor = getVendorName(gpu_device);
    std::cout << "Using GPU: " << gpu_name << std::endl;
    std::cout << "Testing " << queue_type << " queue..." << std::endl;
    
    // Create context and command queue
    cl_context context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context!" << std::endl;
        return 1;
    }
    
    cl_command_queue command_queue = clCreateCommandQueue(context, gpu_device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue!" << std::endl;
        return 1;
    }
    
    // Read kernel source
    std::ifstream srcFile("kernels/queue_dispatch.cl");
    if (!srcFile) {
        std::cerr << "Error: Could not open kernels/queue_dispatch.cl" << std::endl;
        return 1;
    }
    std::string src(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));
    
    // Build options
    std::string buildOpts = "-I./kernels -DMY_QUEUE_LENGTH=4096 -DMY_QUEUE_FACTOR=12 -DGROUPS=256 -DWORK=100";
    
    // Queue-specific defines
    if (queue_type == "sfq") {
        buildOpts += " -DUSE_SFQ_QUEUE";
    } else if (queue_type == "ms") {
        buildOpts += " -DUSE_MS_QUEUE";
    } else if (queue_type == "tz") {
        buildOpts += " -DUSE_TZ_QUEUE";
    }
    
    // Vendor-specific optimizations with reasonable failsafe values
    if (vendor.find("AMD") != std::string::npos) {
        buildOpts += " -DAMD -DWARP=64 -DFAILSAFE=1000";
    } else if (vendor.find("NVIDIA") != std::string::npos) {
        buildOpts += " -DNVIDIA -DWARP=32 -DFAILSAFE=1000";
    } else if (vendor.find("Intel") != std::string::npos) {
        buildOpts += " -DINTEL -DWARP=16 -DFAILSAFE=1000";
    }
    
    std::cout << "Build options: " << buildOpts << std::endl;
    
    // Create and build program
    const char* src_ptr = src.c_str();
    size_t src_size = src.length();
    cl_program program = clCreateProgramWithSource(context, 1, &src_ptr, &src_size, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program!" << std::endl;
        return 1;
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
    
    std::cout << "Kernel built successfully!" << std::endl;
    
    // Calculate queue size
    size_t queue_size;
    if (queue_type == "ms") {
        queue_size = sizeof(ms_queue_layout);
        std::cout << "MS Queue size: " << queue_size << " bytes" << std::endl;
    } else if (queue_type == "sfq") {
        queue_size = 4096 * 3 * sizeof(uint32_t);
    } else if (queue_type == "tz") {
        queue_size = (4096 + 5) * sizeof(uint32_t);
    }
    
    // Run simple test first
    std::cout << "\n=== Running Simple Test ===" << std::endl;
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "simple_queue_test", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create simple_queue_test kernel! Error: " << err << std::endl;
        return 1;
    }
    
    const size_t barrier_size = 1000 * sizeof(uint32_t);
    const int num_threads = 64;
    
    cl_mem barrier_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, barrier_size, NULL, &err);
    cl_mem queue_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, queue_size, NULL, &err);
    cl_mem metrics_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_threads * 2 * sizeof(uint32_t), NULL, &err);
    cl_mem timing_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(uint64_t), NULL, &err);
    
    // Initialize queue data
    if (queue_type == "sfq") {
        std::vector<uint32_t> init_data(4096 * 3, 0);
        clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, init_data.size() * sizeof(uint32_t), init_data.data(), 0, NULL, NULL);
    } else if (queue_type == "ms") {
        ms_queue_layout init_queue = {};
        
        // Initialize head and tail to point to dummy node (node 1)
        init_queue.head.ptr = 1;
        init_queue.head.count = 0;
        init_queue.tail.ptr = 1;
        init_queue.tail.count = 0;
        
        // Initialize nodes
        for (int i = 0; i < 4097; i++) {
            if (i == 1) {
                init_queue.nodes[i].free = 1; // FREE_FALSE - dummy node is occupied
                init_queue.nodes[i].value = 0;
                init_queue.nodes[i].next.ptr = 0;
                init_queue.nodes[i].next.count = 0;
            } else {
                init_queue.nodes[i].free = 0; // FREE_TRUE - available
                init_queue.nodes[i].value = 0;
                init_queue.nodes[i].next.ptr = 0;
                init_queue.nodes[i].next.count = 0;
            }
        }
        
        // Initialize hazard arrays to UINT_MAX
        for (int i = 0; i < 1500; i++) {
            init_queue.hazard1[i] = UINT_MAX;
            init_queue.hazard2[i] = UINT_MAX;
        }
        
        init_queue.base_spin = 0;
        
        clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, sizeof(ms_queue_layout), &init_queue, 0, NULL, NULL);
        std::cout << "MS queue initialized with dummy node at index 1" << std::endl;
    } else if (queue_type == "tz") {
        std::vector<uint32_t> init_data(4096 + 5, 0);
        init_data[0] = 0;  // head
        init_data[1] = 1;  // tail
        init_data[2] = 4294967295;  // vnull
        for (int i = 3; i < 4096 + 5; i++) {
            init_data[i] = 4294967294;  // null_0
        }
        init_data[3] = 4294967295;  // first to null_1
        clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, init_data.size() * sizeof(uint32_t), init_data.data(), 0, NULL, NULL);
    }
    
    // Initialize barrier
    std::vector<uint32_t> barrier_data(1000, 0);
    clEnqueueWriteBuffer(command_queue, barrier_buf, CL_TRUE, 0, barrier_size, barrier_data.data(), 0, NULL, NULL);
    
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &barrier_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &queue_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &metrics_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &timing_buf);
    int pattern = 0;
    int operations = 1000;
    clSetKernelArg(kernel, 4, sizeof(int), &pattern);
    clSetKernelArg(kernel, 5, sizeof(int), &operations);
    
    // Launch kernel
    size_t global_size = 64;
    size_t local_size = 32;
    
    std::cout << "Launching simple test with " << global_size << " threads..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to launch simple test kernel! Error: " << err << std::endl;
        return 1;
    }
    
    clFinish(command_queue);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Simple test completed in " << duration.count() << " ms" << std::endl;
    
    // Read results
    std::vector<uint32_t> metrics(num_threads * 2);
    clEnqueueReadBuffer(command_queue, metrics_buf, CL_TRUE, 0, num_threads * 2 * sizeof(uint32_t), metrics.data(), 0, NULL, NULL);
    
    // Calculate total operations
    uint32_t total_ops = 0;
    uint32_t total_failures = 0;
    for (int i = 0; i < num_threads; i++) {
        total_ops += metrics[i * 2];
        total_failures += metrics[i * 2 + 1];
    }
    
    std::cout << "Total operations: " << total_ops << std::endl;
    std::cout << "Total failures: " << total_failures << std::endl;
    
    if (total_ops > 0) {
        std::cout << "SUCCESS: Simple test completed!" << std::endl;
        
        // Run SIMPLE queue validation first
        std::cout << "\n=== Running Queue Logic Validation ===" << std::endl;
        
        cl_kernel validate_kernel = clCreateKernel(program, "validate_queue_logic", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Could not create validate_queue_logic kernel, skipping..." << std::endl;
        } else {
            // Re-initialize queue for clean test
            if (queue_type == "sfq") {
                std::vector<uint32_t> init_data(4096 * 3, 0);
                clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, init_data.size() * sizeof(uint32_t), init_data.data(), 0, NULL, NULL);
            } else if (queue_type == "ms") {
                ms_queue_layout init_queue = {};
                init_queue.head.ptr = 1;
                init_queue.head.count = 0;
                init_queue.tail.ptr = 1;
                init_queue.tail.count = 0;
                
                for (int i = 0; i < 4097; i++) {
                    if (i == 1) {
                        init_queue.nodes[i].free = 1;
                        init_queue.nodes[i].value = 0;
                        init_queue.nodes[i].next.ptr = 0;
                        init_queue.nodes[i].next.count = 0;
                    } else {
                        init_queue.nodes[i].free = 0;
                        init_queue.nodes[i].value = 0;
                        init_queue.nodes[i].next.ptr = 0;
                        init_queue.nodes[i].next.count = 0;
                    }
                }
                
                for (int i = 0; i < 1500; i++) {
                    init_queue.hazard1[i] = UINT_MAX;
                    init_queue.hazard2[i] = UINT_MAX;
                }
                
                init_queue.base_spin = 0;
                clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, sizeof(ms_queue_layout), &init_queue, 0, NULL, NULL);
            } else if (queue_type == "tz") {
                std::vector<uint32_t> init_data(4096 + 5, 0);
                init_data[0] = 0;
                init_data[1] = 1;
                init_data[2] = 4294967295;
                for (int i = 3; i < 4096 + 5; i++) {
                    init_data[i] = 4294967294;
                }
                init_data[3] = 4294967295;
                clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, init_data.size() * sizeof(uint32_t), init_data.data(), 0, NULL, NULL);
            }
            
            // Create results buffer for 10 threads * 3 values each
            cl_mem validate_results_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 30 * sizeof(uint32_t), NULL, &err);
            
            clSetKernelArg(validate_kernel, 0, sizeof(cl_mem), &queue_buf);
            clSetKernelArg(validate_kernel, 1, sizeof(cl_mem), &validate_results_buf);
            
            // Launch 10 threads (1 producer, 9 consumers)
            size_t ten = 10;
            size_t local_ten = 10;
            
            std::cout << "Testing 1 producer + 9 consumers..." << std::endl;
            
            auto val_start = std::chrono::high_resolution_clock::now();
            
            err = clEnqueueNDRangeKernel(command_queue, validate_kernel, 1, NULL, &ten, &local_ten, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                std::cout << "Failed to launch validation kernel! Error: " << err << std::endl;
            } else {
                clFinish(command_queue);
                
                auto val_end = std::chrono::high_resolution_clock::now();
                auto val_duration = std::chrono::duration_cast<std::chrono::milliseconds>(val_end - val_start);
                
                // Read validation results
                std::vector<uint32_t> val_results(30);
                clEnqueueReadBuffer(command_queue, validate_results_buf, CL_TRUE, 0, 30 * sizeof(uint32_t), val_results.data(), 0, NULL, NULL);
                
                std::cout << "Validation completed in " << val_duration.count() << " ms" << std::endl;
                
                uint32_t total_produced = 0;
                uint32_t total_consumed = 0;
                uint32_t total_val_failures = 0;
                
                for (int i = 0; i < 10; i++) {
                    uint32_t ops = val_results[i * 3 + 0];
                    uint32_t failures = val_results[i * 3 + 1];
                    uint32_t thread_id = val_results[i * 3 + 2];
                    
                    if (thread_id == 0) {
                        total_produced += ops;
                        std::cout << "Producer (thread 0): produced=" << ops << ", failures=" << failures << std::endl;
                    } else {
                        total_consumed += ops;
                        std::cout << "Consumer " << thread_id << ": consumed=" << ops << ", failures=" << failures << std::endl;
                    }
                    
                    total_val_failures += failures;
                }
                
                std::cout << "\nValidation Summary:" << std::endl;
                std::cout << "Total produced: " << total_produced << std::endl;
                std::cout << "Total consumed: " << total_consumed << std::endl;
                std::cout << "Total failures: " << total_val_failures << std::endl;
                
                if (total_produced >= 15 && total_consumed >= 10) {
                    std::cout << "SUCCESS: Queue logic works correctly!" << std::endl;
                } else {
                    std::cout << "ISSUE: Queue may have problems - low throughput" << std::endl;
                }
            }
            
            clReleaseMemObject(validate_results_buf);
            clReleaseKernel(validate_kernel);
        }
        
        // NOW run the reordered throughput tests
        runThroughputTest(context, command_queue, program, queue_type, queue_size, gpu_device);
    } else {
        std::cout << "FAILED: No operations completed in simple test" << std::endl;
    }
    
    // Cleanup
    clReleaseKernel(kernel);
    clReleaseMemObject(barrier_buf);
    clReleaseMemObject(queue_buf);
    clReleaseMemObject(metrics_buf);
    clReleaseMemObject(timing_buf);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    
    return 0;
}

void runThroughputTest(cl_context context, cl_command_queue command_queue, cl_program program, 
                      const std::string& queue_type, size_t queue_size, cl_device_id device) {
    
    std::cout << "\n=== Running Throughput Tests ===" << std::endl;
    
    // Test configurations - REORDERED: lightest to heaviest workloads
    std::vector<std::string> test_names = {
        "scheduler_simulation",      // Lightest - mixed producer/consumer
        "bfs_simulation",           // Medium - graph traversal pattern  
        "burst_pattern_test",       // Heavy - burst loads
        "contention_pattern_test"   // HEAVIEST - high contention (do this LAST)
    };
    
    std::vector<int> thread_counts = {64, 128, 256, 512};
    std::vector<int> pattern_types = {0, 1,2,3};
    
    for (const auto& test_name : test_names) {
        std::cout << "\n--- Running " << test_name << " ---" << std::endl;
        
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, test_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            std::cout << "Kernel " << test_name << " not available, skipping..." << std::endl;
            continue;
        }
        
        for (int threads : thread_counts) {
            for (int pattern : pattern_types) {
                // Create buffers
                const size_t barrier_size = 1000 * sizeof(uint32_t);
                const int operations = 1000;  // Reduced from 1000 to 128 for RTX 3090
                
                cl_mem barrier_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, barrier_size, NULL, &err);
                cl_mem queue_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, queue_size, NULL, &err);
                cl_mem metrics_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, threads * sizeof(uint32_t), NULL, &err);
                cl_mem timing_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(uint64_t), NULL, &err);
                
                bool test_success = false;
                
                // Initialize queue (same as before)
                if (queue_type == "sfq") {
                    std::vector<uint32_t> init_data(4096 * 3, 0);
                    clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, init_data.size() * sizeof(uint32_t), init_data.data(), 0, NULL, NULL);
                } else if (queue_type == "ms") {
                    ms_queue_layout init_queue = {};
                    init_queue.head.ptr = 1;
                    init_queue.head.count = 0;
                    init_queue.tail.ptr = 1;
                    init_queue.tail.count = 0;
                    
                    for (int i = 0; i < 4097; i++) {
                        if (i == 1) {
                            init_queue.nodes[i].free = 1;
                            init_queue.nodes[i].value = 0;
                            init_queue.nodes[i].next.ptr = 0;
                            init_queue.nodes[i].next.count = 0;
                        } else {
                            init_queue.nodes[i].free = 0;
                            init_queue.nodes[i].value = 0;
                            init_queue.nodes[i].next.ptr = 0;
                            init_queue.nodes[i].next.count = 0;
                        }
                    }
                    
                    for (int i = 0; i < 1500; i++) {
                        init_queue.hazard1[i] = UINT_MAX;
                        init_queue.hazard2[i] = UINT_MAX;
                    }
                    
                    init_queue.base_spin = 0;
                    clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, sizeof(ms_queue_layout), &init_queue, 0, NULL, NULL);
                } else if (queue_type == "tz") {
                    std::vector<uint32_t> init_data(4096 + 5, 0);
                    init_data[0] = 0;
                    init_data[1] = 1;
                    init_data[2] = 4294967295;
                    for (int i = 3; i < 4096 + 5; i++) {
                        init_data[i] = 4294967294;
                    }
                    init_data[3] = 4294967295;
                    clEnqueueWriteBuffer(command_queue, queue_buf, CL_TRUE, 0, init_data.size() * sizeof(uint32_t), init_data.data(), 0, NULL, NULL);
                }
                
                // Initialize barrier
                std::vector<uint32_t> barrier_data(1000, 0);
                clEnqueueWriteBuffer(command_queue, barrier_buf, CL_TRUE, 0, barrier_size, barrier_data.data(), 0, NULL, NULL);
                
                // FIX 1: Initialize barrier properly to prevent deadlock
                cl_kernel barr = clCreateKernel(program, "barrier_init", &err);
                if (err == CL_SUCCESS) {
                    size_t one = 1;
                    clSetKernelArg(barr, 0, sizeof(cl_mem), &barrier_buf);
                    clSetKernelArg(barr, 1, sizeof(uint32_t), &threads);  // grid x-dim
                    clSetKernelArg(barr, 2, sizeof(uint32_t), &one);      // grid y-dim = 1
                    clEnqueueNDRangeKernel(command_queue, barr, 1, NULL, &one, &one, 0, NULL, NULL);
                    clFinish(command_queue);
                    clReleaseKernel(barr);
                }
                
                // Set kernel arguments
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &barrier_buf);
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &queue_buf);
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &metrics_buf);
                clSetKernelArg(kernel, 3, sizeof(cl_mem), &timing_buf);
                clSetKernelArg(kernel, 4, sizeof(int), &pattern);
                clSetKernelArg(kernel, 5, sizeof(int), &operations);
                
                // Launch kernel
                size_t global_size = threads;
                size_t local_size = std::min(threads, 256);
                while (global_size % local_size != 0) local_size--;
                
                auto start = std::chrono::high_resolution_clock::now();
                
                err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
                if (err == CL_SUCCESS) {
                    clFinish(command_queue);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    
                    // Read results
                    std::vector<uint32_t> metrics_data(threads);
                    clEnqueueReadBuffer(command_queue, metrics_buf, CL_TRUE, 0, threads * sizeof(uint32_t), metrics_data.data(), 0, NULL, NULL);
                    
                    uint32_t total_ops = 0;
                    for (uint32_t ops : metrics_data) {
                        total_ops += ops;
                    }
                    
                    double throughput = total_ops / (duration.count() / 1000000.0);
                    
                    std::cout << test_name << " - Threads: " << threads 
                             << ", Pattern: " << pattern 
                             << ", Ops: " << total_ops 
                             << ", Time: " << duration.count() << "us"
                             << ", Throughput: " << throughput << " ops/sec" << std::endl;
                    
                    test_success = true;
                } else {
                    std::cout << "Failed to launch " << test_name << " with " << threads << " threads, pattern " << pattern << std::endl;
                }
                
                // Cleanup buffers
                clReleaseMemObject(barrier_buf);
                clReleaseMemObject(queue_buf);
                clReleaseMemObject(metrics_buf);
                clReleaseMemObject(timing_buf);
            }
        }
        
        clReleaseKernel(kernel);
    }
}