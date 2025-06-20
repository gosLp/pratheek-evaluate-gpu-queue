#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>
#include <fstream>
#include <vector>
#include <string>
#include <memory>

#define __CL_ENABLE_EXTENSIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/opencl.hpp>
#endif

int main (int argc, char **argv) {
   try {
    // Read the file queue_dispatch source cl kernel
    std::ifstream srcFile("kernels/queue_dispatch.cl");
    if (!srcFile) {
        std::cerr << "Error no file queue dispatch found!";
        return 1;
    }
    std::string src(std::istreambuf_iterator<char>(srcFile), (std::istreambuf_iterator<char>()));

    // 2. Get all platforms and pick the first GPU device we find
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    if(platforms.empty()){ 
        std::cerr << "No GPu platforms found! \n";
        return 1;
    }

    cl::Device gpu;
    bool found = false;
    for ( auto &plat : platform) {
        std::vector<cl::Device> devices;
        plat.getDevices(CL_DEVICE_TYPE_GPU, devices);
        if (!devices.empty()) {
            gpu = devices.front();
            found = true;
            break;
        }
    }

    if (!found) {
        std::cerr <<"No Gpu devices found \n";
        return 1;
    }

    std::cout << " Using Gpu device : " << gpu.getInfo<CL_DEVICE_NAME>() << "\n";

    // 3. Create context & queue
    cl::Context context;
    cl::CommandQueue queue(context, gpu);

    // 4. Build program, with "-I." so that #includes in your .cl resolve

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.size()));
    cl::Program program(context, sources);
    
    std::String buildOpts = "-I. -DMY_QUEUE_LENGTH=4096";

    std::String vendor = gpu.getInfo<CL_DEVICE_VENDOR>();

    if (vendor.find("AMD") != std::string::npos) {
        buildOpts += "-DAMD -DWARP=64 -DFAILSAFE=100000";
    } else if (vendor.find("NVIDIA") != std::string::npos) {
        buildOpts += "-DNVIDIA -DNOFAILSAFE";
    } else if (vendor.find("Intel") != std::string::npos) {
        buildOpts += "-DINTEL -DWAPP=16 -DNOFAILSAFE";
    }

    program.build({ buildOpts });

    // 5. Create and launch the SFQ kernel
    cl::Kernel kernel(program, 'ms_queue_copy_test');

    kernel.setArg(0, someBuffer);
    kernel.setArg(1, anotherBuffer);


    // Launch one work-group of one thread just to verify it runs
    cl::NDRange global(1);
    cl::NDRange local (1);


   } 
}


