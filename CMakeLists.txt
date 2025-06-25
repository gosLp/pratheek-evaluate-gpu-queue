cmake_minimum_required(VERSION 3.10)
project(QueuePerformanceTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Create executable
add_executable(queue_test main.cpp)

# Link OpenCL
target_link_libraries(queue_test OpenCL::OpenCL)

# Include directories (for OpenCL headers if needed)
target_include_directories(queue_test PRIVATE ${OpenCL_INCLUDE_DIRS})

# Copy kernel files to build directory
file(GLOB KERNEL_FILES "kernels/*.cl" "kernels/*.h")
foreach(KERNEL_FILE ${KERNEL_FILES})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME)
    configure_file(${KERNEL_FILE} ${CMAKE_BINARY_DIR}/kernels/${KERNEL_NAME} COPYONLY)
endforeach()

# Set compiler flags for debug/release
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Custom targets for testing different queue types
add_custom_target(test-sfq
    COMMAND queue_test sfq
    DEPENDS queue_test
    COMMENT "Testing SFQ queue with all patterns"
)

add_custom_target(test-ms
    COMMAND queue_test ms
    DEPENDS queue_test
    COMMENT "Testing MS queue with all patterns"
)

add_custom_target(test-tz
    COMMAND queue_test tz
    DEPENDS queue_test
    COMMENT "Testing TZ queue with all patterns"
)

add_custom_target(test-all
    COMMAND queue_test sfq
    COMMAND queue_test ms
    COMMAND queue_test tz
    DEPENDS queue_test
    COMMENT "Testing all queue types"
)