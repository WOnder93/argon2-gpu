cmake_minimum_required(VERSION 2.6)

project(argon2-gpu CXX)
set(BINARY_INSTALL_DIR /usr/local/bin)
set(LIBRARY_INSTALL_DIR /usr/local/lib)
set(INCLUDE_INSTALL_DIR /usr/local/include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(argon2-opencl SHARED
    lib/argon2-opencl/argon2params.cpp
    lib/argon2-opencl/blake2b.cpp
    lib/argon2-opencl/device.cpp
    lib/argon2-opencl/globalcontext.cpp
    lib/argon2-opencl/kernelloader.cpp
    lib/argon2-opencl/programcontext.cpp
    lib/argon2-opencl/processingunit.cpp
)
target_include_directories(argon2-opencl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-opencl PRIVATE include/argon2-opencl lib/argon2-opencl)
target_link_libraries(argon2-opencl -lOpenCL)

add_executable(argon2-opencl-test src/argon2-opencl-test/main.cpp)
target_include_directories(argon2-opencl-test PRIVATE src/argon2-opencl-test)
target_link_libraries(argon2-opencl-test
    argon2-opencl -lOpenCL
)

add_executable(argon2-opencl-bench
    src/argon2-opencl-bench/benchmark.cpp
    src/argon2-opencl-bench/main.cpp
)
target_include_directories(argon2-opencl-bench PRIVATE src/argon2-opencl-bench)
target_link_libraries(argon2-opencl-bench
    argon2-opencl -lOpenCL
)

add_test(test argon2-opencl-test)

install(TARGETS argon2-opencl DESTINATION ${LIBRARY_INSTALL_DIR})
install(FILES
    include/argon2-opencl/cl.hpp
    include/argon2-opencl/opencl.h
    include/argon2-opencl/argon2-common.h
    include/argon2-opencl/argon2params.h
    include/argon2-opencl/device.h
    include/argon2-opencl/globalcontext.h
    include/argon2-opencl/programcontext.h
    include/argon2-opencl/processingunit.h
    DESTINATION ${INCLUDE_INSTALL_DIR}
)
install(TARGETS argon2-opencl-bench argon2-opencl-test DESTINATION ${BINARY_INSTALL_DIR})
