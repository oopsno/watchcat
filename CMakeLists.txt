cmake_minimum_required (VERSION 3.5)
project (gleeman CXX)

set(CMAKE_CXX_STANDARD 14)
find_package(CUDA  8.0  REQUIRED)
find_package(Boost 1.63 REQUIRED)

set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
include_directories(${PROJECT_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})

include(cmake/catch.cmake)
set(CATCH_VERSION 1.9.3)
set(CATCH_SHA224  71871bcb95073f77fc29827b5a9dbbeaacc6f5fc5cd192f4dfe0aa16)
download_catch()

add_library(gleeman-uniform src/gleeman/error.cpp src/gleeman/uniform.cpp src/gleeman/exception.cpp src/gleeman/context.cpp)
target_link_libraries(gleeman-uniform ${CUDA_LIBRARIES})

add_library(gleeman-device src/gleeman/device.cpp)
target_link_libraries(gleeman-device ${CUDA_LIBRARIES})

add_executable(gleeman src/gleeman.cpp include/gleeman/context.hpp)
target_link_libraries(gleeman gleeman-device gleeman-uniform)

option(USE_NVML "enable NVML support")

enable_testing()
set(CATCH_MAIN test/catch_main.cpp)

cuda_add_executable(test_traits test/traits.cpp ${CATCH_MAIN})
add_test(NAME test_traits COMMAND test_traits)

add_executable(test_nat test/nat.cpp ${CATCH_MAIN})
add_test(NAME test_nat COMMAND test_nat)

add_executable(test_error test/error.cpp ${CATCH_MAIN})
target_link_libraries(test_error gleeman-uniform)
add_test(NAME test_error COMMAND test_error)

file(GLOB TEST_SOURCES test/*.cpp)
cuda_add_executable(test_all ${TEST_SOURCES})
target_link_libraries(test_all gleeman-uniform)
