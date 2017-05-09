cmake_minimum_required (VERSION 3.5)
project (gleeman CXX)

set(CMAKE_CXX_STANDARD 14)

if (NOT APPLE)
    find_package(CUDA 8.0 COMPONENTS nvml)
    set(USE_CUDA YES)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
    set(CUDA_RUNTIME ${CUDA_LIBRARIES})
    if (WIN32)
        string(REPLACE cudart_static cuda CUDA_DRIVER ${CUDA_LIBRARIES})
        string(REPLACE cudart_static nvml CUDA_NVML   ${CUDA_LIBRARIES})
    endif()
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
find_package(Boost REQUIRED COMPONENTS program_options)

add_subdirectory(3rd/yaml-cpp)
add_subdirectory(3rd/cryptopp)
add_subdirectory(3rd/Catch)

set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
include_directories(${PROJECT_INCLUDE_DIR}
                    ${Boost_INCLUDE_DIRS}
                    3rd/yaml-cpp/include
                    3rd/cryptopp
                    3rd/Catch/single_include)

option(USE_NVML "enable NVML support")
option(USE_NAT  "enable NAT  support")

if (USE_NVML)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVML")
endif( )

if (USE_NAT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NAT")
endif()

add_library(gleeman-universal src/gleeman/error.cpp
                              src/gleeman/exception.cpp
                              src/gleeman/context.cpp
                              src/gleeman/crypto.cpp
                              src/gleeman/job.cpp
                              src/gleeman/yaml.cpp
                              src/gleeman/executable.cpp
                              src/gleeman/nat.cpp)
target_link_libraries(gleeman-universal cryptopp-static)

add_library(gleeman-device src/gleeman/device.cpp
                           src/gleeman/placeholder.cpp)
target_link_libraries(gleeman-device ${CUDA_RUNTIME} ${CUDA_DRIVER})

add_executable(gleeman src/gleeman.cpp)
target_link_libraries(gleeman gleeman-universal gleeman-device ${Boost_LIBRARIES})

enable_testing()
set(CATCH_MAIN test/catch_main.cpp)

add_executable(test_traits test/traits.cpp ${CATCH_MAIN})
add_test(NAME test_traits COMMAND test_traits)

add_executable(test_nat test/nat.cpp ${CATCH_MAIN})
add_test(NAME test_nat COMMAND test_nat)

add_executable(test_error test/error.cpp ${CATCH_MAIN})
target_link_libraries(test_error gleeman-universal)
add_test(NAME test_error COMMAND test_error)

add_executable(test_context test/context.cpp ${CATCH_MAIN})
target_link_libraries(test_context gleeman-universal)
add_test(NAME test_context COMMAND test_context)

file(GLOB TEST_SOURCES test/*.cpp)
add_executable(test_all ${TEST_SOURCES})
target_link_libraries(test_all gleeman-universal)
