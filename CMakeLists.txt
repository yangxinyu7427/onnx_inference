cmake_minimum_required(VERSION 3.16)
project(OrtProject)

set(CMAKE_CXX_STANDARD 14)

link_directories (${PROJECT_SOURCE_DIR}/lib)
include_directories( ${PROJECT_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME}   testort.cpp)
target_link_libraries (${PROJECT_NAME}  libonnxruntime.so)