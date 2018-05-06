cmake_minimum_required(VERSION 3.0)
project(EdgesSubPix)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

OPTION(WITH_OPENMP "Use OpenMP acceleration" OFF)

if (WITH_OPENMP)
    FIND_PACKAGE(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Compiler 
if (WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
endif()

add_library(${PROJECT_NAME} STATIC EdgesSubPix.cpp EdgesSubPix.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(Edge test_EdgesSubPix.cpp)
target_link_libraries(Edge ${PROJECT_NAME} ${OpenCV_LIBS})

