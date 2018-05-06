cmake_minimum_required(VERSION 3.0)
project(EdgesSubPix VERSION 1.1)

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
include(InstallRequiredSystemLibraries)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

OPTION(WITH_OPENMP "Use OpenMP acceleration" OFF)
OPTION(BUILD_TESTS "Enable testing" OFF)

if (WITH_OPENMP)
    FIND_PACKAGE(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#------------------------------------------------------------------------------
# Build
#------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_POSITION_INDEPENDENT_CODE YES)

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

if (BUILD_TESTS)
    enable_testing()
    add_test(NAME EdgeRuns COMMAND Edge ${CMAKE_CURRENT_SOURCE_DIR}/test.png)
endif(BUILD_TESTS)

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
configure_package_config_file(
    package-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    PATH_VARS
    CMAKE_INSTALL_INCLUDEDIR
    CMAKE_INSTALL_LIBDIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    )

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )


install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAMESPACE}Config
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RESOURCE DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAMESPACE}"
    )
