cmake_minimum_required(VERSION 3.5)
project(wacdiff LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Parallelize
include(ProcessorCount)
ProcessorCount(N_CORES)
if(N_CORES GREATER 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N_CORES})
endif()

# Release optimizations
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release"
OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
        $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug"
OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")
endif()

option(WACDIFF_PROFILING "Enable profiling" OFF)
if(WACDIFF_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg -fno-omit-frame-pointer")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg -fno-omit-frame-pointer")
endif()

# Packages
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

# This code uses HDF5's MPI-IO API, so it must link against the parallel
# (MPI-enabled) HDF5 build rather than the serial one.
set(HDF5_PREFER_PARALLEL TRUE)
if(NOT HDF5_C_COMPILER_EXECUTABLE)
    find_program(HDF5_C_COMPILER_EXECUTABLE NAMES h5pcc h5cc)
endif()
if(NOT HDF5_CXX_COMPILER_EXECUTABLE)
    # Debian/Ubuntu name the parallel C++ wrapper h5c++.openmpi instead of
    # the h5pc++ name expected by FindHDF5.
    find_program(HDF5_CXX_COMPILER_EXECUTABLE
        NAMES h5c++.openmpi h5pc++ h5c++)
endif()
find_package(HDF5 REQUIRED COMPONENTS C CXX)

include(FetchContent)
FetchContent_Declare(
  pugixml
  GIT_REPOSITORY https://github.com/zeux/pugixml.git
  GIT_TAG        v1.16
)
FetchContent_MakeAvailable(pugixml)
FetchContent_Declare(
    multi
    GIT_REPOSITORY https://gitlab.com/correaa/boost-multi.git
)
FetchContent_MakeAvailable(multi)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.0
)
FetchContent_MakeAvailable(spdlog)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)
FetchContent_Declare(
    exprtk
    GIT_REPOSITORY https://github.com/ArashPartow/exprtk.git
    GIT_TAG        1e4a80b
)
FetchContent_MakeAvailable(exprtk)
add_library(exprtk INTERFACE)
target_include_directories(exprtk INTERFACE ${exprtk_SOURCE_DIR})

# Application
add_executable(wacdiff
    src/main.cpp
    src/distributed_grid.cpp
    src/dsg_archive.cpp
    src/initial_conditions.cpp
    src/solver.cpp
)
target_include_directories(wacdiff PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_precompile_headers(wacdiff PRIVATE src/pch.hpp)
target_link_libraries(wacdiff PUBLIC
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    hdf5::hdf5
    hdf5::hdf5_cpp
    pugixml::pugixml
    argparse
    multi
    spdlog
    exprtk
)
target_link_options(wacdiff PRIVATE -Wl,-z,muldefs)
