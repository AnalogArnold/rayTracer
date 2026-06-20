cmake_minimum_required(VERSION 3.15)
project(rtmaincpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS --verbose) # Uncomment for troubleshooting in building

# Include Eigen headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Optimization flags
set(CMAKE_BUILD_TYPE Release) # not for now
#set(CMAKE_BUILD_TYPE Debug)

# Find Python and nanobind
find_package(OpenMP REQUIRED)
find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 3.4 QUIET)

if (Eigen3_FOUND)
    message(STATUS "Using system-installed Eigen3")
    if (TARGET Eigen3::Eigen)
        get_target_property(EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
        message(STATUS "Eigen3 include dir: ${EIGEN_INC}")
    endif()
    if (DEFINED Eigen3_DIR)
        message(STATUS "Eigen3 CMake config located at: ${Eigen3_DIR}")
    endif()
else()
    message(STATUS "System Eigen3 not found. fetching with FetchContent")
    include(FetchContent)

    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)

    message(STATUS "Using Eigen3 from FetchContent")
    message(STATUS "Eigen3 source dir: ${eigen_SOURCE_DIR}")
    message(STATUS "Eigen3 binary dir: ${eigen_BINARY_DIR}")
endif()


function(apply_compile_flags target_name)
    if(MSVC)
        # Windows (MSVC)
        target_compile_options(${target_name} PRIVATE /std:c++17 /openmp)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Applying MSVC debug flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE /Od /Zi)
        else()
            message(STATUS "Applying MSVC release flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE /O2)
        endif()
    else()
        # Linux / macOS
        target_compile_options(${target_name} PRIVATE -std=c++17 -fopenmp)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Applying debug flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE -O0 -g)
            target_link_options(${target_name} PRIVATE -fopenmp -g)
        else()
            message(STATUS "Applying release flags for ${target_name}")
            target_compile_options(${target_name} PRIVATE -O3)
            target_link_options(${target_name} PRIVATE -fopenmp)
        endif()
    endif()
endfunction()


# Troubleshooting messages in case detection/imports go wrong
#message("Python_FOUND:${Python_FOUND}")
#message("Python_VERSION:${Python_VERSION}")
#message("Python_Development_FOUND:${Python_Development_FOUND}")
#message("Python_LIBRARIES:${Python_LIBRARIES}")
#message("Python_Executable:${Python_EXECUTABLE}")
#message("Python_Interpreter_FOUND: ${Python_Interpreter_FOUND}")
#message("Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}")

# Gather all source C++ files under one variable to save ourselves having to list all of them manually
file(GLOB RAYTRACER_SRC_FILES *.cpp)
file(GLOB SOBOL_SRC_FILES sobol/*.cpp)

# Your source files
#add_library(rtmaincpp MODULE ${RAYTRACER_SRC_FILES})

# Link against nanobind and Python
nanobind_add_module(rtmaincpp MODULE ${RAYTRACER_SRC_FILES} ${SOBOL_SRC_FILES})
#target_link_libraries(rtmaincppcd 
#    PRIVATE
#        nanobind
#        Python3::Python
#)


target_include_directories(rtmaincpp PRIVATE
    ${PYTHON_INCLUDE_DIR}
)

target_link_libraries(rtmaincpp PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)

target_link_directories(rtmaincpp PRIVATE
    ${PYTHON_LIBRARY_DIR}
)

target_compile_options(rtmaincpp PRIVATE -O2 -fno-omit-frame-pointer)

# Ensure the output is a proper Python extension
if(WIN32)
    # Windows needs .pyd
    set_target_properties(rtmaincpp PROPERTIES
        PREFIX ""         # removes "lib" prefix
        SUFFIX ".pyd"
    )
else()
    # Linux/macOS
    set_target_properties(rtmaincpp PROPERTIES
        PREFIX ""         # removes "lib" prefix
        SUFFIX ".so"
    )
endif()

apply_compile_flags(rtmaincpp)

install(TARGETS rtmaincpp LIBRARY DESTINATION /home/micha/rayTracer/raytracer) # home laptop
#install(TARGETS rtmaincpp LIBRARY DESTINATION /home/tudent/rayTracer) # work laptop