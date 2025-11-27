cmake_minimum_required(VERSION 3.15)
project(rtmaincpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS --verbose) # Uncomment for troubleshooting in building

# Include Eigen headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Optimization flags
set(CMAKE_BUILD_TYPE Release)

# Find Python and nanobind
find_package(Python 3.11 COMPONENTS Interpreter Development EXACT REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

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

# Your source files
#add_library(rtmaincpp MODULE ${RAYTRACER_SRC_FILES})

# Link against nanobind and Python
nanobind_add_module(rtmaincpp MODULE ${RAYTRACER_SRC_FILES})
#target_link_libraries(rtmaincpp
#    PRIVATE
#        nanobind
#        Python3::Python
#)


target_include_directories(rtmaincpp PRIVATE
    ${PYTHON_INCLUDE_DIR}
)

target_link_directories(rtmaincpp PRIVATE
    ${PYTHON_LIBRARY_DIR}
)

target_compile_options(rtmaincpp PRIVATE -O2)

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

install(TARGETS rtmaincpp LIBRARY DESTINATION /home/tudent/rayTracer)