cmake_minimum_required(VERSION 3.10)
project(watermarking_project)

# Option to enable verbose debug logging (compiled in with macro ENABLE_DEBUG_LOG)
option(ENABLE_DEBUG_LOG "Enable verbose debug logging" OFF)
if(ENABLE_DEBUG_LOG)
    add_compile_definitions(ENABLE_DEBUG_LOG)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES src/*.cpp)

add_executable(main ${SRC_FILES})
target_link_libraries(main PRIVATE ${ARMADILLO_LIBRARIES} ${OpenCV_LIBS})

enable_testing()
add_subdirectory(tests)
