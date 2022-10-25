cmake_minimum_required(VERSION 2.6)

project(yolop)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(ZED 3)
if(ZED_FOUND)
    find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
else(ZED_FOUND)
    find_package(CUDA  REQUIRED)
endif(ZED_FOUND)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)
# zed
include_directories(/usr/local/zed/include)
link_directories(/usr/local/zed/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# to generate plugins
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

# to generate trt and test image dir
add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(yolov5 nvinfer cudart myplugins ${OpenCV_LIBS})
add_definitions(-O3 -pthread)


# to test with zed camera
if(ZED_FOUND)
    set(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

    add_executable(yolop ${PROJECT_SOURCE_DIR}/main.cpp)
    target_link_libraries(yolop nvinfer)
    target_link_libraries(yolop ${ZED_LIBS})
    target_link_libraries(yolop cudart)
    target_link_libraries(yolop myplugins)
    target_link_libraries(yolop ${OpenCV_LIBS})
endif(ZED_FOUND)
