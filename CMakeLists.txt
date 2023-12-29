cmake_minimum_required(VERSION 3.2)
project(rtdetr)                                
set(OpenCV_DIR "E:\\Opencv\\opencv_vs\\build")    
set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}\\include)
set(OpenCV_LIB_DIRS ${OpenCV_DIR}\\x64\\vc16\\lib) 
set(OpenCV_Debug_LIBS "opencv_world480d.lib")     
set(OpenCV_Release_LIBS "opencv_world480.lib")   
set(TRT_DIR "E:\\Tensorrt\\8.6") 
set(TRT_SAM_DIR "E:\\Tensorrt\\8.6\\samples\\common")   
set(TRT_INCLUDE_DIRS ${TRT_DIR}\\include) 
set(TRT_LIB_DIRS ${TRT_DIR}\\lib)          
add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA)  
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_INCLUDE_DIRS}) 
link_directories(${TRT_LIB_DIRS}) 
include_directories(${OpenCV_INCLUDE_DIRS}) 
link_directories(${OpenCV_LIB_DIRS})  
include_directories(${TRT_SAM_DIR}) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    lib path: ${OpenCV_LIB_DIRS}")
message(STATUS "    Debug libraries: ${OpenCV_Debug_LIBS}")
message(STATUS "    Release libraries: ${OpenCV_Release_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
add_executable(rtdetr ${PROJECT_SOURCE_DIR}/rtdetr.cpp ${PROJECT_SOURCE_DIR}/logger.cpp ${PROJECT_SOURCE_DIR}/logger.h)   
target_link_libraries(rtdetr "nvinfer" "nvinfer_plugin") 
target_link_libraries(rtdetr debug ${OpenCV_Debug_LIBS}) 
target_link_libraries(rtdetr optimized ${OpenCV_Release_LIBS}) 
target_link_libraries(rtdetr ${CUDA_LIBRARIES}) 
