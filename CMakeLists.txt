cmake_minimum_required(VERSION 2.8)
project(MCV_ORB_SLAM)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -std=c++11")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 2.4.3 QUIET)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
  endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include/
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/utils.cpp
src/system.cpp
src/camera_system.cpp
src/camera_model.cpp
src/converter.cpp
src/feature_extractor.cpp
src/keyframe_database.cpp
src/tracking.cpp
src/local_mapping.cpp
src/loop_closing.cpp
src/viewer.cpp
src/map_publisher.cpp
src/frame_publisher.cpp
src/multi_video.cpp
src/shift_video.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/third_party/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/third_party/g2o/lib/libg2o.so
)

add_executable(MCV_ORB_SLAM_example example/example.cpp)
target_link_libraries(MCV_ORB_SLAM_example ${PROJECT_NAME})