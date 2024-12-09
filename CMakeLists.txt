cmake_minimum_required (VERSION 3.0)

PROJECT(chapter_07)

set (CMAKE_CXX_STANDARD 11)

IF(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
ENDIF()

set(OpenCV_DIR "C:/OpenCV/opencv/build/x64/vc16/lib")

# Requires OpenCV
FIND_PACKAGE( OpenCV 4.0.0 REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

ADD_EXECUTABLE( DetectAndOverlay DetectAndOverlay.cpp )
TARGET_LINK_LIBRARIES( DetectAndOverlay ${OpenCV_LIBS} )
