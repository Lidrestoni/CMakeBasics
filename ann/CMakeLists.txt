add_definitions(-Wall -std=c++0x)
cmake_minimum_required(VERSION 2.8)
project( Ann )
find_package( OpenCV REQUIRED )
add_executable( Ann ann.cpp )
target_link_libraries( Ann ${OpenCV_LIBS} )
