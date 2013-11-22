CMAKE_MINIMUM_REQUIRED (VERSION 2.6)
PROJECT (allthingsfintech)
SET(CMAKE_CXX_COMPILER "/usr/bin/g++")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
SET(CMAKE_BUILD_TYPE "Debug")

ADD_EXECUTABLE(allthingsfintech src/main.cpp)
INCLUDE_DIRECTORIES("/usr/local/include")
TARGET_LINK_LIBRARIES(allthingsfintech boost_unit_test_framework 
boost_test_exec_monitor QuantLib)


 



