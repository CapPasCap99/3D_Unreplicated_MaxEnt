cmake_minimum_required(VERSION 3.6)
project(Processed_new_r0)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(/home/joris/Downloads/eigen-eigen-323c052e1731)
set(SOURCE_FILES main.cpp Initialize.h Moves.h Energy_changes.h global.h Functions.h)
add_executable(Raw_data_Inverse ${SOURCE_FILES})