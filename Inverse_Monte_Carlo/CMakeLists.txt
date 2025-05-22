cmake_minimum_required(VERSION 3.6)
project(Processed_new_r0)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(/home/capucine/Documents/test/Inverse_Monte_Carlo/eigen-3.4.0)
set(SOURCE_FILES main.cpp Initialize.h Moves.h Energy_changes.h global.h Functions.h)
add_executable(Processed_new_r0 ${SOURCE_FILES})