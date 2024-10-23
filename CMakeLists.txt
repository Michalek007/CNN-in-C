cmake_minimum_required(VERSION 3.23)
project(CNN C)

set(CMAKE_C_STANDARD 99)

add_executable(CNN main.c cnn.c cnn_test.c cnn_test.h)
