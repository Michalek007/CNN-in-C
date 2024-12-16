cmake_minimum_required(VERSION 3.23)
project(CNN C)

set(CMAKE_C_STANDARD 99)

add_executable(CNN main.c src/cnn.c tests/cnn_test.c tests/cnn_test_networks.c models/mtcnn.c tests/mtcnn_test.c models/pnet.c tests/pnet_test.c models/rnet.c tests/rnet_test.c)

target_include_directories(CNN PRIVATE src/include)
target_include_directories(CNN PRIVATE models/include)
target_include_directories(CNN PRIVATE tests/include)