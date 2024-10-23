#include <stdio.h>
#include "cnn.h"
#include "cnn_test.h"


int main() {
    printf("Hello, World!\n");

//    CNN_dev();
//
//    CNN_testFcLayer0();
//    CNN_testFcLayer1();
//    CNN_testConv0();
//    CNN_testConvDev();
//    CNN_testConv1();
//    CNN_testConv2();
//    CNN_testConv3();
//    CNN_testReLU();
//    CNN_testMaxPoolDefault0();
//    CNN_testMaxPoolDefault1();
//    CNN_testMaxPoolDefault2();

    CNN_testFashionMnist();
//    CNN_testFashionMnist1();
    return 0;
}
