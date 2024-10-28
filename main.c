#include <stdio.h>
#include "cnn.h"
#include "cnn_test.h"


int main() {
    printf("Hello, World!\n");

    CNNTest_Dev();

    CNNTest_FcLayer0();
    CNNTest_FcLayer1();
    CNNTest_FcLayer2();
    CNNTest_Conv0();
    CNNTest_ConvDev();
    CNNTest_Conv1();
    CNNTest_Conv2();
    CNNTest_Conv3();
    CNNTest_Conv4();
    CNNTest_Conv5();
    CNNTest_Conv6();
    CNNTest_Conv7();
    CNNTest_Conv8();
    CNNTest_Conv9();
    CNNTest_Conv_0();
    CNNTest_Conv_1();
    CNNTest_ReLU();
    CNNTest_MaxPoolDefault0();
    CNNTest_MaxPoolDefault1();
    CNNTest_MaxPoolDefault2();
    CNNTest_MaxPoolDefault3();
    CNNTest_MaxPool0();
    CNNTest_MaxPool1();
    CNNTest_MaxPool_0();
    CNNTest_MaxPool_1();

    CNNTest_FashionMnist();
    CNNTest_FashionMnist1();
    CNNTest_FashionMnist2();
    return 0;
}
