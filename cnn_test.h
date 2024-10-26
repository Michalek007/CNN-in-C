//
// Created by Michał on 15.10.2024.
//

#ifndef CNN_CNN_TEST_H
#define CNN_CNN_TEST_H

#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "cnn.h"

void CNNTest_Dev();

void CNNTest_FcLayer0();
void CNNTest_FcLayer1();
void CNNTest_ConvDev();
void CNNTest_Conv0();
void CNNTest_Conv1();
void CNNTest_Conv2();
void CNNTest_Conv3();
void CNNTest_ReLU();
void CNNTest_MaxPoolDefault0();
void CNNTest_MaxPoolDefault1();
void CNNTest_MaxPoolDefault2();

void CNNTest_FashionMnist();
void CNNTest_FashionMnist1();

#endif //CNN_CNN_TEST_H
