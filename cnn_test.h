//
// Created by Michał on 15.10.2024.
//

#ifndef CNN_CNN_TEST_H
#define CNN_CNN_TEST_H

#include <stdbool.h>

void CNNTest_Dev();

bool equalFloat(float x, float y, float margin);
bool equalFloatDefault(float x, float y);

void CNNTest_FcLayer0();
void CNNTest_FcLayer1();
void CNNTest_FcLayer2();
void CNNTest_ConvDev();
void CNNTest_Conv0();
void CNNTest_Conv1();
void CNNTest_Conv2();
void CNNTest_Conv3();
void CNNTest_Conv4();
void CNNTest_Conv5();
void CNNTest_Conv6();
void CNNTest_Conv7();
void CNNTest_Conv8();
void CNNTest_Conv9();
void CNNTest_Conv_0();
void CNNTest_Conv_1();
void CNNTest_Conv_2();
void CNNTest_ReLU();
void CNNTest_MaxPoolDefault0();
void CNNTest_MaxPoolDefault1();
void CNNTest_MaxPoolDefault2();
void CNNTest_MaxPoolDefault3();
void CNNTest_MaxPool0();
void CNNTest_MaxPool1();
void CNNTest_MaxPool_0();
void CNNTest_MaxPool_1();
void CNNTest_MaxPool_2();
void CNNTest_MaxPool_3();
void CNNTest_MaxPool_4();
void CNNTest_MaxPool_5();
void CNNTest_PReLU0();
void CNNTest_PReLU1();
void CNNTest_Softmax0();
void CNNTest_Softmax2D0();
void CNNTest_Softmax2D1();
void CNNTest_Softmax2D2();
void CNNTest_Permute0();
void CNNTest_Permute1();
void CNNTest_Permute2();
void CNNTest_BoxIou0();
void CNNTest_BoxIou1();
void CNNTest_BoxNms0();
void CNNTest_BoxNms1();
void CNNTest_AdaptiveAveragePool0();
void CNNTest_AdaptiveAveragePool1();

#endif //CNN_CNN_TEST_H
