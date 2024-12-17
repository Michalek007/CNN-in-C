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
void CNNTest_ConvSymmetric0();
void CNNTest_ConvSymmetric1();
void CNNTest_ConvSymmetric2();
void CNNTest_ConvSymmetric3();
void CNNTest_ConvSymmetric4();
void CNNTest_ConvSymmetric5();
void CNNTest_ConvSymmetric6();
void CNNTest_ConvSymmetric7();
void CNNTest_ConvSymmetric8();
void CNNTest_ConvSymmetric9();
void CNNTest_Conv0();
void CNNTest_Conv1();
void CNNTest_Conv2();
void CNNTest_ReLU();
void CNNTest_MaxPoolBasic0();
void CNNTest_MaxPoolBasic1();
void CNNTest_MaxPoolBasic2();
void CNNTest_MaxPoolBasic3();
void CNNTest_MaxPoolSymmetric0();
void CNNTest_MaxPoolSymmetric1();
void CNNTest_MaxPool0();
void CNNTest_MaxPool1();
void CNNTest_MaxPool2();
void CNNTest_MaxPool3();
void CNNTest_MaxPool4();
void CNNTest_MaxPool5();
void CNNTest_PReLU0();
void CNNTest_PReLU1();
void CNNTest_Softmax0();
void CNNTest_Softmax2D0();
void CNNTest_Softmax2D1();
void CNNTest_Softmax2D2();
void CNNTest_Permute0();
void CNNTest_Permute1();
void CNNTest_Permute2();
void CNNTest_Permute3();
void CNNTest_Permute4();
void CNNTest_BoxIou0();
void CNNTest_BoxIou1();
void CNNTest_BoxNms0();
void CNNTest_BoxNmsIdx0();
void CNNTest_AdaptiveAveragePool0();
void CNNTest_AdaptiveAveragePool1();
void CNNTest_AdaptiveAveragePool2();
void CNNTest_AdaptiveAveragePool3();
void CNNTest_AdaptiveAveragePool4();
void CNNTest_AveragePoolBasic0();
void CNNTest_LeakyReLU0();
void CNNTest_BatchNorm0();

#endif //CNN_CNN_TEST_H
