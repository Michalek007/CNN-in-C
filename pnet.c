//
// Created by Michał on 19.11.2024.
//

#include "pnet.h"
#include "pnet_weights.h"
#include "cnn.h"
#include <stdlib.h>
#include <math.h>


void PNet_Model(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, float* outputReg, float* outputProb){
//    float output1[6760];

    size_t newInputHeight = inputHeight - 2; size_t newInputWidth = inputWidth - 2;
    float* output1 = malloc(10*newInputHeight*newInputWidth*sizeof(float));
    CNN_ConvLayerForward_(inputChannels, inputHeight, inputWidth, 10, 3, 3, 1, 1, 0, 0, input, pnet_weight0, pnet_bias0, output1);
    inputHeight = newInputHeight; inputWidth = newInputWidth;

    CNN_PReLU(10, inputHeight, inputWidth, output1, pnet_weight1);
    free(input);
//    float output3[1690];
    newInputHeight = ceilf(inputHeight/2.0f); newInputWidth = ceilf(inputWidth/2.0f);
    float* output3 = malloc(10*newInputHeight*newInputWidth*sizeof(float));
    CNN_MaxPoolForward_(10, inputHeight, inputWidth, 2, 2, 2, 2, 0, 0, 1, output1, output3);
    inputHeight = newInputHeight; inputWidth = newInputWidth;
//    inputHeight = ceilf(inputHeight/2.0f); inputWidth = ceilf(inputWidth/2.0f);
    free(output1);
//    float output4[1936];
    newInputHeight = inputHeight - 2; newInputWidth = inputWidth - 2;
    float* output4 = malloc(16*newInputHeight*newInputWidth*sizeof(float));
    CNN_ConvLayerForward_(10, inputHeight, inputWidth, 16, 3, 3, 1, 1, 0, 0, output3, pnet_weight2, pnet_bias1, output4);
    inputHeight = newInputHeight; inputWidth = newInputWidth;
    CNN_PReLU(16, inputHeight, inputWidth, output4, pnet_weight3);
    free(output3);
//    float output6[2592];
    newInputHeight = inputHeight - 2; newInputWidth = inputWidth - 2;
    float* output6 = malloc(32*newInputHeight*newInputWidth*sizeof(float));
    CNN_ConvLayerForward_(16, inputHeight, inputWidth, 32, 3, 3, 1, 1, 0, 0, output4, pnet_weight4, pnet_bias2, output6);
    inputHeight = newInputHeight; inputWidth = newInputWidth;
    CNN_PReLU(32, inputHeight, inputWidth, output6, pnet_weight5);
    free(output4);

    CNN_ConvLayerForward_(32, inputHeight, inputWidth, 2, 1, 1, 1, 1, 0, 0, output6, pnet_weight6, pnet_bias3, outputProb);
    CNN_Softmax2D(2, inputHeight, inputWidth, 0, outputProb);

    CNN_ConvLayerForward_(32, inputHeight, inputWidth, 4, 1, 1, 1, 1, 0, 0, output6, pnet_weight7, pnet_bias4, outputReg);
    free(output6);
}

size_t PNet_GetOutputRegSize(size_t inputHeight, size_t inputWidth){
    size_t outputHeight = ceilf((inputHeight - 2) / 2.0f) - 4;
    size_t outputWidth = ceilf((inputWidth - 2) / 2.0f) - 4;
    size_t outputChannels = 4;
    return outputChannels * outputHeight * outputWidth;
}

size_t PNet_GetOutputProbSize(size_t inputHeight, size_t inputWidth){
    size_t outputHeight = ceilf((inputHeight - 2) / 2.0f) - 4;
    size_t outputWidth = ceilf((inputWidth - 2) / 2.0f) - 4;
    size_t outputChannels = 2;
    return outputChannels * outputHeight * outputWidth;
}
