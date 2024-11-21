//
// Created by Michał on 15.10.2024.
//

#ifndef CNN_CNN_H
#define CNN_CNN_H

#include <stddef.h>
#include <stdint.h>

void CNN_FcLayerForward(size_t inputLen, size_t outputLen, const float* input, const float* weights, const float* biases, float* output);

void CNN_ConvLayerForward_(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernelHeight, size_t kernelWidth, int strideH, int strideW, int paddingH, int paddingW, const float* input, const float* weights, const float* biases, float* output);

void CNN_ConvLayerForward(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernelHeight, size_t kernelWidth, int stride, int padding, const float* input, const float* weights, const float* biases, float* output);

void CNN_ConvLayerForwardDefault(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernel, const float* input, const float* weights, const float* biases, float* output);

void CNN_ReLU(size_t inputLen, float* input);

void CNN_MaxPoolForward_(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int strideH, int strideW, int paddingH, int paddingW, int ceilMode, const float* input, float* output);

void CNN_MaxPoolForward(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int stride, int padding, const float* input, float* output);

void CNN_MaxPoolForwardDefault(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernel, const float* input, float* output);

void CNN_PReLU(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, const float* weights);

void CNN_Softmax2D(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t dim, float* input);

void CNN_Softmax(size_t inputLen, float* input);

void CNN_Permute(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t outputHeight, size_t outputWidth, const float* input, float* output);

void CNN_BoxIou(size_t boxesLen, const float* boxes, size_t boxesLen2, const float* boxes2, float* output);

float CNN_Iou(float x, float y, float x2, float y2, float xp, float yp, float x2p, float y2p);

int CNN_BoxNms(size_t boxesLen, const float* boxes, const float* scores, float iouThreshold, float* output);

int CNN_BoxNmsIdx(size_t boxesLen, const float* boxes, const float* scores, float iouThreshold, int* boxesIndexes);

void CNN_AdaptiveAveragePool(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputHeight, size_t outputWidth, const float* input, float* output);

void CNN_AdaptiveAveragePool_Uint8(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputHeight, size_t outputWidth, const uint8_t * input, float* output);

#endif //CNN_CNN_H
