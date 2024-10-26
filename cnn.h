//
// Created by Michał on 15.10.2024.
//

#ifndef CNN_CNN_H
#define CNN_CNN_H

#include <stddef.h>
#include <assert.h>

void CNN_FcLayerForward(size_t inputLen, size_t outputLen, const float* input, const float* weights, const float* biases, float* output);

void CNN_ConvLayerForward(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t outputChannels, size_t kernelWidth, size_t kernelHeight, int stride, int padding, const float* input, const float* weights, const float* biases, float* output);

void CNN_ConvLayerForwardDefault(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t outputChannels, size_t kernel, const float* input, const float* weights, const float* biases, float* output);

void CNN_ReLU(size_t inputLen, const float* input, float* output);

void CNN_MaxPoolForward(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t kernelWidth, size_t kernelHeight, int stride, int padding, const float* input, float* output);

void CNN_MaxPoolForwardDefault(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t kernel, const float* input, float* output);


#endif //CNN_CNN_H
