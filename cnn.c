//
// Created by Michał on 15.10.2024.
//
#include "cnn.h"

void CNN_FcLayerForward(size_t inputLen, size_t outputLen, const float* input, const float* weights, const float* biases, float* output){
   for (size_t i=0;i<outputLen;++i){
       float outputValue = 0;
       for (size_t j=0;j<inputLen;++j){
           outputValue += weights[i*inputLen+j]*input[j];
       }
       output[i] = outputValue + biases[i];
   }
}

void CNN_ConvLayerForward(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t outputChannels, size_t kernelWidth, size_t kernelHeight,
                          int stride, int padding, const float* input, const float* weights, const float* biases, float* output){
    size_t outputWidth = (inputWidth-kernelWidth)/stride+1;
    size_t outputHeight = (inputHeight-kernelHeight)/stride+1;
    assert(outputWidth>0);
    assert(outputHeight>0);
    for (size_t o=0;o<outputChannels;++o){
        for (size_t i=0;i<outputWidth;i+=stride){
            for (size_t j=0;j<outputHeight;j+=stride){
                float outputValue = 0;
                for (size_t p =0;p<inputChannels;p++){
                    for (size_t k=0;k<kernelWidth;k++){
                        for (size_t l=0;l<kernelHeight;l++){
                            outputValue += input[p*inputWidth*inputHeight+(i+k)*inputWidth+j+l] * weights[o*kernelHeight*kernelWidth*inputChannels+p*kernelHeight*kernelWidth+k*kernelWidth+l];
                        }
                    }
                }
//                o*outputWidth*outputHeight+i*outputWidth+j
                output[o*outputWidth*outputHeight+i*outputWidth + j] = outputValue + biases[o];
            }
        }
    }
}

void CNN_ConvLayerForwardDefault(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t outputChannels, size_t kernel, const float* input, const float* weights, const float* biases, float* output){
    CNN_ConvLayerForward(inputChannels, inputWidth, inputHeight, outputChannels, kernel, kernel, 1, 0, input, weights, biases, output);
}

void CNN_ReLU(size_t inputLen, const float* input, float* output){
    for (size_t i=0;i<inputLen;++i){
        if (input[i] < 0){
            output[i] = 0;
        }
        else{
            output[i] = input[i];
        }
    }
}

void CNN_MaxPoolForward(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t kernelWidth, size_t kernelHeight, int stride, int padding, const float* input, float* output){
    size_t outputWidth = (inputWidth-kernelWidth)/stride+1;
    size_t outputHeight = (inputHeight-kernelHeight)/stride+1;
    assert(outputWidth>0);
    assert(outputHeight>0);
    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<inputWidth;i+=stride){
            for (size_t j=0;j<inputHeight;j+=stride){
                float maxValue = 0;
                for (size_t k=0;k<kernelWidth;k++){
                    for (size_t l=0;l<kernelHeight;l++){
                        float targetValue = input[o*inputWidth*inputHeight+(i+k)*inputWidth+j+l];
                        if (maxValue < targetValue || (k == 0 && l == 0)){
                            maxValue = targetValue;
                        }
                    }
                }
                output[o*outputWidth*outputHeight+i/stride*outputWidth + j/stride] = maxValue;
            }
        }
    }
}

void CNN_MaxPoolForwardDefault(size_t inputChannels, size_t inputWidth, size_t inputHeight, size_t kernel, const float* input, float* output){
    CNN_MaxPoolForward(inputChannels, inputWidth, inputHeight, kernel, kernel, (int)kernel, 0, input, output);
}
