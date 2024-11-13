//
// Created by Michał on 15.10.2024.
//
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
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

void CNN_ConvLayerForward_(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernelHeight, size_t kernelWidth, int strideH, int strideW, int paddingH, int paddingW, const float* input, const float* weights, const float* biases, float* output){
    size_t outputHeight = (inputHeight-kernelHeight+2*paddingH)/strideH+1;
    size_t outputWidth = (inputWidth-kernelWidth+2*paddingW)/strideW+1;
    assert(outputHeight>0);
    assert(outputWidth>0);

    if (paddingH != 0 || paddingW != 0){
        int newHeight = inputHeight+2*paddingH;
        int newWidth = inputWidth+2*paddingW;
        size_t newInputSize = newHeight*newWidth*sizeof(float);
        float *newInput = (float*) malloc(newInputSize);
        memset(newInput, 0, newInputSize);

        for (int i=0;i<inputHeight;i++){
            memcpy(newInput+(newWidth*paddingH)+paddingW+i*newWidth, input+i*inputWidth, inputWidth*sizeof(float));
        }

        input = newInput;
        inputHeight = newHeight;
        inputWidth = newWidth;
    }

    for (size_t o=0;o<outputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float outputValue = 0;
                for (size_t p =0;p<inputChannels;p++){
                    for (size_t k=0;k<kernelHeight;k++){
                        for (size_t l=0;l<kernelWidth;l++){
                            int weightsIndex = o*kernelWidth*kernelHeight*inputChannels + p*kernelWidth*kernelHeight + k*kernelWidth +l;
                            int inputIndex = p*inputHeight*inputWidth + (i*strideH+k)*inputWidth + j*strideW + l;
                            outputValue += input[inputIndex] * weights[weightsIndex];
                        }
                    }
                }
                int outputIndex = o*outputHeight*outputWidth + i*outputWidth + j;
                output[outputIndex] = outputValue + biases[o];
            }
        }
    }
}

void CNN_ConvLayerForward(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernelHeight, size_t kernelWidth,
                          int stride, int padding, const float* input, const float* weights, const float* biases, float* output){
    CNN_ConvLayerForward_(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, stride, padding, padding, input, weights, biases, output);
}

void CNN_ConvLayerForwardDefault(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t kernel, const float* input, const float* weights, const float* biases, float* output){
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernel, kernel, 1, 0, input, weights, biases, output);
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

void CNN_MaxPoolForward_(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int strideH, int strideW, int paddingH, int paddingW, int ceilMode, const float* input, float* output){
    size_t outputHeight, outputWidth;
    if (ceilMode){
        outputHeight = ceilf((inputHeight-kernelHeight+2*paddingH)/(float)strideH+1);
        outputWidth = ceilf((inputWidth-kernelWidth+2*paddingW)/(float)strideW+1);
    }
    else{
        outputHeight = (inputHeight-kernelHeight+2*paddingH)/strideH+1;
        outputWidth = (inputWidth-kernelWidth+2*paddingW)/strideW+1;
    }
    assert(outputHeight>0);
    assert(outputWidth>0);

    int paddingRight = 0;
    int paddingBottom = 0;
    if (ceilMode){
        paddingRight = (outputWidth-1)*strideW + kernelWidth - inputWidth;
        paddingBottom = (outputHeight-1)*strideH + kernelHeight - inputHeight;
    }

    if (paddingH != 0 || paddingW != 0 || paddingRight !=0 || paddingBottom != 0){
        int newHeight = inputHeight+2*paddingH + paddingBottom;
        int newWidth = inputWidth+2*paddingW + paddingRight;
        size_t newInputSize = newHeight*newWidth*sizeof(float);
        float *newInput = (float*) malloc(newInputSize);
        memset(newInput, 0, newInputSize);

        for (int i=0;i<inputHeight;i++){
            memcpy(newInput+(newWidth*paddingH)+paddingW+i*newWidth, input+i*inputWidth, inputWidth*sizeof(float));
        }

        input = newInput;
        inputHeight = newHeight;
        inputWidth = newWidth;
    }

    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<outputHeight;i++){
            for (size_t j=0;j<outputWidth;j++){
                float maxValue = 0;
                for (size_t k=0;k<kernelHeight;k++){
                    for (size_t l=0;l<kernelWidth;l++){
                        int inputIndex = o*inputHeight*inputWidth + (i*strideH+k)*inputWidth + j*strideW + l;
                        float targetValue = input[inputIndex];
                        if (maxValue < targetValue || (k == 0 && l == 0)){
                            maxValue = targetValue;
                        }
                    }
                }
                int outputIndex = o*outputHeight*outputWidth+i*outputWidth + j;
                output[outputIndex] = maxValue;
            }
        }
    }
}

void CNN_MaxPoolForward(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernelHeight, size_t kernelWidth, int stride, int padding, const float* input, float* output){
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, stride, padding, padding, 0, input, output);
}

void CNN_MaxPoolForwardDefault(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t kernel, const float* input, float* output){
    CNN_MaxPoolForward(inputChannels, inputHeight, inputWidth, kernel, kernel, (int)kernel, 0, input, output);
}

void CNN_PReLU(size_t inputChannels, size_t inputHeight, size_t inputWidth, const float* input, const float* weights, float* output){
    for (size_t o=0;o<inputChannels;++o){
        for (size_t i=0;i<inputHeight;++i){
            for (size_t j=0;j<inputWidth;++j){
                size_t index = o*inputHeight*inputWidth + i*inputWidth + j;
                float a = 1;
                if (input[index] < 0)
                    a = weights[o];
                output[index] = a * input[index];
            }
        }
    }
}

void CNN_Softmax(size_t inputLen, const float* input, float* output){
    float sum = 0;
    for (size_t i=0;i<inputLen;++i){
        output[i] = expf(input[i]);
        sum += output[i];
    }
    for (size_t i=0;i<inputLen;++i){
        output[i] = output[i] / sum;
    }
}

void CNN_Softmax2D(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t dim, const float* input, float* output){
    size_t firstLoop = inputChannels;
    size_t secondLoop = inputHeight;
    size_t thirdLoop = inputWidth;
    if (dim == 0){
        firstLoop = inputHeight;
        secondLoop = inputWidth;
        thirdLoop = inputChannels;
    }
    else if (dim == 1){
        firstLoop = inputChannels;
        secondLoop = inputWidth;
        thirdLoop = inputHeight;
    }


    for (size_t o=0;o<firstLoop;++o){
        for (size_t i=0;i<secondLoop;++i){
            float sum = 0;
            size_t step = 0;
            while (step<2){
                for (size_t j=0;j<thirdLoop;++j){
                    size_t index;
                    if (dim == 0){
                        index =  j*inputHeight*inputWidth + o*inputWidth + i;
                    }
                    else if (dim == 1){
                        index =  o*inputHeight*inputWidth + j*inputWidth + i;
                    }
                    else{
                        index =  o*inputHeight*inputWidth + i*inputWidth + j;
                    }

                    if (step == 0){
                        output[index] = expf(input[index]);
                        sum += output[index];
                    }
                    else if (step == 1){
                        output[index] = output[index] / sum;
                    }
                }
                ++step;
            }
        }
    }
}

void CNN_Permute(size_t inputChannels, size_t inputHeight, size_t inputWidth, size_t outputChannels, size_t outputHeight, size_t outputWidth, const float* input, float* output){

}
