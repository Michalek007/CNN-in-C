//
// Created by Michał on 19.11.2024.
//

#include "pnet_test.h"
#include "cnn_test.h"
#include "pnet_test_data.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pnet.h"

void PnetTest_Model0(){
    size_t inputChannels = 3;
    size_t inputHeight = 61;
    size_t inputWidth = 61;
    size_t inputSize = inputChannels*inputHeight*inputWidth*sizeof(float);
    float* input = malloc(inputSize);
    memcpy(input, modelInput0, inputSize);
//    for (size_t i=0;i<inputChannels*inputHeight*inputWidth;++i){
//        input[i] = modelInput0[i];
//    }
    size_t outputProbSize = PNet_GetOutputProbSize(inputHeight, inputWidth);
    size_t outputRegSize = PNet_GetOutputRegSize(inputHeight, inputWidth);
//    float* outputProb = malloc(outputProbSize);
//    float* outputReg = malloc(outputRegSize);
    float outputProb[outputProbSize];
    float outputReg[outputRegSize];

    PNet_Model(inputChannels, inputHeight, inputWidth, input, outputReg, outputProb);

    for (size_t i=0;i<2704;++i){
        printf("Output [%d]: %f\n", i, outputReg[i]);
        assert(equalFloatDefault(outputReg[i], expectedOutputReg0[i]));
    }

    for (size_t i=0;i<1352;++i){
        printf("Output [%d]: %f\n", i, outputProb[i]);
        assert(equalFloatDefault(outputProb[i], expectedOutputProb0[i]));
    }
//    free(outputProb);
//    free(outputReg);
}

void PnetTest_GetOutputRegSize0(){

}

void PnetTest_GetOutputProbSize0(){

}
