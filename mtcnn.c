//
// Created by Michał on 15.11.2024.
//

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "mtcnn.h"

void MTCNN_DetectFace(size_t inputChannels, size_t inputHeight, size_t inputWidth, const float* input, float* output){
    float thresholdPNet = 0.9;
    float thresholdRNet = 0.9;
    float minSize = 20;
    float factor = 0.709;

    // scale pyramid
    float m = 12.0 / minSize;
    float minL = fminf(inputHeight, inputWidth) * m;
    float scaleI = m;
    float* scales = malloc(10*sizeof(float));
    size_t scalesSize;
    for (scalesSize=0;minL>=12.0;minL*=factor){
        scales[scalesSize] = scaleI;
        scaleI *= factor;
        ++scalesSize;
    }

    for (size_t i=0;i<scalesSize;++i){

    }
}

void MTCNN_GenerateBoundingBox(size_t inputHeight, size_t inputWidth, const float* reg, const float* score, float scale, float threshold, float* output){
    int stride = 2;
    int cellSize = 12;

    size_t indexesSize = 10;
    int* indexes = calloc(indexesSize, indexesSize*sizeof(int)); // array must be created using malloc & size must be dynamically modified if exceeds max size
    size_t idx = 0;
    for (size_t i=0;i<inputHeight;++i){
        for (size_t j=0;j<inputWidth;++j){
            size_t index = i*inputWidth + j;
            if (score[index] > threshold){
                if (idx >= indexesSize){
                    indexesSize += 10;
                    int* newIndexes = calloc(indexesSize, indexesSize*sizeof(int));
                    memcpy(newIndexes, indexes, (indexesSize-10)*sizeof(int));
                    free(indexes);
                    indexes = newIndexes;
                }
                indexes[idx] = index;
                ++idx;
            }
        }
    }
    float newReg[idx*4]; // same as above
    for (size_t i=0;i<idx;++i) {
        int row = indexes[i] / inputWidth;
        int column = indexes[i] % inputWidth;
        for (size_t o=0;o<4;++o){
            size_t index = o*inputHeight*inputWidth + row*inputWidth + column;
            newReg[4*i+o] = reg[index];
        }
    }

    float q1[idx*2];
    float q2[idx*2];
    for (size_t i=0;i<idx;++i) {
        int row = indexes[i] / inputWidth;
        int column = indexes[i] % inputWidth;
        q1[i*2] = floorf((stride * column + 1) / scale);
        q1[i*2+1] = floorf((stride * row + 1) / scale);
        q2[i*2] = floorf((stride * column + cellSize) / scale);
        q2[i*2+1] = floorf((stride * row + cellSize) / scale);
    }

    for (size_t i=0;i<idx;++i) {
        output[9*i] = q1[i*2];
        output[9*i+1] = q1[i*2+1];
        output[9*i+2] = q2[i*2];
        output[9*i+3] = q2[i*2+1];
        output[9*i+4] = score[indexes[i]];
        output[9*i+5] = newReg[4*i];
        output[9*i+6] = newReg[4*i+1];
        output[9*i+7] = newReg[4*i+2];
        output[9*i+8] = newReg[4*i+3];
    }
    free(indexes);
}
