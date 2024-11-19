//
// Created by Michał on 15.11.2024.
//

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "mtcnn.h"
#include "cnn.h"
#include "dev_data.h"
#include "cnn_test.h"
#include "pnet.h"

void MTCNN_DetectFace(size_t inputChannels, size_t inputHeight, size_t inputWidth, const float* input, float* output){
    float thresholdPNet = 0.9;
    float thresholdRNet = 0.9;
    float minSize = 20;
    float factor = 0.709;
    float iouThresholdPNet0 = 0.5;
    float iouThresholdPNet1 = 0.5;
    float iouThresholdRNet = 0.05;

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

    size_t maxBoxesPerScale = 10; // should be given as argument
    size_t boxesMaxSize = scalesSize*9*maxBoxesPerScale; // or this
    float boxes[boxesMaxSize];
    size_t currentBoxesCount = 0;
    memset(boxes, 0, boxesMaxSize*sizeof(float));
    for (size_t i=0;i<scalesSize;++i){
        size_t outputHeight = inputHeight*scales[i]+1;
        size_t outputWidth = inputWidth*scales[i]+1;
        float* scaledOutput = malloc(inputChannels*outputHeight*outputWidth*sizeof(float));
        CNN_AdaptiveAveragePool(inputChannels, inputHeight, inputWidth, outputHeight, outputWidth, input, scaledOutput);
//        if (i == 0){
//            for (size_t j=0;j<11163;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput0[j]));
//            }
//        }
//        else if (i == 1){
//            for (size_t j=0;j<5547;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput1[j]));
//            }
//        }
//        else if (i == 2){
//            for (size_t j=0;j<2883;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput2[j]));
//            }
//        }
//        else if (i == 3){
//            for (size_t j=0;j<1452;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput3[j]));
//            }
//        }
//        else if (i == 4){
//            for (size_t j=0;j<768;++j){
//                printf("Output [%d]: %f\n", j, scaledOutput[j]);
//                assert(equalFloatDefault(scaledOutput[j], expectedOutput4[j]));
//            }
//        }
        for (size_t j=0;j<inputChannels*outputHeight*outputWidth;++j){
            scaledOutput[j] = (scaledOutput[j] - 127.5f) * 0.0078125f;
        }
        size_t outputRegSize = PNet_GetOutputRegSize(outputHeight, outputWidth);
        size_t outputProbSize = PNet_GetOutputProbSize(outputHeight, outputWidth);
        float outputReg[outputRegSize];
        float outputProb[outputProbSize];
        PNet_Model(inputChannels, outputHeight, outputWidth, scaledOutput, outputReg, outputProb);
        size_t regOutputHeight = PNet_GetOutputRegHeight(outputHeight);
        size_t regOutputWidth = PNet_GetOutputRegWidth(outputWidth);

        size_t outputBoxMaxSize = 9*regOutputHeight*regOutputWidth;
//        size_t outputBoxMaxSize = 9*maxBoxesPerScale;
        float outputBox[outputBoxMaxSize];
        memset(outputBox, 0, outputBoxMaxSize*sizeof(float));
        MTCNN_GenerateBoundingBox(regOutputHeight, regOutputWidth, outputReg, outputProb, scales[i], thresholdPNet, outputBox);

//        if (i == 0){
//            for (size_t j=0;j<18;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput01[j]));
//            }
//        }
//        else if (i == 1){
//            for (size_t j=0;j<18;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput11[j]));
//            }
//        }
//        else if (i == 2){
//            for (size_t j=0;j<36;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput21[j]));
//            }
//        }
//        else if (i == 3){
//            for (size_t j=0;j<36;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput31[j]));
//            }
//        }
//        else if (i == 4){
//            for (size_t j=0;j<9;++j){
//                printf("Output [%d]: %f\n", j, outputBox[j]);
//                assert(equalFloatDefault(outputBox[j], expectedOutput41[j]));
//            }
//        }
        size_t boxesLen = 0;
        for (size_t j=0; j<outputBoxMaxSize;j+=9){
            if (outputBox[j] == 0){
                break;
            }
            ++boxesLen;
        }
        currentBoxesCount += MTCNN_BoxNms(boxesLen, outputBox, iouThresholdPNet0, boxes+currentBoxesCount*9);
//        free(scaledOutput);
    }
    currentBoxesCount = MTCNN_BoxNms(currentBoxesCount, boxes, iouThresholdPNet1, boxes);
//    float expectedOutput[] = {51.0, 21.0, 70.0, 40.0, 0.9039, -0.03787, 0.01272, 0.00483, 0.22522, 54.0, 28.0, 73.0, 46.0, 0.95487, -0.05535, 0.12994, -0.09492, 0.26756, 44.0, 25.0, 70.0, 51.0, 0.92829, 0.11208, 0.01449, 0.05255, 0.14467, 43.0, 29.0, 79.0, 66.0, 0.97301, 0.02569, -0.03714, -0.10525, 0.06338, 23.0, 23.0, 74.0, 74.0, 0.99966, 0.03534, -0.03008, -0.0268, 0.10018, 19.0, 19.0, 92.0, 92.0, 0.99322, 0.09031, 0.02829, -0.19878, 0.00652};
//    for (size_t i=0;i<54;++i){
//        printf("Output [%d]: %f\n", i, boxes[i]);
//        assert(equalFloatDefault(boxes[i], expectedOutput[i]));
//    }

    size_t outputIndex = 0;
    for (size_t i=0;i<currentBoxesCount*9;i+=9) {
        float regW = boxes[i+2] - boxes[i];
        float regH = boxes[i+3] - boxes[i+1];
        boxes[outputIndex] = boxes[i] + boxes[i+5] * regW;
        boxes[outputIndex+1] = boxes[i+1] + boxes[i+6] * regH;
        boxes[outputIndex+2] = boxes[i+2] + boxes[i+7] * regW;
        boxes[outputIndex+3] = boxes[i+3] + boxes[i+8] * regH;
        boxes[outputIndex+4] = boxes[i+4];
        outputIndex += 5;
    }
//    float expectedOutput[] = {50.28052, 21.24174, 70.09169, 44.27912, 0.9039, 52.94841, 30.33889, 71.19657, 50.81613, 0.95487,46.91397, 25.37681, 71.36622, 54.7613, 0.92829, 43.92469, 27.62577, 75.2111, 68.34497, 0.97301, 24.80256, 21.46568, 72.63345, 79.10918, 0.99966, 25.59233, 21.06518, 77.4894, 92.47623, 0.99322};
//    for (size_t i=0;i<30;++i){
//        printf("Output [%d]: %f\n", i, boxes[i]);
//        assert(equalFloatDefault(boxes[i], expectedOutput[i]));
//    }
    MTCNN_Rerec(currentBoxesCount, boxes);
    int padArray[currentBoxesCount*4];
    MTCNN_Pad(inputHeight, inputWidth, currentBoxesCount, boxes, padArray);

    int x = 0;
}

void MTCNN_GenerateBoundingBox(size_t inputHeight, size_t inputWidth, const float* reg, const float* score, float scale, float threshold, float* output){
    int stride = 2;
    int cellSize = 12;

    size_t indexesSize = 10;
    int* indexes = calloc(indexesSize, indexesSize*sizeof(int)); // array must be created using malloc & size must be dynamically modified if exceeds max size
    size_t idx = 0;
    for (size_t i=0;i<inputHeight;++i){
        for (size_t j=0;j<inputWidth;++j){
            size_t index = inputWidth*inputHeight + i*inputWidth + j; // equivalent to: score[1][i][j]
            if (score[index] > threshold){
                if (idx >= indexesSize){
                    indexesSize += 10;
                    int* newIndexes = calloc(indexesSize, indexesSize*sizeof(int));
                    memcpy(newIndexes, indexes, (indexesSize-10)*sizeof(int));
                    free(indexes);
                    indexes = newIndexes;
                }
                indexes[idx] = index - inputWidth*inputHeight;
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
        output[9*i+4] = score[indexes[i] + inputHeight*inputWidth]; // equivalent to: score[1][index]
        output[9*i+5] = newReg[4*i];
        output[9*i+6] = newReg[4*i+1];
        output[9*i+7] = newReg[4*i+2];
        output[9*i+8] = newReg[4*i+3];
    }
    free(indexes);
}

int MTCNN_BoxNms(size_t boxesLen, const float* boxes, float iouThreshold, float* output){
    if (boxesLen == 1){
        memcpy(output, boxes, 9*sizeof(float));
        return 1;
    }
    int* indexes = calloc(boxesLen, boxesLen*sizeof(int));
    for (size_t i=0;i<boxesLen;++i){
        indexes[i] = -1;
    }
    for (size_t i=0;i<boxesLen*9;i+=9){
        if (indexes[i/9] == -2)
            continue;
        for (size_t j=i+9;j<boxesLen*9;j+=9){
            if (indexes[j/9] == -2)
                continue;

            float iou = CNN_Iou(boxes[i], boxes[i+1], boxes[i+2], boxes[i+3], boxes[j], boxes[j+1], boxes[j+2], boxes[j+3]);
            int boxI = i;
            int boxJ = j;

            if (iou > iouThreshold){
                if (boxes[i+4] >= boxes[j+4]){
                    boxJ = -2;
                }
                else{
                    boxI = -2;
                }
            }
            indexes[i/9] = boxI;
            indexes[j/9] = boxJ;
            if (boxI == -2)
                break;
        }
    }
    size_t outputIndex = 0;
    int outputBoxesLen = 0;
    for (size_t i=0;i<boxesLen;++i){
        int inputIndex = indexes[i];
        if (inputIndex < 0)
            continue;
        output[outputIndex] = boxes[inputIndex];
        output[outputIndex+1] = boxes[inputIndex+1];
        output[outputIndex+2] = boxes[inputIndex+2];
        output[outputIndex+3] = boxes[inputIndex+3];
        output[outputIndex+4] = boxes[inputIndex+4];
        output[outputIndex+5] = boxes[inputIndex+5];
        output[outputIndex+6] = boxes[inputIndex+6];
        output[outputIndex+7] = boxes[inputIndex+7];
        output[outputIndex+8] = boxes[inputIndex+8];
        outputIndex += 9;
        ++outputBoxesLen;
    }
    free(indexes);
    return outputBoxesLen;
}

void MTCNN_Rerec(size_t boxesLen, float* boxes){
    for (size_t i=0;i<boxesLen*5;i+=5){
        float w = boxes[i+2] - boxes[i];
        float h = boxes[i+3] - boxes[i+1];
        float l = fmaxf(h, w);
        boxes[i] = boxes[i] + w*0.5 - l*0.5;
        boxes[i+1] = boxes[i+1] + h*0.5 - l*0.5;
        boxes[i+2] = boxes[i] + l;
        boxes[i+3] = boxes[i+1] + l;
    }
}

void MTCNN_Pad(size_t inputHeight, size_t inputWidth, size_t boxesLen, float* boxes, int* output){
    size_t outputIndex = 0;
    for (size_t i=0;i<boxesLen*5;i+=5){
        output[outputIndex] = boxes[i] < 1 ? 1 : boxes[i];
        output[outputIndex+1] = boxes[i+1] < 1 ? 1 : boxes[i+1];
        output[outputIndex+2] = boxes[i+2] > inputWidth ? inputWidth : boxes[i+2];
        output[outputIndex+3] = boxes[i+3] > inputHeight ? inputHeight : boxes[i+3];
        outputIndex += 4;
    }
}
