//
// Created by Michał on 15.11.2024.
//

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "mtcnn_test.h"
#include "mtcnn.h"
#include "cnn_test.h"
#include "mtcnn_test_data.h"
#include "cnn.h"

void MTCNNTest_GenerateBoundingBox0(){

}

void MTCNNTest_GenerateBoundingBox1(){

}

void MTCNNTest_GenerateBoundingBox2(){

}

void MTCNNTest_DetectFace0(){
    size_t maxBoxesCount = 5;
    float output[5*maxBoxesCount];
    int faces = MTCNN_DetectFace(3, 100, 100, detectFaceInput0, output);
    assert(faces == 1);
    float expectedOutput[] = {18.58723, 20.20803, 74.28621, 75.90701, 0.99956};
    for (size_t i=0;i<5;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloat(output[i], expectedOutput[i], 0.01f));
    }
}

void MTCNNTest_DetectFace1(){
    size_t maxBoxesCount = 5;
    float output[5*maxBoxesCount];
    int faces = MTCNN_DetectFace(3, 100, 100, detectFaceInput1, output);
    assert(faces == 1);
    float expectedOutput[] = {21.73954, 20.08783, 86.66113, 85.00943, 0.9992};
    for (size_t i=0;i<5;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloat(output[i], expectedOutput[i], 0.01f));
    }
}

void MTCNNTest_DetectFace2(){
    size_t maxBoxesCount = 5;
    float output[5*maxBoxesCount];
    int faces = MTCNN_DetectFace(3, 50, 50, detectFaceInput2, output);
    assert(faces == 1);
    float expectedOutput[] = {26.4718, 9.5409, 50.2393, 33.30841, 0.99982};
    for (size_t i=0;i<5;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloat(output[i], expectedOutput[i], 0.01f));
    }
}

void MTCNNTest_DetectFace3(){
    size_t maxBoxesCount = 5;
    float output[5*maxBoxesCount];
    int faces = MTCNN_DetectFace(3, 100, 100, detectFaceInput3, output);
    assert(faces == 3);
    float expectedOutput[] = {33.71898, 36.16474, 50.14524, 52.591, 0.99605, 7.04073, 15.27417, 38.05519, 46.28862, 0.99894, 62.07874, 27.25788, 91.55754, 56.73669, 0.99476};
    for (size_t i=0;i<15;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloat(output[i], expectedOutput[i], 0.01f));
    }
}

void MTCNNTest_BoxNms0(){
    float input[] = {51.0, 21.0, 70.0, 40.0, 0.9039, -0.03787, 0.01272, 0.00483, 0.22522, 54.0, 28.0, 73.0, 46.0, 0.95487, -0.05535, 0.12994, -0.09492, 0.26756};
    float output[18];
    size_t outputBoxLen = MTCNN_BoxNms(2, input, 0.5, output);
    assert(outputBoxLen == 2);
    float expectedOutput[] = {51.0, 21.0, 70.0, 40.0, 0.9039, -0.03787, 0.01272, 0.00483, 0.22522, 54.0, 28.0, 73.0, 46.0, 0.95487, -0.05535, 0.12994, -0.09492, 0.26756};
    for (size_t i=0;i<18;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void MTCNNTest_BoxNms1(){
    float input[] = {44.0, 25.0, 70.0, 51.0, 0.92829, 0.11208, 0.01449, 0.05255, 0.14467, 49.0, 25.0, 75.0, 51.0, 0.92458, 0.00878, 0.05492, -0.10532, 0.13889};
    float output[9];
    size_t outputBoxLen = MTCNN_BoxNms(2, input, 0.5, output);
    assert(outputBoxLen == 1);
    float expectedOutput[] = {44.0, 25.0, 70.0, 51.0, 0.92829, 0.11208, 0.01449, 0.05255, 0.14467};
    for (size_t i=0;i<9;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void MTCNNTest_BoxNms2(){
    float input[] = {43.0, 23.0, 79.0, 59.0, 0.92751, -0.02197, -0.00361, -0.12863, 0.13968, 23.0, 29.0, 59.0, 66.0, 0.99779, -0.00084, -0.08113, 0.01422, 0.14977, 43.0, 29.0, 79.0, 66.0, 0.97301, 0.02569, -0.03714, -0.10525, 0.06338, 43.0, 36.0, 79.0, 72.0, 0.95918, -0.01279, -0.11376, -0.08597, 0.05275};
    float output[2];
    size_t outputBoxLen = MTCNN_BoxNms(4, input, 0.5, output);
    assert(outputBoxLen == 2);
    float expectedOutput[] = {23.0, 29.0, 59.0, 66.0, 0.99779, -0.00084, -0.08113, 0.01422, 0.14977, 43.0, 29.0, 79.0, 66.0, 0.97301, 0.02569, -0.03714, -0.10525, 0.06338};
    for (size_t i=0;i<18;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void MTCNNTest_BoxNms3(){
    float input[] = {14.0, 23.0, 65.0, 74.0, 0.96042, 0.15332, -0.01909, 0.04666, 0.07502, 23.0, 23.0, 74.0, 74.0, 0.99966, 0.03534, -0.03008, -0.0268, 0.10018, 32.0, 23.0, 84.0, 74.0, 0.99519, -0.07423, -0.03817, -0.18115, 0.07282, 23.0, 32.0, 74.0, 84.0, 0.99754, 0.00332, -0.17798, -0.00412, 0.02096};
    float output[1];
    size_t outputBoxLen = MTCNN_BoxNms(4, input, 0.5, output);
    assert(outputBoxLen == 1);
    float expectedOutput[] = {23.0, 23.0, 74.0, 74.0, 0.99966, 0.03534, -0.03008, -0.0268, 0.10018};
    for (size_t i=0;i<9;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void MTCNNTest_BoxNms4(){
    float input[] = {19.0, 19.0, 92.0, 92.0, 0.99322, 0.09031, 0.02829, -0.19878, 0.00652};
    float output[1];
    size_t outputBoxLen = MTCNN_BoxNms(1, input, 0.5, output);
    assert(outputBoxLen == 1);
    float expectedOutput[] = {19.0, 19.0, 92.0, 92.0, 0.99322, 0.09031, 0.02829, -0.19878, 0.00652};
    for (size_t i=0;i<9;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void MTCNNTest_Rerec0(){
    float boxes[] = {24.80256, 21.46568, 72.63345, 79.10918, 0.99966, 25.59233, 21.06518, 77.4894, 92.47623, 0.99322, 43.92469, 27.62577, 75.2111, 68.34497, 0.97301, 52.94841, 30.33889, 71.19657, 50.81613, 0.95487, 46.91397, 25.37681, 71.36622, 54.7613, 0.92829, 50.28052, 21.24174, 70.09169, 44.27912, 0.9039};
    MTCNN_Rerec(6, boxes);
    float expectedOutput[] = {19.89625, 21.46569, 77.53975, 79.10919, 0.99966, 15.83533, 21.06518, 87.24639, 92.47623, 0.99322, 39.20829, 27.62577, 79.92749, 68.34497, 0.97301, 51.83387, 30.33889, 72.31112, 50.81613, 0.95487, 44.44785, 25.37681, 73.83234, 54.7613, 0.92829, 48.66741, 21.24174, 71.7048, 44.27912, 0.9039};
    for (size_t i=0;i<30;++i){
        printf("Output [%d]: %f\n", i, boxes[i]);
        assert(equalFloatDefault(boxes[i], expectedOutput[i]));
    }
}

void MTCNNTest_Pad0(){
    float boxes[] = {19.89625, 21.46569, 77.53975, 79.10919, 0.99966, 15.83533, 21.06518, 87.24639, 92.47623, 0.99322, 39.20829, 27.62577, 79.92749, 68.34497, 0.97301, 51.83387, 30.33889, 72.31112, 50.81613, 0.95487, 44.44785, 25.37681, 73.83234, 54.7613, 0.92829, 48.66741, 21.24174, 71.7048, 44.27912, 0.9039};
    int output[4*6];
    MTCNN_Pad(100, 100, 6, boxes, output);
    int expectedOutput[] = {19, 21, 77, 79, 15, 21, 87, 92, 39, 27, 79, 68, 51, 30, 72, 50, 44, 25, 73, 54, 48, 21, 71, 44};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %d\n", i, output[i]);
        assert(output[i] == expectedOutput[i]);
    }
}

void MTCNNTest_DetectFace4(){
    size_t maxBoxesCount = 5;
    float output[5*maxBoxesCount];
    uint8_t scaledInput[60*80*3];
    CNN_AdaptiveAveragePool_Uint8_Uint8(3, 120, 160, 60, 80, detectFaceInput4, scaledInput);
    int faces = MTCNN_DetectFace(3, 60, 80, scaledInput, output);
    assert(faces == 1);
    float expectedOutput[] = {26.566429, 10.839249, 63.781918, 48.054737, 0.991460};
    for (size_t i=0;i<faces*5;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}
