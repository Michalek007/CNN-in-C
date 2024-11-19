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

void MTCNNTest_GenerateBoundingBox0(){
    float output[18];
    float expectedOutput[] = {114.0, 38.0, 133.0, 56.0, 0.90236, 0.04277, -0.05898, -0.04223, 0.06484, 218.0, 401.0, 236.0, 419.0, 0.92792, 0.00324, -0.05295, -0.0464, 0.10173};
    MTCNN_GenerateBoundingBox(131, 104, reg0, probs0, 0.6, 0.9, output);
    for (size_t i=0;i<18;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void MTCNNTest_GenerateBoundingBox1(){

}

void MTCNNTest_GenerateBoundingBox2(){

}

void MTCNNTest_DetectFace0(){
    const float input[3*405*362];
    float output[4*10];
    MTCNN_DetectFace(3, 450, 362, input, output);
}

void MTCNNTest_DetectFace1(){
    float output[4*10];
    MTCNN_DetectFace(3, 100, 100, detectFaceInput0, output);
}

void MTCNNTest_DetectFace2(){

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
