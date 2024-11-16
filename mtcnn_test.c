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

}

void MTCNNTest_DetectFace2(){

}
