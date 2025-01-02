//
// Created by Michał on 02.01.2025.
//

#include "lite_face.h"
#include "lite_face_weights.h"
#include "cnn.h"


void LiteFace_Model(const float* input, float* outputEmbedding){
    float output1[43264];
    CNN_ConvLayer(3, 100, 100, 16, 3, 3, 2, 2, 3, 3, input, lite_face_weight0, lite_face_bias0, output1);
    CNN_BatchNorm(16, 52, 52, output1, lite_face_weight1, lite_face_bias1, lite_face_mean0, lite_face_var0);
    CNN_LeakyReLU(16, 52, 52, output1, 0.1f);
    float output2[10816];
    CNN_MaxPool(16, 52, 52, 2, 2, 2, 2, 0, 0, 0, output1, output2);
    float output3[21632];
    CNN_ConvLayer(16, 26, 26, 32, 3, 3, 1, 1, 1, 1, output2, lite_face_weight2, lite_face_bias2, output3);
    CNN_BatchNorm(32, 26, 26, output3, lite_face_weight3, lite_face_bias3, lite_face_mean1, lite_face_var1);
    CNN_LeakyReLU(32, 26, 26, output3, 0.1f);
    float output4[5408];
    CNN_MaxPool(32, 26, 26, 2, 2, 2, 2, 0, 0, 0, output3, output4);
    float output5[10816];
    CNN_ConvLayer(32, 13, 13, 64, 3, 3, 1, 1, 1, 1, output4, lite_face_weight4, lite_face_bias4, output5);
    CNN_BatchNorm(64, 13, 13, output5, lite_face_weight5, lite_face_bias5, lite_face_mean2, lite_face_var2);
    CNN_LeakyReLU(64, 13, 13, output5, 0.1f);
    float output6[2304];
    CNN_MaxPool(64, 13, 13, 2, 2, 2, 2, 0, 0, 0, output5, output6);
    float output7[4608];
    CNN_ConvLayer(64, 6, 6, 128, 3, 3, 1, 1, 1, 1, output6, lite_face_weight6, lite_face_bias6, output7);
    CNN_BatchNorm(128, 6, 6, output7, lite_face_weight7, lite_face_bias7, lite_face_mean3, lite_face_var3);
    CNN_LeakyReLU(128, 6, 6, output7, 0.1f);
    float output8[128];
    CNN_AveragePool(128, 6, 6, 6, 6, 1, 1, 0, 0, 0, output7, output8);
    float output9[512];
    CNN_FcLayer(128, 512, output8, lite_face_weight8, lite_face_bias8, output9);
    CNN_LeakyReLU(512, 1, 1, output9, 0.1f);
    CNN_FcLayer(512, 128, output9, lite_face_weight9, lite_face_bias9, outputEmbedding);
    CNN_NormalizeLp(128, 2, outputEmbedding);
}
