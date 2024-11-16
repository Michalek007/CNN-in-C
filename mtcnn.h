//
// Created by Michał on 15.11.2024.
//

#ifndef CNN_MTCNN_H
#define CNN_MTCNN_H

#include <stddef.h>

void MTCNN_DetectFace(size_t inputChannels, size_t inputHeight, size_t inputWidth, const float* input, float* output);

void MTCNN_GenerateBoundingBox(size_t inputHeight, size_t inputWidth, const float* reg, const float* score, float scale, float threshold, float* output);

#endif //CNN_MTCNN_H
