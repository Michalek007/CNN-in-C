//
// Created by Michał on 02.01.2025.
//

#ifndef CNN_LITE_FACE_H
#define CNN_LITE_FACE_H

#include <stddef.h>

void LiteFace_Model(size_t inputChannels, size_t inputHeight, size_t inputWidth, float* input, float* outputEmbedding);

#endif //CNN_LITE_FACE_H
