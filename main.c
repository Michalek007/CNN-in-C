#include <stdio.h>
#include "cnn.h"
#include "cnn_test.h"
#include "cnn_test_networks.h"
#include "mtcnn_test.h"
#include "pnet_test.h"
#include "rnet_test.h"
#include "lite_face_test.h"

int main() {
    printf("Hello, World!\n");

//    CNNTest_Dev();
//    return 0;

//    RNetTest_Model0();
//    PnetTest_Model0();
//    return 0;


//    MTCNNTest_GenerateBoundingBox0();
//    MTCNNTest_GenerateBoundingBox1();
//    MTCNNTest_GenerateBoundingBox2();
//    MTCNNTest_BoxNms0();
//    MTCNNTest_BoxNms1();
//    MTCNNTest_BoxNms2();
//    MTCNNTest_BoxNms3();
//    MTCNNTest_BoxNms4();
//    MTCNNTest_Rerec0();
//    MTCNNTest_Pad0();
//    MTCNNTest_DetectFace0();
//    MTCNNTest_DetectFace1();
//    MTCNNTest_DetectFace2();
//    MTCNNTest_DetectFace3();
//    MTCNNTest_DetectFace4();
//    return 0;
    LiteFaceTest_Model0();
    return 0;
    
    CNNTest_FcLayer0();
    CNNTest_FcLayer1();
    CNNTest_FcLayer2();
    CNNTest_ConvSymmetric0();
    CNNTest_ConvDev();
    CNNTest_ConvSymmetric1();
    CNNTest_ConvSymmetric2();
    CNNTest_ConvSymmetric3();
    CNNTest_ConvSymmetric4();
    CNNTest_ConvSymmetric5();
    CNNTest_ConvSymmetric6();
    CNNTest_ConvSymmetric7();
    CNNTest_ConvSymmetric8();
    CNNTest_ConvSymmetric9();
    CNNTest_Conv0();
    CNNTest_Conv1();
    CNNTest_Conv2();
    CNNTest_ReLU();
    CNNTest_MaxPoolBasic0();
    CNNTest_MaxPoolBasic1();
    CNNTest_MaxPoolBasic2();
    CNNTest_MaxPoolBasic3();
    CNNTest_MaxPoolSymmetric0();
    CNNTest_MaxPoolSymmetric1();
    CNNTest_MaxPool0();
    CNNTest_MaxPool1();
    CNNTest_MaxPool2();
    CNNTest_MaxPool3();
    CNNTest_MaxPool4();
    CNNTest_MaxPool5();
    CNNTest_PReLU0();
    CNNTest_PReLU1();
    CNNTest_Softmax0();
    CNNTest_Softmax2D0();
    CNNTest_Softmax2D1();
    CNNTest_Softmax2D2();
    CNNTest_Permute0();
    CNNTest_Permute1();
//    CNNTest_Permute2();
    CNNTest_Permute3();
    CNNTest_Permute4();
    CNNTest_BoxIou0();
    CNNTest_BoxIou1();
    CNNTest_BoxNms0();
    CNNTest_BoxNmsIdx0();
    CNNTest_AdaptiveAveragePool0();
    CNNTest_AdaptiveAveragePool1();
    CNNTest_AdaptiveAveragePool2();
    CNNTest_AdaptiveAveragePool3();
    CNNTest_AdaptiveAveragePool4();
    CNNTest_AveragePoolBasic0();
    CNNTest_LeakyReLU0();
    CNNTest_BatchNorm0();
    CNNTest_Normalize0();
    CNNTest_NormalizeLp0();

    CNNTest_FashionMnist();
    CNNTest_FashionMnist1();
    CNNTest_FashionMnist2();
    CNNTest_FashionMnist3();
    CNNTest_PNet0();
    CNNTest_RNet0();
    return 0;
}
