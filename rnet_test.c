//
// Created by Michał on 20.11.2024.
//

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rnet_test.h"
#include "cnn_test.h"
#include "rnet_test_data.h"
#include "rnet.h"

void RNetTest_Model0(){
    float outputReg[4];
    float outputProb[2];
    RNet_Model(RNetModelInput0, outputReg, outputProb);

    float expectedOutput[] = {0.13138, -0.01568, -0.06179, 0.01421};
    for (size_t i=0;i<4;++i){
        printf("Output [%d]: %f\n", i, outputReg[i]);
        assert(equalFloatDefault(outputReg[i], expectedOutput[i]));
    }

    float expectedOutput2[] = {0.96175, 0.03825};
    for (size_t i=0;i<2;++i){
        printf("Output [%d]: %f\n", i, outputProb[i]);
        assert(equalFloatDefault(outputProb[i], expectedOutput2[i]));
    }
}
