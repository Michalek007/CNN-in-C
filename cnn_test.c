#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-narrowing-conversions"
//
// Created by Michał on 15.10.2024.
//

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "cnn_test.h"
#include "cnn.h"

void CNNTest_Dev(){

}

bool equalFloat(float x, float y, float margin){
    if (fabsf(x - y) < margin)
        return true;
    return false;
}

bool equalFloatDefault(float x, float y){
    return equalFloat(x, y, 0.001f);
}

void CNNTest_FcLayer0(){
    size_t inputLen = 3;
    size_t outputLen = 2;
    float input[] = {1, 2, 3};

    float weights[] = {-0.2003, -0.0919,  0.1049, -0.5323, -0.5369,  0.1088};
    float biases[] = {0.0011, -0.3997};

    float output[outputLen];
    CNN_FcLayerForward(inputLen, outputLen, input, weights, biases, output);

    float expectedOutput[] = {-0.0684, -1.6795};

    for (size_t i=0;i<outputLen;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_FcLayer1(){
    size_t inputLen = 12;
    size_t outputLen = 8;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weights[] = {-0.15386559069156647, -0.047013670206069946, -0.24653998017311096, 0.2853236794471741, 0.2311784029006958, 0.2203679084777832, -0.11995509266853333, -0.1052200049161911, 0.06479033827781677, 0.26825833320617676, -0.024717718362808228, 0.037596315145492554, 0.2571210265159607, -0.11977535486221313, -0.25397437810897827, -0.18977609276771545, 0.03870341181755066, 0.041199326515197754, -0.28089720010757446, 0.16449686884880066, 0.10630455613136292, 0.2432098388671875, -0.18728362023830414, 0.19309145212173462, -0.06175830960273743, 0.16617482900619507, -0.2762075960636139, -0.06832855939865112, -0.10008285939693451, -0.015487372875213623, -0.01772528886795044, -0.14926041662693024, -0.05384151637554169, -0.23583726584911346, 0.14255139231681824, 0.07024112343788147, 0.22425949573516846, 0.056070804595947266, -0.11912316083908081, 0.1743098795413971, 0.2641991972923279, 0.05884560942649841, -0.0533718466758728, 0.12382543087005615, 0.2223767638206482, 0.21536922454833984, 0.14736688137054443, 0.287800133228302, 0.19475418329238892, -0.10210888087749481, -0.13115951418876648, -0.03175389766693115, 0.14787328243255615, -0.005057096481323242, -0.03434225916862488, 0.14842423796653748, -0.28165754675865173, -0.24936337769031525, -0.2577784061431885, -0.02993437647819519, 0.032215118408203125, -0.19328665733337402, 0.17879971861839294, 0.23820257186889648, -0.1990514099597931, 0.12727850675582886, -0.28434571623802185, 0.2649036645889282, 0.07585272192955017, 0.17377173900604248, 0.26147252321243286, -0.11298634111881256, 0.13652971386909485, -0.10287050902843475, -0.2712175250053406, -0.2040562927722931, -0.06177134811878204, 0.2832469940185547, 0.16985216736793518, 0.0240134596824646, 0.1275714635848999, 0.054313987493515015, 0.045977503061294556, -0.09940420091152191, 0.04419463872909546, -0.26088228821754456, -0.28620490431785583, -0.0846717357635498, 0.20292463898658752, -0.1188981682062149, -0.13337400555610657, 0.07092171907424927, -0.18199826776981354, -0.24571922421455383, -0.11206245422363281, 0.11599600315093994};
    float biases[] = {-0.2834208309650421, -0.209992915391922, 0.11198967695236206, 0.006838679313659668, 0.05985492467880249, -0.03515517711639404, 0.1877993941307068, 0.26682329177856445};
    float expectedOutput[] = {0.7860946655273438, 0.13487300276756287, -0.7200698256492615, 2.5249881744384766, -0.8420420289039612, 1.0255547761917114, 0.24522829055786133, -1.2462044954299927};
    float output[outputLen];
    CNN_FcLayerForward(inputLen, outputLen, input, weights, biases, output);

    for (size_t i=0;i<outputLen;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }

}

void CNNTest_ConvDev(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 1;

    float input [2][2] = {{1, 2}, {1, 2}};
    float weights [2][1][2][2] = {
            {{{-0.4731,  0.4078}, {0.3754,  0.4056}}},
            {{{-0.4612,  0.0842}, {0.3044,  0.4700}}}
    };
    float biases [2][1][1] = {{0.4796}, {0.1459}};
    float output [outputChannels][1][1];

    for (size_t o=0;o<outputChannels;++o){
        for (size_t i=0;i<(inputHeight-kernelHeight)/stride+1;i+=stride){
            for (size_t j=0;j<(inputWidth-kernelWidth)/stride+1;j+=stride){
                float outputValue = 0;
                for (size_t p =0;p<inputChannels;p++){
                    for (size_t k=0;k<kernelHeight;k++){
                        for (size_t l=0;l<kernelWidth;l++){
                            outputValue += input[i+k][j+l] * weights[o][p][k][l];
                        }
                    }
                    output[o][i][j] = outputValue + biases[o][i][j];
                }
            }
        }
    }

    float expectedOutput[] = {2.0088, 1.0974};

    for (size_t i=0;i<outputChannels;++i){
        printf("Output: %f\n", output[i][0][0]);
    }
}

void CNNTest_Conv1(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 1;
    int padding = 0;

    float input[] = {1.0, 2.0, 1.0, 2.0};
    float weights[] = {0.21530169248580933, -0.3739558458328247, 0.14181584119796753, -0.4979541301727295, 0.38280826807022095, -0.3231244683265686, -0.2002696990966797, -0.44069236516952515};
    float biases[] = {-0.07219243049621582, 0.49940162897109985};
    float expectedOutput[] = {-1.4588948488235474, -0.8456935286521912};
    float output[2];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    for (size_t i=0;i<2;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv0(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 1;
    float input [] = {1, 2, 1, 2};
    float weights [] = {-0.4731,  0.4078, 0.3754,  0.4056, -0.4612,  0.0842, 0.3044,  0.4700};
    float biases[] = {0.4796, 0.1459};
    float output [2];

    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, 0, input, weights, biases, output);

    float expectedOutput[] = {2.0088, 1.0974};
    for (size_t i=0;i<2;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv2(){
    size_t inputChannels = 3;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 1;
    int padding = 0;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weights[] = {-0.16665035486221313, 0.18172365427017212, 0.003009498119354248, -0.17354288697242737, -0.11587859690189362, -0.011833757162094116, -0.10530823469161987, 0.18277746438980103, 0.16802793741226196, -0.2470790445804596, -0.19278274476528168, 0.18605414032936096, 0.07907050848007202, 0.07021170854568481, 0.038380593061447144, 0.2479848861694336, 0.12452170252799988, 0.20817404985427856, -0.10956330597400665, -0.22134020924568176, -0.07077376544475555, 0.27043384313583374, 0.025753110647201538, 0.1865921914577484};
    float biases[] = {-0.27182209491729736, 0.24688738584518433};
    float expectedOutput[] = {-0.4452054500579834, 1.858389139175415};
    float output[2];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    for (size_t i=0;i<2;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv3(){
    size_t inputChannels = 3;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 1;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 1;
    int padding = 0;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weights[] = {0.23888885974884033, -0.39786297082901, -0.047030866146087646, -0.27882033586502075, 0.3372369408607483, -0.20797842741012573, -0.39400824904441833, -0.3677806854248047, -0.3841451108455658, -0.15446996688842773, 0.21588653326034546, 0.22090929746627808};
    float biases[] = {-0.28312644362449646, 0.25910741090774536};
    float expectedOutput[] = {-1.5233550071716309, -1.5233550071716309, -0.905842125415802, -0.905842125415802};
    float output[4];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    for (size_t i=0;i<4;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}


void CNNTest_ReLU(){
    float input [] = {-0.4731,  0.4078, 0.3754,  0.4056, -0.4612,  0.0842, 0.3044,  0.4700};
    float output [8];
    CNN_ReLU(8, input, output);
    float expectedOutput[] = {0,  0.4078, 0.3754,  0.4056, 0,  0.0842, 0.3044,  0.4700};
    for (size_t i=0;i<8;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}


void CNNTest_MaxPoolDefault0(){
    size_t inputChannels = 3;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernel = 2;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float expectedOutput[] = {2.0, 2.0, 2.0};
    float output[3];
    CNN_MaxPoolForwardDefault(inputChannels, inputHeight, inputWidth, kernel, input, output);
    for (size_t i=0;i<3;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPoolDefault1(){
    size_t inputChannels = 3;
    size_t inputHeight = 8;
    size_t inputWidth = 8;
    size_t kernel = 2;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float expectedOutput[] = {2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0};
    float output[48];
    CNN_MaxPoolForwardDefault(inputChannels, inputHeight, inputWidth, kernel, input, output);
    for (size_t i=0;i<48;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPoolDefault2(){
    size_t inputChannels = 3;
    size_t inputHeight = 4;
    size_t inputWidth = 4;
    size_t kernel = 2;

    float input[] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    float expectedOutput[] = {2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0};
    float output[12];
    CNN_MaxPoolForwardDefault(inputChannels, inputHeight, inputWidth, kernel, input, output);
    for (size_t i=0;i<12;++i){
        printf("Output: %f\n", output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }

}

void CNNTest_FcLayer2(){
    size_t inputLen = 12;
    size_t outputLen = 8;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weights[] = {0.16995835304260254, 0.22600078582763672, -0.2761087715625763, 0.011386513710021973, 0.13254234194755554, 0.16293412446975708, 0.08361351490020752, 0.20071491599082947, -0.1406496912240982, 0.04577279090881348, -0.2568320035934448, 0.04784095287322998, 0.2762386202812195, -0.11048518121242523, 0.06250417232513428, -0.0355963408946991, -0.22164452075958252, 0.14045783877372742, 0.06824713945388794, 0.13067972660064697, -0.2109280377626419, 0.23961293697357178, -0.0003300905227661133, 0.01168668270111084, 0.05759236216545105, 0.21103209257125854, -0.12336787581443787, 0.23207396268844604, -0.06109607219696045, -0.03237155079841614, -0.15839816629886627, -0.10125082731246948, -0.22038260102272034, 0.12531322240829468, 0.04893115162849426, -0.19492703676223755, -0.08529552817344666, -0.15485630929470062, -0.04665389657020569, -0.05766923725605011, -0.21354517340660095, -0.14076727628707886, 0.02796277403831482, -0.20059585571289062, -0.056377217173576355, 0.0911872386932373, -0.16691803932189941, -0.2764800786972046, 0.19043493270874023, -0.11486083269119263, 0.0975867211818695, -0.27269139885902405, -0.25550931692123413, -0.048815011978149414, -0.27739012241363525, 0.21343660354614258, -0.055867865681648254, 0.0758342444896698, -0.06157584488391876, 0.01748308539390564, -0.2159612774848938, 0.16303667426109314, 0.04684939980506897, 0.20647725462913513, 0.228692889213562, -0.1310933381319046, -0.08482837677001953, -0.08939646184444427, 0.0715867280960083, -0.044201016426086426, 0.06360968947410583, -0.15363213419914246, -0.15858615934848785, 0.11814367771148682, -0.006249725818634033, 0.22657793760299683, -0.15646637976169586, 0.27895408868789673, -0.21727895736694336, -0.074283167719841, -0.17173147201538086, 0.03363567590713501, -0.020136088132858276, -0.0446343719959259, 0.14371776580810547, -0.2637174427509308, -0.2169295847415924, -0.2063857614994049, 0.2197251319885254, 0.25254297256469727, 0.019192367792129517, 0.030066192150115967, -0.19475431740283966, 0.024700284004211426, -0.2228032648563385, 0.04736357927322388};
    float biases[] = {0.2728763818740845, 0.03939947485923767, 0.20954760909080505, -0.25628429651260376, -0.10322943329811096, -0.2817813456058502, -0.06255148351192474, 0.09646877646446228};
    float output[outputLen];
    CNN_FcLayerForward(inputLen, outputLen, input, weights, biases, output);
    float expectedOutput[] = {1.3747004270553589, 0.7661980390548706, 0.23256614804267883, -2.2754743099212646, -0.7247775793075562, -0.2694503366947174, 0.28378742933273315, -0.3862434923648834};
    for (size_t i=0;i<8;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv4(){
    size_t inputChannels = 3;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 1;
    int padding = 0;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weights[] = {0.018087685108184814, -0.2867906987667084, 0.14836326241493225, 0.1404966115951538, -0.08155675232410431, -0.09423647820949554, 0.09665343165397644, 0.1414070725440979, -0.017324596643447876, -0.14090172946453094, -0.002656877040863037, 0.27652156352996826, -0.2712455093860626, -0.19380900263786316, -0.17967677116394043, 0.04609683156013489, -0.18530984222888947, -0.25858041644096375, 0.2721595764160156, 0.060583293437957764, -0.12617437541484833, 0.25797683000564575, 0.011320412158966064, 0.22692596912384033};
    float biases[] = {0.050495415925979614, 0.1718587577342987};
    float output[2];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    float expectedOutput[] = {0.28505435585975647, -0.0286807119846344};
    for (size_t i=0;i<2;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPoolDefault3(){
    size_t inputChannels = 3;
    size_t inputHeight = 8;
    size_t inputWidth = 8;
    size_t kernel = 2;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float output[48];
    CNN_MaxPoolForwardDefault(inputChannels, inputHeight, inputWidth, kernel, input, output);
    float expectedOutput[] = {2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0};
    for (size_t i=0;i<48;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv5(){
    size_t inputChannels = 3;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 2;
    size_t kernelWidth = 1;
    size_t outputChannels = 2;
    int stride = 1;
    int padding = 0;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weights[] = {0.11188864707946777, -0.19589996337890625, 0.37177562713623047, -0.3102363348007202, 0.2910904884338379, 0.33867979049682617, -0.25160396099090576, 0.35944443941116333, -0.32669776678085327, 0.2901209592819214, -0.22926484048366547, -0.19617070257663727};
    float biases[] = {0.1990358829498291, 0.337726891040802};
    float output[4];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    float expectedOutput[] = {0.8063341379165649, 1.4136323928833008, -0.01644498109817505, -0.3706168532371521};
    for (size_t i=0;i<4;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv6(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 2;
    int padding = 0;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float weights[] = {-0.22702014446258545, -0.13850849866867065, -0.09126222133636475, 0.20569247007369995, 0.11927437782287598, 0.44718605279922485, 0.35928070545196533, -0.4848597049713135};
    float biases[] = {0.08726513385772705, 0.020229637622833252};
    float output[6];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    float expectedOutput[] = {-0.09664928913116455, -0.5988460779190063, -1.1010428667068481, 0.4234374165534973, 1.3052003383636475, 2.1869633197784424};
    for (size_t i=0;i<6;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv7(){
    size_t inputChannels = 3;
    size_t inputHeight = 2;
    size_t inputWidth = 2;
    size_t kernelHeight = 1;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 1;
    int padding = 0;

    float input[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0};
    float weights[] = {-0.1318436861038208, 0.1463903784751892, -0.2260064333677292, 0.16108381748199463, 0.14485830068588257, 0.06885740160942078, -0.22957946360111237, -0.07165250182151794, 0.3125830888748169, 0.3781844973564148, 0.23648786544799805, 0.25071847438812256};
    float biases[] = {0.07167991995811462, 0.09044229984283447};
    float output[4];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    float expectedOutput[] = {0.6113512516021729, 0.6113512516021729, 1.5244348049163818, 1.5244348049163818};
    for (size_t i=0;i<4;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv8(){
    size_t inputChannels = 1;
    size_t inputWidth = 6;
    size_t inputHeight = 2;
    size_t kernelWidth = 2;
    size_t kernelHeight = 2;
    size_t outputChannels = 2;
    int stride = 2;
    int padding = 0;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float weights[] = {-0.3524589538574219, 0.08577579259872437, 0.014021694660186768, -0.19818466901779175, 0.17860078811645508, 0.30151426792144775, -0.03831219673156738, 0.21769064664840698};
    float biases[] = {0.04682958126068115, 0.4851817488670349};
    float output[6];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    float expectedOutput[] = {-0.5164254307746887, -1.418117642402649, -2.319809913635254, 1.6638801097869873, 2.9828672409057617, 4.301854133605957};
    for (size_t i=0;i<6;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPool0(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 4;
    int stride = 2;
    int padding = 0;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output[2];
    CNN_MaxPoolForward(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, padding, input, output);
    float expectedOutput[] = {4.0, 6.0};
    for (size_t i=0;i<2;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv9(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int stride = 2;
    int padding = 1;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float weights[] = {0.07620233297348022, -0.32052844762802124, 0.39662057161331177, 0.36064183712005615, 0.012325644493103027, -0.4210308790206909, 0.0388561487197876, -0.24088072776794434};
    float biases[] = {0.0025184154510498047, 0.15941154956817627};
    float output[16];
    CNN_ConvLayerForward(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, stride, padding, input, weights, biases, output);
    float expectedOutput[] = {0.36316025257110596, 1.8776850700378418, 3.392209768295288, 2.38224196434021, -0.31801003217697144, -0.8066622614860535, -1.2953145503997803, 0.45973241329193115, -0.08146917819976807, -0.48551833629608154, -0.889567494392395, 0.39254844188690186, -0.26161932945251465, -1.0790297985076904, -1.8964403867721558, 0.23336541652679443};
    for (size_t i=0;i<16;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPool1(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 4;
    int stride = 2;
    int padding = 1;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output[6];
    CNN_MaxPoolForward(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, stride, padding, input, output);
    float expectedOutput[] = {3.0, 5.0, 6.0, 3.0, 5.0, 6.0};
    for (size_t i=0;i<6;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv_0(){
    size_t inputChannels = 1;
    size_t inputHeight = 4;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int strideH = 1;
    int strideW = 2;
    int paddingH = 0;
    int paddingW = 0;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float weights[] = {-0.2773746848106384, -0.38028454780578613, -0.321181058883667, 0.012752652168273926, 0.1606420874595642, -0.32405877113342285, 0.16822177171707153, 0.49534308910369873};
    float biases[] = {0.13865447044372559, -0.4579687714576721};
    float output[18];
    CNN_ConvLayerForward_(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, input, weights, biases, output);
    float expectedOutput[] = {-1.194965124130249, -3.1271402835845947, -5.0593156814575195, -1.194965124130249, -3.1271402835845947, -5.0593156814575195, -1.194965124130249, -3.1271402835845947, -5.0593156814575195, 0.21346372365951538, 1.2137601375579834, 2.2140564918518066, 0.21346372365951538, 1.2137601375579834, 2.2140564918518066, 0.21346372365951538, 1.2137601375579834, 2.2140564918518066};
    for (size_t i=0;i<18;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Conv_1(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int strideH = 2;
    int strideW = 2;
    int paddingH = 0;
    int paddingW = 1;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float weights[] = {-0.16223478317260742, -0.4977887272834778, 0.15677815675735474, 0.28064829111099243, 0.010788321495056152, -0.438529372215271, -0.30910801887512207, -0.3300057053565979};
    float biases[] = {-0.1693311333656311, 0.3180118203163147};
    float output[8];
    CNN_ConvLayerForward_(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, input, weights, biases, output);
    float expectedOutput[] = {-0.38647156953811646, -0.8316657543182373, -1.2768597602844238, -0.20207089185714722, -0.4505232572555542, -2.584232807159424, -4.717942714691162, -1.4719064235687256};
    for (size_t i=0;i<8;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPool_0(){
    size_t inputChannels = 1;
    size_t inputHeight = 4;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    int strideH = 1;
    int strideW = 2;
    int paddingH = 0;
    int paddingW = 0;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output[9];
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, input, output);
    float expectedOutput[] = {2.0, 4.0, 6.0, 2.0, 4.0, 6.0, 2.0, 4.0, 6.0};
    for (size_t i=0;i<9;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPool_1(){
    size_t inputChannels = 1;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    int strideH = 2;
    int strideW = 2;
    int paddingH = 0;
    int paddingW = 1;

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output[4];
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, input, output);
    float expectedOutput[] = {1.0, 3.0, 5.0, 6.0};
    for (size_t i=0;i<4;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_PReLU0(){
    size_t inputChannels = 2;
    size_t inputHeight = 2;
    size_t inputWidth = 6;

    const float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const float weights[] = {0.25, 0.25};
    float output[24];
    CNN_PReLU(inputChannels, inputHeight, inputWidth, input, weights, output);
    float expectedOutput[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_PReLU1(){
    size_t inputChannels = 2;
    size_t inputHeight = 2;
    size_t inputWidth = 6;

    const float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    const float weights[] = {0.25, 0.25};
    float output[24];
    CNN_PReLU(inputChannels, inputHeight, inputWidth, input, weights, output);
    float expectedOutput[] = {1.0, 2.0, 3.0, -1.0, -1.25, -1.5, -0.25, -0.5, -0.75, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -1.0, -1.25, -1.5};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Softmax0(){
    size_t inputLen = 12;

    const float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0};
    float output[12];
    CNN_Softmax(inputLen, input, output);
    float expectedOutput[] = {0.004265888594090939, 0.011595887131989002, 0.03152088820934296, 2.874333040381316e-05, 1.0574080079095438e-05, 3.889986601279816e-06, 0.0005773252341896296, 0.0002123860758729279, 7.813247066223994e-05, 0.08568266034126282, 0.23290961980819702, 0.6331139802932739};
    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Softmax2D0(){
    size_t inputChannels = 2;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t dim = 1;

    const float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    float output[24];
    CNN_Softmax2D(inputChannels, inputHeight, inputWidth, dim, input, output);
    float expectedOutput[] = {0.8807970285415649, 0.9820137619972229, 0.9975274205207825, 0.00033535013790242374, 4.539786823443137e-05, 6.144174221844878e-06, 0.11920291930437088, 0.01798621006309986, 0.0024726232513785362, 0.9996646642684937, 0.9999545812606812, 0.9999938011169434, 0.2689414322376251, 0.11920291930437088, 0.04742587357759476, 0.9996646642684937, 0.9999545812606812, 0.9999938011169434, 0.7310585975646973, 0.8807970285415649, 0.9525741338729858, 0.00033535013790242374, 4.539786823443137e-05, 6.144174221844878e-06};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Softmax2D1(){
    size_t inputChannels = 2;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t dim = 2;

    const float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    float output[24];
    CNN_Softmax2D(inputChannels, inputHeight, inputWidth, dim, input, output);
    float expectedOutput[] = {0.08994855731725693, 0.24450553953647614, 0.6646349430084229, 0.0006060686428099871, 0.000222960181417875, 8.202246681321412e-05, 0.0006060685846023262, 0.00022296016686595976, 8.202245953725651e-05, 0.08994854986667633, 0.24450550973415375, 0.6646348834037781, 0.0016408504452556372, 0.0016408504452556372, 0.0016408504452556372, 0.08958739042282104, 0.24352379143238068, 0.6619662642478943, 0.08994855731725693, 0.24450553953647614, 0.6646349430084229, 0.0006060686428099871, 0.000222960181417875, 8.202246681321412e-05};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

#pragma clang diagnostic pop