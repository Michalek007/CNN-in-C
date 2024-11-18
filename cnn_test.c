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
    CNN_ReLU(8, input);
    float expectedOutput[] = {0,  0.4078, 0.3754,  0.4056, 0,  0.0842, 0.3044,  0.4700};
    for (size_t i=0;i<8;++i){
        printf("Output: %f\n", input[i]);
        assert(equalFloatDefault(input[i], expectedOutput[i]));
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
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, 0, input, output);
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
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, 0, input, output);
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

    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const float weights[] = {0.25, 0.25};
    CNN_PReLU(inputChannels, inputHeight, inputWidth, input, weights);
    float expectedOutput[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, input[i]);
        assert(equalFloatDefault(input[i], expectedOutput[i]));
    }
}

void CNNTest_PReLU1(){
    size_t inputChannels = 2;
    size_t inputHeight = 2;
    size_t inputWidth = 6;

    float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    const float weights[] = {0.25, 0.25};
    CNN_PReLU(inputChannels, inputHeight, inputWidth, input, weights);
    float expectedOutput[] = {1.0, 2.0, 3.0, -1.0, -1.25, -1.5, -0.25, -0.5, -0.75, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -1.0, -1.25, -1.5};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, input[i]);
        assert(equalFloatDefault(input[i], expectedOutput[i]));
    }
}

void CNNTest_Softmax0(){
    size_t inputLen = 12;

    float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0};
    CNN_Softmax(inputLen, input);
//    CNN_Softmax2D(inputLen, 1, 1, 0, input);
    float expectedOutput[] = {0.004265888594090939, 0.011595887131989002, 0.03152088820934296, 2.874333040381316e-05, 1.0574080079095438e-05, 3.889986601279816e-06, 0.0005773252341896296, 0.0002123860758729279, 7.813247066223994e-05, 0.08568266034126282, 0.23290961980819702, 0.6331139802932739};
    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, input[i]);
        assert(equalFloatDefault(input[i], expectedOutput[i]));
    }
}

void CNNTest_Softmax2D0(){
    size_t inputChannels = 2;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t dim = 1;

    float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    CNN_Softmax2D(inputChannels, inputHeight, inputWidth, dim, input);
    float expectedOutput[] = {0.8807970285415649, 0.9820137619972229, 0.9975274205207825, 0.00033535013790242374, 4.539786823443137e-05, 6.144174221844878e-06, 0.11920291930437088, 0.01798621006309986, 0.0024726232513785362, 0.9996646642684937, 0.9999545812606812, 0.9999938011169434, 0.2689414322376251, 0.11920291930437088, 0.04742587357759476, 0.9996646642684937, 0.9999545812606812, 0.9999938011169434, 0.7310585975646973, 0.8807970285415649, 0.9525741338729858, 0.00033535013790242374, 4.539786823443137e-05, 6.144174221844878e-06};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, input[i]);
        assert(equalFloatDefault(input[i], expectedOutput[i]));
    }
}

void CNNTest_Softmax2D1(){
    size_t inputChannels = 2;
    size_t inputHeight = 2;
    size_t inputWidth = 6;
    size_t dim = 2;

    float input[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    CNN_Softmax2D(inputChannels, inputHeight, inputWidth, dim, input);
    float expectedOutput[] = {0.08994855731725693, 0.24450553953647614, 0.6646349430084229, 0.0006060686428099871, 0.000222960181417875, 8.202246681321412e-05, 0.0006060685846023262, 0.00022296016686595976, 8.202245953725651e-05, 0.08994854986667633, 0.24450550973415375, 0.6646348834037781, 0.0016408504452556372, 0.0016408504452556372, 0.0016408504452556372, 0.08958739042282104, 0.24352379143238068, 0.6619662642478943, 0.08994855731725693, 0.24450553953647614, 0.6646349430084229, 0.0006060686428099871, 0.000222960181417875, 8.202246681321412e-05};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, input[i]);
        assert(equalFloatDefault(input[i], expectedOutput[i]));
    }
}

void CNNTest_Softmax2D2(){
    size_t inputChannels = 3;
    size_t inputHeight = 4;
    size_t inputWidth = 4;
    size_t dim = 0;

    float input[] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    CNN_Softmax2D(inputChannels, inputHeight, inputWidth, dim, input);
    float expectedOutput[] = {0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408};
    for (size_t i=0;i<48;++i){
        printf("Output [%d]: %f\n", i, input[i]);
        assert(equalFloatDefault(input[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPool_2(){
    size_t inputChannels = 1;
    size_t inputHeight = 3;
    size_t inputWidth = 3;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    int strideH = 2;
    int strideW = 2;
    int paddingH = 0;
    int paddingW = 0;
    int ceilMode = 1;

    float input[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    float output[4];
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, ceilMode, input, output);
    float expectedOutput[] = {2.0, 3.0, 2.0, 3.0};
    for (size_t i=0;i<4;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPool_3(){
    size_t inputChannels = 1;
    size_t inputHeight = 3;
    size_t inputWidth = 3;
    size_t kernelHeight = 3;
    size_t kernelWidth = 3;
    int strideH = 3;
    int strideW = 3;
    int paddingH = 1;
    int paddingW = 1;
    int ceilMode = 1;

    float input[] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
    float output[4];
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, ceilMode, input, output);
    float expectedOutput[] = {2.0, 3.0, 2.0, 3.0};
    for (size_t i=0;i<4;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }

}

void CNNTest_Permute0(){
    float input[] = {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.};
    float expectedOutput[] = {1.,  7.,  4., 10.,  2.,  8.,  5., 11.,  3.,  9.,  6., 12.};
    float output[12];
    CNN_Permute(2, 2, 3, 2, 1, 0, input, output);

    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Permute1(){
    const float input[] = {0.07491911947727203, -0.573714554309845, -0.13826946914196014, -1.2384555339813232, 0.47688913345336914, -2.2013368606567383, -0.806811511516571, 2.151743173599243, -0.906222939491272, 0.9465761780738831, -0.4878195822238922, 0.23776130378246307, -0.3059251010417938, 1.4042332172393799, 1.1850194931030273, 0.4500694274902344, 1.2556042671203613, -0.7118940353393555, 0.24999912083148956, 0.14926737546920776, 0.09790348261594772, 0.1800849288702011, -0.6902297139167786, 1.0386160612106323, 1.8090888261795044, 0.36945268511772156, 0.5157124996185303, 0.2951139211654663, 0.34282368421554565, 0.20059479773044586, -1.8759745359420776, -1.416788101196289, -2.263699769973755, -0.1725410372018814, 1.352281093597412, -0.3209694027900696, -0.725623369216919, 0.7260327339172363, -0.22696086764335632, 0.2523721754550934, 1.099727988243103, -0.013196445070207119, 0.7871556282043457, -0.5624353885650635, 0.16245092451572418, -0.7626120448112488, -1.2851669788360596, 0.14673079550266266, -0.3650456666946411, -0.6272300481796265, -1.2253317832946777, -0.9350672364234924, 0.13199932873249054, 1.2244755029678345, 1.4259756803512573, 0.9910181164741516, -0.9950786828994751, 0.7977780699729919, -1.0137310028076172, -0.4637638032436371, -0.41683852672576904, 0.5414223670959473, 0.8733385801315308, -0.2469087541103363, -1.05905282497406, -2.022399663925171, 0.48879602551460266, -0.4685709476470947, -0.060121286660432816, 0.039991773664951324, 1.1326111555099487, 0.2125067114830017, 0.1967778205871582, -0.8220263123512268, -1.5607943534851074, -0.9146565794944763, 0.3190099895000458, -0.1047997921705246, -0.41474705934524536, 1.6853482723236084, 0.08826547116041183, 0.8508538603782654, 0.16959014534950256, 0.5014199614524841, -0.1480254828929901, 0.16122138500213623, 0.4583452045917511, 0.4070591926574707, 1.4826240539550781, -0.13384875655174255, 1.4832240343093872, -1.094735860824585, 0.21726691722869873, 1.309174656867981, 0.7584121227264404, -0.9412562847137451, -0.41310006380081177, 0.13099971413612366, -1.3644415140151978, 1.6134446859359741, -0.2425689995288849, 0.14106984436511993, -0.01902897283434868, -0.33171290159225464, 1.1081796884536743, -0.046644676476716995, 1.3527500629425049, -0.04567516967654228, -1.60672128200531, 0.0647188276052475, 0.6454123258590698, 0.03333680331707001, 0.8137394189834595, -1.448591709136963, -1.2735321521759033, -2.3891143798828125, -0.18596398830413818, 1.2372586727142334, -1.0956717729568481, 1.2941339015960693, -0.5841081142425537, 1.4186115264892578, 2.0603199005126953, -0.49825024604797363, -0.7715930342674255, 0.3952375650405884, 0.7454057335853577, 0.8690166473388672, -0.13453173637390137, 0.6125386357307434, 1.0211611986160278, -0.13143205642700195, 0.6172171235084534, 0.9663199782371521, -0.6778762936592102, -0.08583598583936691, 0.5761382579803467, -0.02600751630961895, -0.5502633452415466, 0.5300790667533875, -0.6941102147102356, -0.34841248393058777, -1.6315027475357056, -0.48554912209510803, -0.1368006467819214, 0.9771149754524231, 0.9309884905815125, -2.0534098148345947, 0.8773152232170105, -0.4883657395839691, -1.7625906467437744, -0.3720616400241852, 0.00019672677444759756, 0.269660085439682, 0.7324068546295166, -0.05411079153418541, -0.5974106192588806, -0.19692060351371765, -0.6203262209892273, -0.6794617176055908, 0.3738899528980255, 1.313607931137085, -0.846519947052002, -1.013859510421753, 0.5183437466621399, -1.524000883102417, -0.21042941510677338, 0.5542832016944885, 0.43197259306907654, -0.005256354808807373, 0.9546734690666199, 0.06129147484898567, -0.3100130259990692, 0.877453088760376, 1.127124547958374, -1.259102463722229, -1.0883680582046509, -1.244791030883789, -0.4751364290714264, 0.6061283349990845, 0.40607258677482605, 0.357437402009964, -0.0598842054605484, 1.1640738248825073, 0.9615376591682434, 2.288764238357544, -1.8076404333114624, -1.2032560110092163, -0.4938305914402008, 0.6050059795379639, 0.28053608536720276, -1.7126201391220093, 1.0068429708480835, -1.2892639636993408, -0.828646183013916, 0.5733478665351868, -0.39881983399391174, 0.19434121251106262, -0.1619756519794464, -0.9769981503486633, -1.4202741384506226, 0.33199313282966614, -0.4694424569606781, 0.032870709896087646, -0.8323355317115784, -0.6261593699455261, -0.20332764089107513, -2.032764434814453, -0.9337175488471985, 1.5792866945266724, -0.5342894196510315, 0.40996435284614563, -0.5806167721748352, -1.3150864839553833, -0.9891746640205383, 0.4744143784046173, -0.4365184009075165, 0.6378611922264099, -0.1140231341123581, -0.43145692348480225, -0.02673310600221157, 0.2865615785121918, -0.5840519070625305, 1.561942458152771, -0.679022490978241, 1.3799285888671875, -1.5215784311294556, 0.31602272391319275, -0.9734873175621033, 0.2373565435409546, -0.20389777421951294, -0.5579631328582764, -0.6180006861686707, -0.5842962861061096, -1.3879578113555908, -0.6433700323104858, -0.767336368560791, 0.9789437651634216, 0.40119850635528564, 1.293819785118103, 0.6467535495758057, 1.0601226091384888, 1.102365255355835, 0.43517711758613586, -0.4316968023777008, -0.09518375992774963, 0.666888415813446, -2.1067259311676025, 2.1237144470214844, -0.47431063652038574, 0.7115092873573303, -0.00232259021140635, 3.688671350479126, 0.7685257196426392, 0.9244097471237183, -0.6728811860084534, 0.13666932284832, -0.6138792634010315, -0.7789258360862732, 0.5298307538032532, 0.9410841464996338, -0.26364296674728394, -1.631320595741272, -0.2281312346458435, -0.20351678133010864, -1.700934886932373, 1.4458973407745361, 0.525955080986023, 0.76544588804245, -1.5790064334869385, -0.8890570998191833, -0.1775456964969635, -0.2348594069480896, 0.2691391110420227, 0.22020778059959412, -0.9900065064430237, 1.4102404117584229, 0.1984729915857315, 0.21345297992229462, -0.45221835374832153, -1.7562272548675537, -1.240940809249878, -0.08012203127145767, -0.1743582934141159, 0.18408045172691345, 1.1738080978393555, -1.0755255222320557, 1.2936749458312988, -0.5495040416717529, -0.5076972842216492, -0.02838996984064579, -0.2676224112510681, 1.6728523969650269, -1.2395190000534058, 0.7775407433509827, -0.8728329539299011, 0.28574690222740173, -1.6269980669021606, -1.5749119520187378, -1.564265251159668, -0.814448893070221, -2.2138020992279053, 1.520660638809204, 0.04751304164528847, 0.049844883382320404, -1.383689284324646, 0.17526741325855255, -0.09227866679430008, -0.40754371881484985, -0.9241226315498352, 1.3930108547210693, -1.986972689628601, 0.02101977914571762, 0.5683088302612305, 0.9833573698997498, 0.6360165476799011, -1.1131792068481445, 0.08455799520015717, 2.1261990070343018, 0.27322208881378174, 0.23064261674880981, -2.377373218536377, 0.4429630637168884, 0.20857560634613037, 0.25324907898902893, -0.15871913731098175, 0.6192786693572998, -0.07566341757774353, -0.1633690744638443, 0.10565747320652008, -1.333229422569275, 0.6719474196434021, 0.24972805380821228, -0.987452507019043, 0.12092873454093933, 0.8167706727981567, -0.8448460698127747, -0.7924440503120422, 0.5134503245353699, -0.6435934901237488, 1.0931018590927124, -0.05178174376487732, -0.47656428813934326, -0.04082018509507179, 0.21836735308170319, 1.2830942869186401, 0.9579315185546875, 0.8620895147323608, 1.1214361190795898, -0.2251983880996704, 0.2442939430475235, -0.07476481795310974, -2.834813356399536, -0.04484608396887779, 0.7386118173599243, 0.05742933228611946, 0.7689725756645203, -0.10029765218496323, -0.9348399639129639, 0.2758881449699402, -0.0850132629275322, 0.7421786785125732, -0.8580819368362427, 0.9952678084373474, 1.0420430898666382, 1.4465869665145874, -0.7036322355270386, 1.9408643245697021, -0.24415749311447144, -1.394842505455017, 0.688807487487793, -3.333200693130493, 0.0605938658118248, 0.9957181215286255, 0.1840677112340927, 0.11268848180770874, 0.015805985778570175, -0.13042744994163513, -0.41480374336242676, 0.2852029502391815, 0.7431244254112244, 0.07596447318792343, 0.7342748045921326, 1.3599159717559814, -0.8376988768577576, 0.8779494762420654, 0.23867538571357727, 0.960811972618103, 0.2805459499359131, 0.7167245745658875, 0.021732283756136894, -0.5465378165245056, -0.9927756190299988, -0.4521341919898987, -0.8115254044532776, 0.7834792733192444, -0.08311812579631805, 0.7826903462409973, 0.21772724390029907, 0.2292194366455078, 0.27459099888801575, -0.8749102354049683, 0.7150236964225769, 0.3916119635105133, 1.2432258129119873, 0.8236457705497742, 0.3865121901035309, 0.659575343132019, -1.070685863494873, -0.12804117798805237, -0.032271865755319595, -1.1852329969406128, -0.4196166694164276, -0.5783042907714844, 0.706031084060669, -0.2523065507411957, 0.045021262019872665, -0.4080619215965271, 0.11820866167545319, 0.0869029313325882, 0.5062050819396973, -0.6996136903762817, 0.7424921989440918, 0.26882749795913696, 0.5550364255905151, -1.451810598373413, -0.14643456041812897, 0.05419212579727173, 0.8096597790718079, 0.6714154481887817, 1.492905616760254, 0.15628927946090698};
    float expectedOutput[] = {0.07491911947727203, -0.1368006467819214, -0.5495040416717529, -0.3059251010417938, -0.5974106192588806, -0.814448893070221, 1.8090888261795044, 0.43197259306907654, 0.02101977914571762, -0.725623369216919, 0.40607258677482605, 0.25324907898902893, -0.3650456666946411, 1.0068429708480835, -0.8448460698127747, -0.41683852672576904, -0.8323355317115784, 1.1214361190795898, 0.1967778205871582, -0.4365184009075165, -0.0850132629275322, -0.1480254828929901, -0.9734873175621033, 0.0605938658118248, -0.41310006380081177, 0.6467535495758057, -0.8376988768577576, -1.60672128200531, 3.688671350479126, -0.08311812579631805, -0.5841081142425537, -0.20351678133010864, -1.070685863494873, 0.6172171235084534, 1.4102404117584229, 0.5062050819396973, -0.573714554309845, 0.9771149754524231, -0.5076972842216492, 1.4042332172393799, -0.19692060351371765, -2.2138020992279053, 0.36945268511772156, -0.005256354808807373, 0.5683088302612305, 0.7260327339172363, 0.357437402009964, -0.15871913731098175, -0.6272300481796265, -1.2892639636993408, -0.7924440503120422, 0.5414223670959473, -0.6261593699455261, -0.2251983880996704, -0.8220263123512268, 0.6378611922264099, 0.7421786785125732, 0.16122138500213623, 0.2373565435409546, 0.9957181215286255, 0.13099971413612366, 1.0601226091384888, 0.8779494762420654, 0.0647188276052475, 0.7685257196426392, 0.7826903462409973, 1.4186115264892578, -1.700934886932373, -0.12804117798805237, 0.9663199782371521, 0.1984729915857315, -0.6996136903762817, -0.13826946914196014, 0.9309884905815125, -0.02838996984064579, 1.1850194931030273, -0.6203262209892273, 1.520660638809204, 0.5157124996185303, 0.9546734690666199, 0.9833573698997498, -0.22696086764335632, -0.0598842054605484, 0.6192786693572998, -1.2253317832946777, -0.828646183013916, 0.5134503245353699, 0.8733385801315308, -0.20332764089107513, 0.2442939430475235, -1.5607943534851074, -0.1140231341123581, -0.8580819368362427, 0.4583452045917511, -0.20389777421951294, 0.1840677112340927, -1.3644415140151978, 1.102365255355835, 0.23867538571357727, 0.6454123258590698, 0.9244097471237183, 0.21772724390029907, 2.0603199005126953, 1.4458973407745361, -0.032271865755319595, -0.6778762936592102, 0.21345297992229462, 0.7424921989440918, -1.2384555339813232, -2.0534098148345947, -0.2676224112510681, 0.4500694274902344, -0.6794617176055908, 0.04751304164528847, 0.2951139211654663, 0.06129147484898567, 0.6360165476799011, 0.2523721754550934, 1.1640738248825073, -0.07566341757774353, -0.9350672364234924, 0.5733478665351868, -0.6435934901237488, -0.2469087541103363, -2.032764434814453, -0.07476481795310974, -0.9146565794944763, -0.43145692348480225, 0.9952678084373474, 0.4070591926574707, -0.5579631328582764, 0.11268848180770874, 1.6134446859359741, 0.43517711758613586, 0.960811972618103, 0.03333680331707001, -0.6728811860084534, 0.2292194366455078, -0.49825024604797363, 0.525955080986023, -1.1852329969406128, -0.08583598583936691, -0.45221835374832153, 0.26882749795913696, 0.47688913345336914, 0.8773152232170105, 1.6728523969650269, 1.2556042671203613, 0.3738899528980255, 0.049844883382320404, 0.34282368421554565, -0.3100130259990692, -1.1131792068481445, 1.099727988243103, 0.9615376591682434, -0.1633690744638443, 0.13199932873249054, -0.39881983399391174, 1.0931018590927124, -1.05905282497406, -0.9337175488471985, -2.834813356399536, 0.3190099895000458, -0.02673310600221157, 1.0420430898666382, 1.4826240539550781, -0.6180006861686707, 0.015805985778570175, -0.2425689995288849, -0.4316968023777008, 0.2805459499359131, 0.8137394189834595, 0.13666932284832, 0.27459099888801575, -0.7715930342674255, 0.76544588804245, -0.4196166694164276, 0.5761382579803467, -1.7562272548675537, 0.5550364255905151, -2.2013368606567383, -0.4883657395839691, -1.2395190000534058, -0.7118940353393555, 1.313607931137085, -1.383689284324646, 0.20059479773044586, 0.877453088760376, 0.08455799520015717, -0.013196445070207119, 2.288764238357544, 0.10565747320652008, 1.2244755029678345, 0.19434121251106262, -0.05178174376487732, -2.022399663925171, 1.5792866945266724, -0.04484608396887779, -0.1047997921705246, 0.2865615785121918, 1.4465869665145874, -0.13384875655174255, -0.5842962861061096, -0.13042744994163513, 0.14106984436511993, -0.09518375992774963, 0.7167245745658875, -1.448591709136963, -0.6138792634010315, -0.8749102354049683, 0.3952375650405884, -1.5790064334869385, -0.5783042907714844, -0.02600751630961895, -1.240940809249878, -1.451810598373413, -0.806811511516571, -1.7625906467437744, 0.7775407433509827, 0.24999912083148956, -0.846519947052002, 0.17526741325855255, -1.8759745359420776, 1.127124547958374, 2.1261990070343018, 0.7871556282043457, -1.8076404333114624, -1.333229422569275, 1.4259756803512573, -0.1619756519794464, -0.47656428813934326, 0.48879602551460266, -0.5342894196510315, 0.7386118173599243, -0.41474705934524536, -0.5840519070625305, -0.7036322355270386, 1.4832240343093872, -1.3879578113555908, -0.41480374336242676, -0.01902897283434868, 0.666888415813446, 0.021732283756136894, -1.2735321521759033, -0.7789258360862732, 0.7150236964225769, 0.7454057335853577, -0.8890570998191833, 0.706031084060669, -0.5502633452415466, -0.08012203127145767, -0.14643456041812897, 2.151743173599243, -0.3720616400241852, -0.8728329539299011, 0.14926737546920776, -1.013859510421753, -0.09227866679430008, -1.416788101196289, -1.259102463722229, 0.27322208881378174, -0.5624353885650635, -1.2032560110092163, 0.6719474196434021, 0.9910181164741516, -0.9769981503486633, -0.04082018509507179, -0.4685709476470947, 0.40996435284614563, 0.05742933228611946, 1.6853482723236084, 1.561942458152771, 1.9408643245697021, -1.094735860824585, -0.6433700323104858, 0.2852029502391815, -0.33171290159225464, -2.1067259311676025, -0.5465378165245056, -2.3891143798828125, 0.5298307538032532, 0.3916119635105133, 0.8690166473388672, -0.1775456964969635, -0.2523065507411957, 0.5300790667533875, -0.1743582934141159, 0.05419212579727173, -0.906222939491272, 0.00019672677444759756, 0.28574690222740173, 0.09790348261594772, 0.5183437466621399, -0.40754371881484985, -2.263699769973755, -1.0883680582046509, 0.23064261674880981, 0.16245092451572418, -0.4938305914402008, 0.24972805380821228, -0.9950786828994751, -1.4202741384506226, 0.21836735308170319, -0.060121286660432816, -0.5806167721748352, 0.7689725756645203, 0.08826547116041183, -0.679022490978241, -0.24415749311447144, 0.21726691722869873, -0.767336368560791, 0.7431244254112244, 1.1081796884536743, 2.1237144470214844, -0.9927756190299988, -0.18596398830413818, 0.9410841464996338, 1.2432258129119873, -0.13453173637390137, -0.2348594069480896, 0.045021262019872665, -0.6941102147102356, 0.18408045172691345, 0.8096597790718079, 0.9465761780738831, 0.269660085439682, -1.6269980669021606, 0.1800849288702011, -1.524000883102417, -0.9241226315498352, -0.1725410372018814, -1.244791030883789, -2.377373218536377, -0.7626120448112488, 0.6050059795379639, -0.987452507019043, 0.7977780699729919, 0.33199313282966614, 1.2830942869186401, 0.039991773664951324, -1.3150864839553833, -0.10029765218496323, 0.8508538603782654, 1.3799285888671875, -1.394842505455017, 1.309174656867981, 0.9789437651634216, 0.07596447318792343, -0.046644676476716995, -0.47431063652038574, -0.4521341919898987, 1.2372586727142334, -0.26364296674728394, 0.8236457705497742, 0.6125386357307434, 0.2691391110420227, -0.4080619215965271, -0.34841248393058777, 1.1738080978393555, 0.6714154481887817, -0.4878195822238922, 0.7324068546295166, -1.5749119520187378, -0.6902297139167786, -0.21042941510677338, 1.3930108547210693, 1.352281093597412, -0.4751364290714264, 0.4429630637168884, -1.2851669788360596, 0.28053608536720276, 0.12092873454093933, -1.0137310028076172, -0.4694424569606781, 0.9579315185546875, 1.1326111555099487, -0.9891746640205383, -0.9348399639129639, 0.16959014534950256, -1.5215784311294556, 0.688807487487793, 0.7584121227264404, 0.40119850635528564, 0.7342748045921326, 1.3527500629425049, 0.7115092873573303, -0.8115254044532776, -1.0956717729568481, -1.631320595741272, 0.3865121901035309, 1.0211611986160278, 0.22020778059959412, 0.11820866167545319, -1.6315027475357056, -1.0755255222320557, 1.492905616760254, 0.23776130378246307, -0.05411079153418541, -1.564265251159668, 1.0386160612106323, 0.5542832016944885, -1.986972689628601, -0.3209694027900696, 0.6061283349990845, 0.20857560634613037, 0.14673079550266266, -1.7126201391220093, 0.8167706727981567, -0.4637638032436371, 0.032870709896087646, 0.8620895147323608, 0.2125067114830017, 0.4744143784046173, 0.2758881449699402, 0.5014199614524841, 0.31602272391319275, -3.333200693130493, -0.9412562847137451, 1.293819785118103, 1.3599159717559814, -0.04567516967654228, -0.00232259021140635, 0.7834792733192444, 1.2941339015960693, -0.2281312346458435, 0.659575343132019, -0.13143205642700195, -0.9900065064430237, 0.0869029313325882, -0.48554912209510803, 1.2936749458312988, 0.15628927946090698};
    float output[3*12*12];
    CNN_Permute(3, 12, 12, 2, 1, 0, input, output);
    for (size_t i=0;i<432;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Permute2() {
    const float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float expectedOutput[] = {1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0};
    float output[12];
    CNN_Permute(2, 2, 3, 2, 0, 1, input, output);
    for (size_t i = 0; i < 12; ++i) {
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_MaxPool_4(){
    size_t inputChannels = 3;
    size_t inputHeight = 4;
    size_t inputWidth = 4;
    size_t kernelHeight = 3;
    size_t kernelWidth = 3;
    int strideH = 2;
    int strideW = 2;
    int paddingH = 0;
    int paddingW = 0;
    int ceilMode = 1;

    float input[] = {0.8279, 0.46341, 0.86545, 0.31142, 0.442, 0.38217, 0.54107, 0.35201, 0.27011, 0.47176, 0.75628, 0.96981, 0.82802, 0.34199, 0.39609, 0.14755, 0.64577, 0.10953, 0.71928, 0.4916, 0.3771, 0.92085, 0.07856, 0.55631, 0.35691, 0.28381, 0.19412, 0.88892, 0.31917, 0.02888, 0.38675, 0.16355, 0.93754, 0.44301, 0.33697, 0.4609, 0.84714, 0.94242, 0.33881, 0.97944, 0.03135, 0.52389, 0.68186, 0.21903, 0.42654, 0.30936, 0.71028, 0.58921};
    float output[12];
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, ceilMode, input, output);
    float expectedOutput[] = {0.86545, 0.96981, 0.82802, 0.96981, 0.92085, 0.88892, 0.38675, 0.88892, 0.94242, 0.97944, 0.71028, 0.71028};
    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }

}

void CNNTest_MaxPool_5(){
    size_t inputChannels = 3;
    size_t inputHeight = 4;
    size_t inputWidth = 4;
    size_t kernelHeight = 3;
    size_t kernelWidth = 3;
    int strideH = 2;
    int strideW = 2;
    int paddingH = 0;
    int paddingW = 0;
    int ceilMode = 1;

    float input[] = {-0.69233, -0.2751, -0.72322, -0.41535, -0.65958, -0.42752, -0.27232, -0.72395, -0.09608, -0.21748, -0.62269, -0.27097, -0.26292, -0.53008, -0.49703, -0.54208, -0.76701, -0.64043, -0.95705, -0.58469, -0.98878, -0.69187, -0.38132, -0.87176, -0.1723, -0.83375, -0.42588, -0.61377, -0.52989, -0.11865, -0.77723, -0.22211, -0.21569, -0.91871, -0.25167, -0.01942, -0.06396, -0.42864, -0.68572, -0.76806, -0.52915, -0.39198, -0.89214, -0.04035, -0.0562, -0.69629, -0.96913, -0.64165};
    float output[12];
    CNN_MaxPoolForward_(inputChannels, inputHeight, inputWidth, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, ceilMode, input, output);
    float expectedOutput[] = {-0.09608, -0.27097, -0.09608, -0.27097, -0.1723, -0.38132, -0.11865, -0.22211, -0.06396, -0.01942, -0.0562, -0.04035};
    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }

}

void CNNTest_Conv_2(){
    size_t inputChannels = 3;
    size_t inputHeight = 4;
    size_t inputWidth = 4;
    size_t kernelHeight = 2;
    size_t kernelWidth = 2;
    size_t outputChannels = 2;
    int strideH = 2;
    int strideW = 2;
    int paddingH = 2;
    int paddingW = 2;

    float input[] = {0.22592, 0.86936, 0.94874, 0.17696, 0.05077, 0.22579, 0.19037, 0.31321, 0.58919, 0.01971, 0.51823, 0.58233, 0.71819, 0.47539, 0.40683, 0.60826, 0.21407, 0.03722, 0.94867, 0.21533, 0.7977, 0.7491, 0.51569, 0.38928, 0.37191, 0.02819, 0.10069, 0.96713, 0.8202, 0.00876, 0.19568, 0.86679, 0.26567, 0.35126, 0.35533, 0.07283, 0.34393, 0.1154, 0.97988, 0.43608, 0.41682, 0.01703, 0.19041, 0.4395, 0.4202, 0.31251, 0.47009, 0.93325};
    float weights[] = {-0.02948, -0.09451, -0.2712, -0.15057, -0.00987, 0.22768, 0.2647, 0.26478, -0.15144, 0.25326, 0.20254, 0.15778, -0.10105, 0.25119, 0.01189, -0.24489, -0.04929, 0.23522, -0.15514, 0.18225, -0.02301, 0.21118, 0.17979, -0.27069};
    float biases[] = {-0.21425, 0.28474};
    float output[32];
    CNN_ConvLayerForward_(inputChannels, inputHeight, inputWidth, outputChannels, kernelHeight, kernelWidth, strideH, strideW, paddingH, paddingW, input, weights, biases, output);
    float expectedOutput[] = {-0.21425, -0.21425, -0.21425, -0.21425, -0.21425, 0.20161, 0.15341, -0.21425, -0.21425, -0.20206, 0.33895, -0.21425, -0.21425, -0.21425, -0.21425, -0.21425, 0.28474, 0.28474, 0.28474, 0.28474, 0.28474, 0.53523, 0.21905, 0.28474, 0.28474, -0.03011, 0.50501, 0.28474, 0.28474, 0.28474, 0.28474, 0.28474};
    for (size_t i=0;i<32;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_BoxIou0(){
    const float boxes[] = {50.0, 50.0, 70.0, 70.0, 10.0, 10.0, 20.0, 20.0};
    const float boxes2[] = {40.0, 40.0, 70.0, 70.0, 60.0, 60.0, 80.0, 80.0};
    float output[4];
    CNN_BoxIou(2, boxes, 2, boxes2, output);
    float expectedOutput[] = {0.44444, 0.14286, 0.0, 0.0};
    for (size_t i=0;i<4;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_BoxIou1(){

}

void CNNTest_BoxNms0(){
    const float boxes[] = {50.0, 50.0, 70.0, 70.0, 10.0, 10.0, 20.0, 20.0, 40.0, 40.0, 70.0, 70.0, 60.0, 60.0, 80.0, 80.0};
    const float scores[] = {0.8, 0.1, 0.1, 0.1};
    float output[4*4];
    CNN_BoxNms(4, boxes, scores, 0.3, output);
    float expectedOutput[] = {50.0, 50.0, 70.0, 70.0, 10.0, 10.0, 20.0, 20.0, 60.0, 60.0, 80.0, 80.0};
    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_BoxNms1(){

}

void CNNTest_AdaptiveAveragePool0(){
    const float input[] = {0.14626, 0.33309, 0.71805, 0.44304, 0.15392, 0.94494, 0.72817, 0.90589, 0.43838, 0.46262, 0.89121, 0.35884, 0.57872, 0.37437, 0.18344, 0.85242, 0.25616, 0.13777, 0.33076, 0.1753, 0.13143, 0.50179, 0.54406, 0.41088, 0.43432, 0.12907, 0.57009, 0.17968, 0.46361, 0.06912, 0.73081, 0.12536, 0.15852, 0.36419, 0.65315, 0.9416, 0.34143, 0.41013, 0.77936, 0.65738, 0.21917, 0.45195, 0.39818, 0.87565, 0.87901, 0.58721, 0.4861, 0.93734, 0.81799, 0.71435, 0.6289, 0.09449, 0.98052, 0.97699, 0.69078, 0.7858, 0.25245, 0.01832, 0.65309, 0.38637, 0.95683, 0.28967, 0.12495, 0.45701, 0.17585, 0.81373, 0.40859, 0.68186, 0.58697, 0.03725, 0.9798, 0.42442, 0.8287, 0.92644, 0.80173, 0.67809, 0.15785, 0.23674, 0.73872, 0.43395, 0.80142, 0.93781, 0.92829, 0.45335, 0.63043, 0.41324, 0.38884, 0.85763, 0.01587, 0.46235, 0.84202, 0.55725, 0.84175, 0.07142, 0.28385, 0.80045, 0.06054, 0.18386, 0.43477, 0.51051, 0.37647, 0.44544, 0.46649, 0.76088, 0.65421, 0.81305, 0.46989, 0.21701, 0.73374, 0.56111, 0.42676, 0.97956, 0.96481, 0.18111, 0.79197, 0.63178, 0.30084, 0.56917, 0.58357, 0.80287, 0.34977, 0.5804, 0.8448, 0.59306, 0.02806, 0.31365, 0.18192, 0.64346, 0.76173, 0.39295, 0.27967, 0.5609, 0.40367, 0.45334, 0.00092, 0.17799, 0.36959, 0.60497, 0.77771, 0.09845, 0.45999, 0.88768, 0.49057, 0.94418, 0.71232, 0.14168, 0.17189, 0.72712, 0.11088, 0.32792, 0.28442, 0.90199, 0.02326, 0.25558, 0.87876, 0.40289, 0.10476, 0.39489, 0.24899, 0.50886, 0.38373, 0.52255, 0.22856, 0.44034, 0.94828, 0.47675, 0.78044, 0.84571, 0.95965, 0.88406, 0.6404, 0.49007, 0.73495, 0.94575, 0.75314, 0.18345, 0.03175, 0.96182, 0.19257, 0.16869, 0.67512, 0.52177, 0.8271, 0.73812, 0.19038, 0.32174, 0.62933, 0.6558, 0.91913, 0.86958, 0.51956, 0.63665};
    float output[48];
    CNN_AdaptiveAveragePool(3, 8, 8, 4, 4, input, output);

    float expectedOutput[] = {0.34509, 0.60278, 0.51299, 0.66748, 0.23933, 0.31396, 0.29149, 0.45278, 0.29846, 0.71715, 0.55445, 0.71504, 0.45078, 0.44071, 0.801, 0.51463, 0.68618, 0.64257, 0.2547, 0.64422, 0.55436, 0.69523, 0.48921, 0.58269, 0.42791, 0.40805, 0.52919, 0.65842, 0.51902, 0.7154, 0.30293, 0.55296, 0.53231, 0.42918, 0.55117, 0.40342, 0.28321, 0.54517, 0.23461, 0.48606, 0.6875, 0.44984, 0.77643, 0.64069, 0.37642, 0.4116, 0.7464, 0.68036};
    for (size_t i=0;i<48;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_AdaptiveAveragePool1(){
    const float input[] = {0.80215, 0.01448, 0.17669, 0.73377, 0.04807, 0.50841, 0.76321, 0.15094, 0.36229, 0.70576, 0.58437, 0.56071, 0.21335, 0.70569, 0.04254, 0.32624, 0.27129, 0.60806, 0.74904, 0.91884, 0.1629, 0.52289, 0.82562, 0.44625, 0.16511, 0.07537, 0.80891, 0.24371, 0.04339, 0.13283, 0.97562, 0.21077, 0.74106, 0.17166, 0.02537, 0.66099, 0.27586, 0.76731, 0.67025, 0.90107, 0.63094, 0.81897, 0.528, 0.51862, 0.01935, 0.87684, 0.28732, 0.51732, 0.36778, 0.90461, 0.51697, 0.58926, 0.35232, 0.44454, 0.88912, 0.38638, 0.78135, 0.70414, 0.00707, 0.31032, 0.82735, 0.95497, 0.17653, 0.88429, 0.5986, 0.27654, 0.50407, 0.79425, 0.12702, 0.18243, 0.13775, 0.68884, 0.188, 0.50804, 0.99797, 0.52592, 0.16559, 0.63618, 0.29979, 0.50223, 0.8009, 0.17814, 0.60417, 0.09325, 0.12581, 0.61481, 0.26436, 0.77895, 0.92345, 0.63243, 0.33049, 0.79152, 0.41618, 0.46956, 0.39929, 0.06657, 0.16842, 0.72488, 0.02164, 0.7183, 0.55643, 0.13119, 0.46786, 0.16387, 0.53836, 0.65469, 0.42527, 0.57335, 0.10891, 0.21953, 0.36656, 0.12672, 0.72173, 0.73554, 0.10555, 0.26934, 0.55803, 0.33989, 0.9101, 0.42384, 0.62218, 0.40004, 0.96707, 0.88075, 0.55887, 0.66755, 0.61862, 0.02095, 0.6578, 0.98993, 0.25985, 0.29081, 0.50705, 0.66061, 0.2273, 0.86619, 0.19948, 0.89845, 0.27333, 0.01007, 0.39881, 0.49216, 0.41488, 0.83756, 0.8734, 0.51601, 0.08238, 0.41071, 0.44703, 0.30143, 0.69377, 0.22825, 0.3383, 0.2347, 0.62217, 0.84961, 0.79525, 0.61878, 0.90286, 0.33424, 0.50418, 0.63002, 0.33037, 0.86794, 0.60394, 0.9013, 0.48047, 0.08423, 0.64531, 0.92221, 0.59176, 0.89442, 0.85004, 0.56091, 0.80318, 0.4401, 0.54521, 0.24979, 0.71348, 0.83992, 0.27027, 0.96767, 0.62642, 0.56589, 0.77135, 0.02381, 0.43607, 0.88808, 0.85752, 0.62378, 0.61403, 0.38718, 0.83718, 0.11515, 0.56488, 0.83495, 0.17695, 0.99349, 0.08414, 0.47361, 0.85958, 0.28505, 0.05509, 0.27241, 0.21324, 0.43588, 0.98803, 0.26031, 0.01549, 0.82918, 0.61408, 0.3529, 0.07484, 0.61967, 0.10463, 0.91333, 0.79174, 0.01296, 0.21886, 0.55146, 0.55846, 0.18291, 0.38273, 0.58366, 0.25315, 0.22742, 0.85459, 0.30249, 0.81613, 0.49202, 0.89479, 0.66525, 0.24534, 0.06305, 0.56968, 0.17226, 0.23456, 0.09135, 0.44669, 0.19402};
    float output[48];
    CNN_AdaptiveAveragePool(3, 8, 10, 4, 4, input, output);

    float expectedOutput[] = {0.39196, 0.32002, 0.43803, 0.58249, 0.57314, 0.39584, 0.47203, 0.45976, 0.57274, 0.45866, 0.59223, 0.46854, 0.49557, 0.5589, 0.48376, 0.42365, 0.52023, 0.35138, 0.43633, 0.63328, 0.39508, 0.45548, 0.47009, 0.42927, 0.50783, 0.63361, 0.54521, 0.47509, 0.42769, 0.5535, 0.54546, 0.50251, 0.63347, 0.6694, 0.45018, 0.57582, 0.61713, 0.58014, 0.55889, 0.65344, 0.37359, 0.22331, 0.56704, 0.31471, 0.48825, 0.3496, 0.31378, 0.39045};
    for (size_t i=0;i<48;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

#pragma clang diagnostic pop