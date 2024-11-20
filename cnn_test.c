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
    int outputLen = CNN_BoxNms(4, boxes, scores, 0.3, output);
    assert(outputLen == 3);
    float expectedOutput[] = {50.0, 50.0, 70.0, 70.0, 10.0, 10.0, 20.0, 20.0, 60.0, 60.0, 80.0, 80.0};
    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
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

void CNNTest_AdaptiveAveragePool2(){
    const float input[] = {0.41971, 0.76645, 0.86907, 0.45764, 0.01566, 0.07286, 0.49147, 0.2519, 0.02297, 0.41317, 0.66711, 0.04982, 0.37641, 0.48692, 0.10973, 0.84662, 0.06058, 0.81748, 0.07238, 0.52961, 0.57995, 0.63888, 0.78378, 0.19391, 0.88453, 0.0271, 0.20269, 0.32747, 0.48163, 0.17269, 0.60681, 0.53736, 0.93836, 0.65587, 0.38315, 0.23031, 0.90334, 0.32386, 0.75294, 0.3409, 0.45488, 0.56122, 0.25478, 0.02114, 0.67083, 0.24247, 0.16701, 0.32743, 0.45142, 0.93492, 0.04505, 0.94803, 0.13342, 0.3592, 0.76077, 0.93586, 0.14427, 0.8636, 0.421, 0.29661, 0.77694, 0.03079, 0.2549, 0.7895, 0.21914, 0.72129, 0.41819, 0.78166, 0.26143, 0.92783, 0.33289, 0.31079, 0.11087, 0.82201, 0.608, 0.29031, 0.95094, 0.95385, 0.15041, 0.4106, 0.04746, 0.02496, 0.69105, 0.75219, 0.44605, 0.99005, 0.2994, 0.72403, 0.59681, 0.25988, 0.51908, 0.74364, 0.16905, 0.07202, 0.63402, 0.98642, 0.83289, 0.98088, 0.67905, 0.69913, 0.44068, 0.59113, 0.18237, 0.48843, 0.09757, 0.26409, 0.59878, 0.86398, 0.291, 0.41101, 0.53465, 0.94776, 0.68761, 0.62862, 0.09387, 0.24178, 0.71497, 0.7216, 0.95605, 0.34184, 0.90211, 0.31661, 0.07228, 0.06904, 0.87724, 0.68633, 0.53539, 0.17847, 0.85139, 0.15942, 0.24716, 0.33642, 0.07571, 0.81356, 0.64053, 0.34179, 0.69857, 0.31509, 0.04411, 0.90815, 0.02947, 0.11897, 0.97778, 0.82783, 0.19556, 0.56836, 0.65186, 0.87021, 0.32009, 0.7305, 0.74655, 0.53008, 0.44138, 0.54795, 0.11952, 0.05447, 0.96748, 0.40957, 0.51635, 0.52019, 0.17961, 0.39159, 0.88532, 0.79997, 0.44055, 0.83908, 0.99516, 0.22918, 0.34718, 0.4405, 0.21295, 0.45459, 0.8327, 0.89591, 0.66658, 0.1151, 0.03641, 0.96343, 0.31927, 0.36504, 0.59836, 0.15559, 0.43365, 0.77661, 0.04159, 0.34006, 0.01612, 0.06207, 0.38087, 0.02001, 0.42031, 0.97014, 0.49071, 0.16748, 0.14317, 0.82111, 0.39616, 0.1027, 0.32707, 0.79662, 0.29888, 0.24417, 0.10427, 0.6691, 0.3451, 0.70652, 0.22134, 0.10266, 0.19987, 0.7747, 0.32132, 0.95622, 0.56018, 0.47543, 0.58192, 0.53594, 0.78545, 0.39991, 0.76648, 0.60951, 0.49713, 0.43351, 0.73285, 0.74007, 0.85643, 0.29137, 0.77877, 0.42358, 0.11316, 0.32239, 0.51204, 0.13683, 0.24711, 0.64959, 0.50233, 0.46143, 0.01022, 0.73003, 0.30049, 0.69939, 0.64256, 0.0155, 0.08042, 0.0553, 0.08736, 0.5979, 0.11189, 0.86663, 0.65177, 0.39252, 0.07359, 0.03462, 0.47872, 0.61364, 0.28913, 0.53388, 0.88051, 0.3762, 0.56969, 0.70521, 0.34127, 0.97096, 0.67329, 0.83301, 0.41569, 0.61658, 0.81925, 0.77672, 0.48264, 0.28732, 0.51198, 0.69087, 0.2787, 0.00841, 0.5846, 0.71963, 0.75513, 0.45269, 0.01361, 0.86396, 0.96925, 0.31328, 0.92523, 0.16286, 0.16065, 0.76707, 0.06295, 0.45076, 0.64996, 0.0701, 0.53982, 0.93465, 0.31714, 0.3899, 0.62406, 0.06729, 0.78958, 0.33014, 0.03384, 0.87631, 0.20298, 0.96123, 0.99619, 0.13794, 0.8184, 0.11825, 0.28513, 0.08584, 0.69307, 0.46495, 0.88861, 0.41397, 0.80116, 0.67586, 0.07346, 0.52798, 0.20019, 0.02918, 0.22149, 0.76631, 0.28533, 0.15825, 0.96421, 0.4957, 0.62694, 0.45374, 0.66849, 0.72098, 0.95546, 0.68804, 0.65808, 0.53458, 0.68494, 0.57417, 0.8475, 0.03854, 0.57974, 0.99458, 0.83075, 0.11901, 0.02984, 0.58968, 0.34293, 0.32629, 0.04942, 0.63431, 0.80431, 0.15588, 0.3489, 0.5119, 0.99891, 0.0972, 0.11608, 0.2237, 0.5794, 0.01014, 0.12162, 0.13207, 0.99649, 0.74198, 0.51154, 0.26406, 0.5045, 0.62829, 0.42762, 0.36504, 0.2552, 0.69226, 0.24712, 0.95239, 0.15572, 0.47492, 0.00123, 0.05767, 0.63761, 0.21472, 0.50891, 0.99888, 0.37768, 0.46022, 0.3813, 0.15786, 0.45222, 0.7718, 0.85413, 0.09282, 0.62832, 0.42397, 0.74198, 0.44417, 0.04965, 0.42469, 0.46798, 0.39189, 0.35515, 0.50105, 0.12418, 0.84706, 0.37538, 0.42054, 0.9003, 0.22334, 0.39578, 0.96762, 0.15781, 0.74908, 0.2881, 0.57086, 0.67662, 0.50395, 0.38275, 0.41894, 0.16029, 0.74873, 0.63937, 0.51925, 0.83691, 0.2419, 0.98624, 0.98855, 0.47276, 0.17053, 0.05999, 0.07403, 0.63382, 0.23751, 0.09918, 0.80222, 0.78585, 0.04312, 0.67445, 0.19133, 0.28755, 0.51482, 0.09869, 0.56612, 0.77604, 0.75782, 0.00611, 0.97867, 0.41596, 0.33642, 0.02262, 0.23336, 0.21649, 0.02796, 0.53503, 0.33498, 0.69241, 0.93894, 0.27243, 0.2489, 0.02895, 0.57525, 0.09195, 0.74235, 0.07383, 0.02698, 0.10742, 0.97885, 0.51715, 0.97015, 0.53307, 0.79623, 0.38561, 0.96842, 0.46773, 0.28876, 0.05643, 0.07473, 0.84343, 0.18936, 0.62182, 0.79613, 0.74289, 0.03513, 0.04281, 0.95593, 0.07945, 0.75331, 0.141, 0.73929, 0.47356, 0.40788, 0.45529, 0.29237, 0.66645, 0.16271, 0.40054, 0.06801, 0.53404, 0.99707, 0.29718, 0.20987, 0.76714, 0.32383, 0.08728, 0.431, 0.18631, 0.42266, 0.6293, 0.13496, 0.65512, 0.31263, 0.7761, 0.39994, 0.03632, 0.2368, 0.86041, 0.99415, 0.72467, 0.34017, 0.86608, 0.8933, 0.71667, 0.50414, 0.12061, 0.67478, 0.4294, 0.41683, 0.983, 0.24234, 0.66324, 0.53597, 0.02173, 0.35435, 0.63073, 0.92393, 0.2281, 0.90303, 0.13958, 0.2678, 0.00182, 0.65082, 0.08002, 0.87263, 0.3524, 0.44134, 0.28545, 0.61639, 0.46928, 0.8362, 0.28683, 0.22088, 0.31914, 0.58034, 0.872, 0.03052, 0.54892, 0.00888, 0.78939, 0.61562, 0.1746, 0.48558, 0.58912, 0.99853, 0.64517, 0.89552, 0.83853, 0.80058, 0.74441, 0.27593, 0.54877, 0.0958, 0.26064, 0.74008, 0.56409, 0.63781, 0.60273, 0.69238, 0.41249, 0.46826, 0.43172, 0.13293, 0.99103, 0.05065, 0.96936, 0.31981, 0.07085, 0.64558, 0.73974, 0.02471, 0.89607, 0.15523, 0.61938, 0.82876, 0.6209, 0.449, 0.38745, 0.06424, 0.31602, 0.55547, 0.20211, 0.96693, 0.20652, 0.46077, 0.70777, 0.43186, 0.70613, 0.5644, 0.90786, 0.18339, 0.34978, 0.62363, 0.63907, 0.83467, 0.91786, 0.16471, 0.23567, 0.75512, 0.80157, 0.70309, 0.05287, 0.56867, 0.99464, 0.34021, 0.34772, 0.14072, 0.90813, 0.66645, 0.16065, 0.936, 0.47169, 0.85942, 0.19079, 0.80577, 0.44683, 0.55211, 0.12069, 0.85824, 0.12884, 0.82617, 0.41936, 0.06684, 0.56093, 0.86987, 0.02499, 0.09034, 0.04851, 0.23116, 0.87119, 0.45051, 0.12034, 0.86214, 0.62526, 0.17083, 0.95279, 0.00422, 0.62155, 0.36245, 0.54747, 0.80292, 0.92659, 0.79094, 0.32633, 0.02034, 0.13405, 0.04988, 0.98123, 0.90159, 0.46168, 0.94299, 0.26387, 0.68287, 0.05351, 0.96329, 0.92358, 0.31397, 0.56189, 0.02443, 0.01667, 0.84526, 0.64138, 0.6416, 0.01467, 0.84031, 0.74636, 0.45764, 0.92929, 0.29842, 0.60811, 0.09938, 0.77369, 0.02959, 0.2567, 0.07671, 0.58569, 0.30002, 0.16612, 0.65894, 0.11082, 0.51134, 0.91326, 0.64176, 0.03131, 0.13253, 0.12678, 0.64083, 0.91447, 0.29563, 0.85929, 0.4779, 0.24713, 0.77374, 0.10167, 0.37475, 0.82068, 0.5102, 0.65481, 0.98289, 0.05547, 0.37835, 0.60407, 0.83161, 0.09068, 0.46884, 0.44703, 0.99463, 0.86338, 0.3064, 0.20878, 0.13948, 0.05524, 0.4746, 0.61655, 0.68296, 0.72686, 0.91229, 0.94958, 0.3027, 0.83178, 0.10646, 0.48825, 0.41378, 0.72631, 0.38068, 0.63337, 0.88084, 0.2808, 0.4949, 0.97522, 0.13323, 0.12398, 0.30493, 0.72963, 0.88278, 0.30954, 0.59329, 0.88203, 0.36218, 0.76817, 0.00395, 0.03803, 0.24124, 0.32621, 0.39459, 0.56015, 0.8338, 0.60471, 0.28218, 0.44794, 0.59326, 0.71869, 0.47563, 0.7809, 0.77444, 0.07368, 0.01429, 0.37385, 0.70819, 0.36489, 0.78948, 0.5618, 0.13351, 0.8849, 0.16063, 0.27253, 0.11623, 0.18451, 0.68068, 0.92625, 0.86756, 0.49902, 0.62932, 0.92495, 0.06432, 0.19685, 0.70569, 0.82963, 0.4237, 0.21704, 0.46514, 0.12626, 0.18899, 0.12872, 0.14951, 0.8668, 0.88906, 0.28644, 0.77939, 0.18703, 0.31891, 0.53332, 0.03134, 0.48086, 0.60573, 0.95484, 0.89811, 0.81512, 0.83228, 0.684, 0.11855, 0.46705, 0.12914, 0.57556, 0.41972, 0.88466, 0.48636, 0.78646, 0.71029, 0.06578, 0.2159, 0.25761, 0.45868, 0.81598, 0.50188, 0.44694, 0.33583, 0.09889, 0.88124, 0.60592, 0.40757, 0.6789, 0.69358, 0.13188, 0.89218, 0.63739, 0.40498, 0.53087, 0.65349, 0.18315, 0.62453, 0.30571, 0.00075, 0.08174, 0.2811, 0.23167, 0.49329, 0.37856, 0.23221, 0.7339, 0.96161, 0.33095, 0.45475, 0.67138, 0.37798, 0.84992, 0.38398, 0.02681, 0.60977, 0.91287, 0.13388, 0.06394, 0.63653, 0.76535, 0.15337, 0.44829, 0.28276, 0.95118, 0.46137, 0.9655, 0.06825, 0.26756, 0.60195, 0.3421, 0.44469, 0.17668, 0.40856, 0.28981, 0.86536, 0.64053, 0.10962, 0.56967, 0.22311, 0.019, 0.0542, 0.61398, 0.60964, 0.88911, 0.03624, 0.29402, 0.42054, 0.84567, 0.25341, 0.34923, 0.36711, 0.72294, 0.03119, 0.91765, 0.56676, 0.51428, 0.6174, 0.38391, 0.47482, 0.20654, 0.91928, 0.56137, 0.62405, 0.3261, 0.75879, 0.92154, 0.82539, 0.42892, 0.4412, 0.41149, 0.49072, 0.94317, 0.44968, 0.25755, 0.5261, 0.41721, 0.02596, 0.68577, 0.68774, 0.72983, 0.53431, 0.43442, 0.8295, 0.99675, 0.47262, 0.51065, 0.82735, 0.12713, 0.42607, 0.39489, 0.22943, 0.90687, 0.57121, 0.21299, 0.25275, 0.14995, 0.35407, 0.82996, 0.728, 0.45593, 0.33888, 0.5951, 0.74996, 0.95155, 0.25775, 0.88046, 0.24528, 0.29055, 0.44958, 0.63484, 0.0623, 0.74503, 0.99257, 0.23161, 0.50064, 0.7554, 0.52057, 0.95985, 0.97323, 0.56599, 0.31217, 0.95869, 0.44735, 0.19837, 0.64308, 0.05395, 0.18718, 0.0008, 0.00944, 0.72735, 0.56015, 0.71764, 0.31325, 0.20649, 0.49446, 0.29072, 0.23352, 0.4597, 0.29243, 0.30037, 0.0912, 0.49716, 0.57603, 0.41449, 0.90604, 0.56562, 0.05757, 0.1778, 0.36121, 0.22399, 0.08086, 0.50227, 0.4295, 0.36756, 0.31692, 0.32938, 0.02942, 0.85938, 0.36184, 0.14056, 0.35809, 0.52936, 0.1374, 0.98416, 0.07659, 0.06439, 0.80581, 0.25547, 0.6433, 0.4137, 0.6, 0.24638, 0.3584, 0.06709, 0.0339, 0.65987, 0.5224, 0.86181, 0.35059, 0.07024, 0.55524, 0.48173, 0.75071, 0.32043, 0.48786, 0.74408, 0.3594, 0.30332, 0.26785, 0.09114, 0.10553, 0.31864, 0.02823, 0.27703, 0.7903, 0.95108, 0.60304, 0.4516, 0.47819, 0.29831, 0.79328, 0.68933, 0.45074, 0.9791, 0.67928, 0.57296, 0.37148, 0.35501, 0.40534, 0.92332, 0.90955, 0.28754, 0.27981, 0.59214, 0.39412, 0.96912, 0.77, 0.34601, 0.71009, 0.27973, 0.12058, 0.43007, 0.67175, 0.42906, 0.22736, 0.33681, 0.79074, 0.39357, 0.83267, 0.77769, 0.60125, 0.26702, 0.27801, 0.97826, 0.14447, 0.89435, 0.12212, 0.82749, 0.41758, 0.54533, 0.50745, 0.03573, 0.11478, 0.93958, 0.72965, 0.78963, 0.56586, 0.16176, 0.46808, 0.64006, 0.18792, 0.06775, 0.73641, 0.37239, 0.10913, 0.82953, 0.74838, 0.9906, 0.19557, 0.48419, 0.77067, 0.27305, 0.02366, 0.64342, 0.0893, 0.57729, 0.10251, 0.42958, 0.25755, 0.26037, 0.03115, 0.85371, 0.08535, 0.00999, 0.11485, 0.82233, 0.12959, 0.83266, 0.29491, 0.41332, 0.51188, 0.82168, 0.58578, 0.53144, 0.25915, 0.17227, 0.40343, 0.1175, 0.31589, 0.20097, 0.90269, 0.81967, 0.68686, 0.995, 0.15855, 0.25439, 0.83907, 0.01484, 0.36147, 0.51496, 0.91745, 0.53456, 0.92194, 0.0838, 0.37381, 0.05217, 0.34912, 0.60413, 0.21711, 0.21814, 0.39287, 0.51673, 0.66054, 0.77869, 0.09999, 0.64299, 0.34027, 0.84072, 0.46031, 0.09997, 0.67054, 0.45962, 0.95167, 0.83668, 0.02023, 0.89821, 0.35541, 0.71139, 0.67586, 0.58978, 0.46336, 0.75869, 0.39565, 0.36382, 0.52565, 0.79948, 0.21617, 0.54025, 0.9122, 0.44617, 0.41036, 0.86576, 0.31425, 0.94022, 0.46819, 0.97827, 0.42925, 0.11625, 0.42698, 0.44679, 0.67298, 0.13916, 0.36066, 0.56228, 0.61639, 0.91083, 0.52599, 0.2636, 0.08937, 0.3095, 0.60505, 0.36822, 0.47231, 0.1442, 0.50195, 0.89415, 0.8897, 0.49875, 0.90545, 0.72311, 0.00645, 0.69675, 0.98508, 0.04009, 0.52999, 0.26429, 0.08148, 0.45892, 0.31161, 0.74746, 0.677, 0.59297, 0.76272, 0.90953, 0.24385, 0.45766, 0.95467, 0.55123, 0.39979, 0.84089, 0.95364, 0.7643, 0.73824, 0.49121, 0.41625, 0.41938, 0.94076, 0.04933, 0.48693, 0.56647, 0.046, 0.43852, 0.82779, 0.19456, 0.40946, 0.0555, 0.64641, 0.16584, 0.879, 0.11682, 0.62247, 0.20119, 0.1265, 0.91237, 0.61032, 0.51328, 0.61548, 0.78062, 0.79381, 0.42534, 0.37596, 0.60909, 0.40293, 0.96671, 0.84397, 0.04944, 0.03135, 0.28229, 0.07432, 0.55545, 0.85844, 0.79089, 0.23325, 0.66124, 0.3609, 0.31876, 0.79061, 0.73631, 0.92703, 0.78106, 0.96945, 0.53244, 0.25908, 0.31789, 0.49028, 0.11663, 0.32252, 0.38196, 0.53846, 0.69, 0.05573, 0.07306, 0.85195, 0.93559, 0.70086, 0.18834, 0.22183, 0.50333, 0.55466, 0.32856, 0.44554, 0.98246, 0.56764, 0.6469, 0.44614, 0.75951, 0.33266, 0.2666, 0.99217, 0.46665, 0.31196, 0.04721, 0.07833, 0.27237, 0.22246, 0.44228, 0.92275, 0.91528, 0.79627, 0.99873, 0.49631, 0.94894, 0.77363, 0.26253, 0.0501, 0.93201, 0.85233, 0.4088, 0.67128, 0.5512, 0.04067, 0.14607, 0.51197, 0.40463, 0.61743, 0.95595, 0.13112, 0.0837, 0.48267, 0.2754, 0.27735, 0.72963, 0.51451, 0.00674, 0.37669, 0.26564, 0.31133, 0.62343, 0.95785, 0.70991, 0.68741, 0.218, 0.62083, 0.12439, 0.14438, 0.96188, 0.73118, 0.29242, 0.57227, 0.27234, 0.259, 0.84348, 0.7371, 0.8176, 0.23985, 0.29967, 0.17103, 0.65053, 0.33262, 0.3548, 0.91768, 0.50341, 0.3889, 0.92943, 0.58121, 0.45228, 0.07514, 0.03534, 0.60336, 0.90015, 0.67436, 0.11303, 0.18163, 0.72398, 0.12467, 0.66885, 0.66303, 0.65654, 0.62991, 0.24104, 0.69145, 0.8181, 0.69845, 0.01852, 0.91998, 0.5637, 0.87964, 0.99652, 0.86464, 0.19513, 0.63823, 0.06083, 0.88732, 0.26225, 0.06707, 0.11456, 0.60779, 0.68274, 0.23914, 0.88133, 0.20975, 0.79339, 0.20604, 0.12165, 0.91135, 0.9352, 0.81765, 0.25965, 0.82514, 0.50597, 0.65136, 0.55757, 0.88822, 0.51207, 0.57363, 0.02996, 0.85712, 0.01505, 0.46933, 0.65102, 0.12338, 0.55857, 0.20706, 0.24041, 0.21171, 0.3926, 0.56996, 0.88861, 0.61095, 0.07745, 0.16635, 0.45288, 0.84172, 0.0339, 0.27603, 0.81247, 0.42284, 0.65922, 0.34052, 0.01744, 0.77814, 0.91172, 0.39143, 0.46996, 0.24141, 0.05366, 0.4732, 0.51927, 0.91599, 0.32929, 0.03457, 0.18581, 0.68922, 0.03289, 0.69714, 0.29826, 0.58449, 0.08799, 0.13915, 0.69037, 0.78564, 0.33864, 0.72685, 0.71972, 0.92987, 0.66514, 0.63528, 0.08935, 0.69018, 0.45542, 0.9163, 0.89897, 0.99146, 0.37994, 0.9181, 0.17446, 0.44528, 0.50722, 0.20414, 0.60385, 0.68928, 0.67852, 0.08145, 0.1014, 0.70253, 0.87904, 0.24246, 0.4806, 0.15165, 0.4261, 0.39122, 0.01827, 0.88083, 0.46568, 0.0062, 0.34338, 0.75959, 0.57898, 0.45241, 0.43256, 0.91567, 0.86851, 0.49858, 0.64306, 0.9041, 0.53672, 0.82738, 0.46578, 0.61685, 0.09268, 0.74464, 0.82879, 0.40717, 0.59088, 0.18131, 0.0518, 0.98637, 0.81161, 0.48275, 0.15137, 0.94576, 0.1689, 0.37749, 0.74541, 0.7001, 0.34856, 0.88372, 0.49917, 0.11372, 0.65044, 0.75417, 0.25739, 0.38752, 0.34183, 0.85809, 0.95324, 0.78873, 0.98558, 0.06344, 0.42315, 0.3955, 0.4843, 0.70958, 0.66345, 0.56272, 0.94479, 0.14641, 0.82128, 0.15994, 0.67254, 0.77448, 0.31172, 0.13313, 0.4359, 0.71038, 0.9639, 0.77881, 0.722, 0.27854, 0.36523, 0.09505, 0.38361, 0.55662, 0.25965, 0.14819, 0.23199, 0.81028, 0.52627, 0.00432, 0.95545, 0.12427, 0.32032, 0.03848, 0.72224, 0.62379, 0.46834, 0.13115, 0.56731, 0.0612, 0.65453, 0.73398, 0.81224, 0.82144, 0.61492, 0.50308, 0.15587, 0.15469, 0.48594, 0.53812, 0.63089, 0.93017, 0.54879, 0.91042, 0.64327, 0.10484, 0.55225, 0.23038, 0.69023, 0.30927, 0.19035, 0.50168, 0.54697, 0.30113, 0.85247, 0.47161, 0.68937, 0.29576, 0.57906, 0.01961, 0.2587, 0.58227, 0.02649, 0.74777, 0.85071, 0.63979, 0.7733, 0.8414, 0.77577, 0.05575, 0.45, 0.18196, 0.52165, 0.92844, 0.87288, 0.58322, 0.55261, 0.94063, 0.56553, 0.44335, 0.85607, 0.82528, 0.19767, 0.94283, 0.19103, 0.68064, 0.36056, 0.99235, 0.22832, 0.5668, 0.44791, 0.35331, 0.23357, 0.67893, 0.39751, 0.69598, 0.63212, 0.13113, 0.22305, 0.70039, 0.57014, 0.28439, 0.82742, 0.0902, 0.72842, 0.92782, 0.83348, 0.66379, 0.14049, 0.66007, 0.89933, 0.13454, 0.92557, 0.74191, 0.15427, 0.30867, 0.91916, 0.65462, 0.6326, 0.29858, 0.11505, 0.0613, 0.22622, 0.23104, 0.0828, 0.27978, 0.89523, 0.32559, 0.77024, 0.673, 0.23268, 0.04782, 0.67574, 0.34953, 0.18275, 0.28048, 0.3486, 0.60455, 0.44816, 0.55602, 0.7074, 0.47673, 0.73777, 0.16062, 0.65524, 0.66768, 0.96299, 0.21586, 0.95003, 0.68669, 0.56805, 0.24598, 0.68737, 0.53608, 0.71785, 0.13149, 0.96957, 0.86053, 0.56254, 0.98671, 0.87436, 0.43742, 0.57452, 0.24097, 0.85558, 0.65633, 0.57942, 0.81994, 0.90681, 0.68913, 0.81058, 0.63116, 0.07018, 0.99121, 0.20986, 0.83628, 0.40637, 0.42057, 0.52023, 0.2786, 0.65307, 0.37972, 0.25394, 0.99284, 0.68356, 0.93742, 0.94171, 0.01807, 0.05626, 0.0823, 0.10384, 0.3449, 0.34811, 0.71746, 0.7639, 0.43939, 0.43065, 0.48855, 0.68831, 0.35837, 0.32565, 0.1144, 0.06226, 0.56799, 0.50033, 0.14059, 0.46262, 0.25697, 0.25952, 0.60736, 0.48216, 0.86945, 0.24385, 0.18505, 0.65111, 0.67396, 0.99864, 0.55225, 0.27446, 0.37697, 0.50812, 0.39136, 0.51574, 0.57383, 0.44678, 0.91703, 0.02145, 0.086, 0.80635, 0.51896, 0.76261, 0.32012, 0.11343, 0.78906, 0.96593, 0.15907, 0.70025, 0.11538, 0.49246, 0.98356, 0.38288, 0.0755, 0.6116, 0.31492, 0.2106, 0.04915, 0.85108, 0.44498, 0.84688, 0.85906, 0.6216, 0.93824, 0.3666, 0.94989, 0.16081, 0.52317, 0.74836, 0.80479, 0.03663, 0.75784, 0.32257, 0.16727, 0.83209, 0.8544, 0.33251, 0.10298, 0.53037, 0.09717, 0.86166, 0.71003, 0.36627, 0.9312, 0.71136, 0.83111, 0.93333, 0.59064, 0.75063, 0.03803, 0.14455, 0.84645, 0.22731, 0.62729, 0.17053, 0.36378, 0.33759, 0.17312, 0.48833, 0.43597, 0.65113, 0.65195, 0.26618, 0.65805, 0.7927, 0.50713, 0.35836, 0.07161, 0.81242, 0.10895, 0.50534, 0.48354, 0.17189, 0.77855, 0.26998, 0.64878, 0.67612, 0.9886, 0.22663, 0.62765, 0.6537, 0.61879, 0.0445, 0.54632, 0.57168, 0.08762, 0.6204, 0.15679, 0.96865, 0.97832, 0.55462, 0.78791, 0.20703, 0.30019, 0.57214, 0.45335, 0.6919, 0.00797, 0.29653, 0.49903, 0.86909, 0.68984, 0.84023, 0.53221, 0.87892, 0.79802, 0.70608, 0.3372, 0.48596, 0.7866, 0.10285, 0.864, 0.83069, 0.47676, 0.78243, 0.89739, 0.5415, 0.09746, 0.27015, 0.83988, 0.08594, 0.91001, 0.42542, 0.89386, 0.90133, 0.64278, 0.49367, 0.99137, 0.38744, 0.09457, 0.25322, 0.31243, 0.80374, 0.81314, 0.90259, 0.5284, 0.66915, 0.91053, 0.61084, 0.56246, 0.1727, 0.28224, 0.80046, 0.87873, 0.7355, 0.47135, 0.68326, 0.98311, 0.68059, 0.02008, 0.38201, 0.16523, 0.99928, 0.62156, 0.23724, 0.7452, 0.50326, 0.35336, 0.36912, 0.00881, 0.78568, 0.38109, 0.20379, 0.44065, 0.48261, 0.09459, 0.5558, 0.43625, 0.86816, 0.92656, 0.82464, 0.45194, 0.81601, 0.81015, 0.89823, 0.08015, 0.7183, 0.35151, 0.93167, 0.49055, 0.38127, 0.40754, 0.65887, 0.42242, 0.01987, 0.10611, 0.06677, 0.40945, 0.03455, 0.82086, 0.0517, 0.79515, 0.48488, 0.4769, 0.86062, 0.35326, 0.56738, 0.06792, 0.14756, 0.01935, 0.78746, 0.5805, 0.92045, 0.9811, 0.10752, 0.75538, 0.14205, 0.22221, 0.72354, 0.09518, 0.38679, 0.20633, 0.29679, 0.32841, 0.11667, 0.25599, 0.49242, 0.37362, 0.40557, 0.71614, 0.48381, 0.16151, 0.63678, 0.45481, 0.02299, 0.61512, 0.3125, 0.95585, 0.25708, 0.97072, 0.6931, 0.58745, 0.87895, 0.99634, 0.03876, 0.98003, 0.73574, 0.84734, 0.44609, 0.63166, 0.23619, 0.04952, 0.25649, 0.15983, 0.09008, 0.29351, 0.42414, 0.1961, 0.84312, 0.74176, 0.77447, 0.44214, 0.12316, 0.36749, 0.50773, 0.7318, 0.77474, 0.2885, 0.31289, 0.36584, 0.40277, 0.00695, 0.7866, 0.88313, 0.30139, 0.20036, 0.80149, 0.86189, 0.46915, 0.21484, 0.25895, 0.55899, 0.72787, 0.94409, 0.0211, 0.069, 0.08122, 0.06173, 0.13612, 0.75419, 0.69239, 0.45032, 0.42282, 0.24408, 0.70351, 0.53333, 0.09951, 0.43722, 0.00654, 0.78822, 0.45086, 0.59614, 0.08684, 0.1496, 0.62403, 0.70169, 0.94608, 0.9351, 0.689, 0.9296, 0.27797, 0.72874, 0.92547, 0.6315, 0.07698, 0.96945, 0.2996, 0.88774, 0.81054, 0.19634, 0.70508, 0.60711, 0.62033, 0.53927, 0.34443, 0.71488, 0.7436, 0.52979, 0.30109, 0.65029, 0.93312, 0.43574, 0.25599, 0.51879, 0.99752, 0.0098, 0.26933, 0.17826, 0.99361, 0.96116, 0.56796, 0.08218, 0.25078, 0.96881, 0.0504, 0.48594, 0.66823, 0.63382, 0.98864, 0.77551, 0.10876, 0.99403, 0.8064, 0.82317, 0.30778, 0.6973, 0.23625, 0.29064, 0.38172, 0.70495, 0.76039, 0.56938, 0.87182, 0.88552, 0.29339, 0.50333, 0.21977, 0.14402, 0.0477, 0.52725, 0.11736, 0.95692, 0.97523, 0.55956, 0.56267, 0.75102, 0.41786, 0.51436, 0.96212, 0.0705, 0.84786, 0.68, 0.00435, 0.42514, 0.36456, 0.30419, 0.62694, 0.31851, 0.94444, 0.80732, 0.12555, 0.1438, 0.29725, 0.86028, 0.38555, 0.20666, 0.49436, 0.26163, 0.62952, 0.47724, 0.90793, 0.84272, 0.55303, 0.51533, 0.15214, 0.03517, 0.53496, 0.62577, 0.70003, 0.57064, 0.57954, 0.31442, 0.6564, 0.87317, 0.55229, 0.6176, 0.86437, 0.40321, 0.44654, 0.86101, 0.51504, 0.88129, 0.74033, 0.76495, 0.68343, 0.395, 0.98954, 0.98382, 0.15784, 0.04476, 0.23768, 0.69427, 0.58015, 0.77948, 0.07404, 0.54726, 0.0478, 0.75971, 0.35637, 0.62076, 0.05932, 0.85528, 0.19773, 0.65699, 0.79675, 0.8397, 0.54166, 0.41307, 0.14876, 0.26578, 0.22075, 0.91928, 0.23346, 0.22455, 0.49046, 0.92992, 0.51925, 0.04164, 0.13122, 0.72704, 0.97941, 0.98539, 0.34478, 0.63186, 0.20614, 0.69663, 0.13502, 0.30013, 0.31958, 0.7085, 0.13438, 0.25713, 0.04921, 0.16338, 0.86543, 0.82348, 0.51328, 0.10519, 0.43114, 0.27325, 0.61705, 0.98958, 0.26063, 0.98154, 0.62925, 0.428, 0.97973, 0.82079, 0.61121, 0.78363, 0.91685, 0.55101, 0.9794, 0.28367, 0.42099, 0.23565, 0.22463, 0.52085, 0.58859, 0.88236, 0.89572, 0.36926, 0.46164, 0.74325, 0.85506, 0.38151, 0.67426, 0.35063, 0.15993, 0.30243, 0.15206, 0.37932, 0.32758, 0.08266, 0.92748, 0.12104, 0.65589, 0.83356, 0.61236, 0.656, 0.44907, 0.78959, 0.19492, 0.15456, 0.50372, 0.57531, 0.34953, 0.8645, 0.07558, 0.94981, 0.69599, 0.36496, 0.86844, 0.16012, 0.23049, 0.44308, 0.24277, 0.26629, 0.51005, 0.4773, 0.33224, 0.12401, 0.44416, 0.85542, 0.57828, 0.96054, 0.94963, 0.10076, 0.61829, 0.22825, 0.70495, 0.26756, 0.26729, 0.66236, 0.75398, 0.51087, 0.83524, 0.63042, 0.43539, 0.15921, 0.21447, 0.88602, 0.5536, 0.81027, 0.62568, 0.11277, 0.14391, 0.67166, 0.40595, 0.03204, 0.97479, 0.55942, 0.79174, 0.37834, 0.59932, 0.97242, 0.57092, 0.62365, 0.83148, 0.06923, 0.35476, 0.91046, 0.94132, 0.8172, 0.34491, 0.59129, 0.41109, 0.20604, 0.75818, 0.42953, 0.64345, 0.94111, 0.03006, 0.41534, 0.7622, 0.19045, 0.61605, 0.50147, 0.41225, 0.34961, 0.08637, 0.33574, 0.90231, 0.68462, 0.60333, 0.67871, 0.07805, 0.38989, 0.88309, 0.74029, 0.17605, 0.63326, 0.4757, 0.15755, 0.05676, 0.49039, 0.14597, 0.90259, 0.09237, 0.92109, 0.31423, 0.353, 0.28824, 0.74743, 0.74181, 0.76056, 0.83823, 0.71981, 0.96666, 0.43577, 0.23434, 0.75965, 0.91939, 0.60081, 0.70056, 0.04108, 0.04899, 0.12765, 0.94375, 0.99956, 0.64525, 0.43777, 0.89058, 0.59759, 0.51953, 0.2311, 0.57256, 0.20824, 0.38051, 0.47144, 0.65012, 0.75362, 0.16746, 0.8031, 0.62247, 0.65775, 0.87046, 0.94664, 0.15882, 0.73181, 0.90263, 0.25488, 0.51571, 0.46149, 0.69101, 0.38205, 0.02294, 0.23138, 0.00089, 0.84658, 0.92951, 0.00615, 0.70994, 0.9109, 0.93907, 0.71427, 0.1411, 0.29544, 0.33605, 0.25828, 0.03481, 0.92396, 0.44708, 0.19509, 0.69205, 0.37145, 0.61367, 0.20791, 0.71207, 0.97124, 0.10686, 0.41748, 0.31577, 0.72092, 0.01627, 0.01051, 0.64885, 0.89681, 0.11105, 0.41515, 0.12894, 0.56529, 0.7534, 0.16991, 0.49915, 0.34392, 0.99502, 0.45288, 0.18901, 0.44809, 0.84004, 0.90904, 0.07038, 0.57323, 0.69722, 0.90411, 0.36954, 0.20235};
    float expectedOutput[] = {0.54481, 0.58897, 0.39832, 0.40348, 0.36587, 0.28844, 0.37241, 0.17798, 0.37277, 0.39592, 0.51948, 0.56507, 0.59257, 0.45953, 0.56719, 0.45249, 0.45873, 0.41611, 0.30939, 0.47433, 0.75937, 0.57627, 0.5785, 0.59941, 0.48742, 0.5697, 0.30364, 0.43782, 0.5572, 0.57311, 0.69113, 0.55025, 0.46911, 0.39557, 0.43809, 0.564, 0.40936, 0.45435, 0.53299, 0.53327, 0.52045, 0.64346, 0.55636, 0.52635, 0.39398, 0.55052, 0.54059, 0.52336, 0.48354, 0.30467, 0.29004, 0.59046, 0.57138, 0.4208, 0.35383, 0.35319, 0.46694, 0.50418, 0.49066, 0.70947, 0.57343, 0.59208, 0.45492, 0.42283, 0.42042, 0.4024, 0.40288, 0.50278, 0.52394, 0.69833, 0.62389, 0.69635, 0.60243, 0.46546, 0.44002, 0.52729, 0.78742, 0.48237, 0.43478, 0.41781, 0.44946, 0.34726, 0.52674, 0.46194, 0.26999, 0.36836, 0.4106, 0.51213, 0.29659, 0.31205, 0.48177, 0.40937, 0.54199, 0.30258, 0.4711, 0.47888, 0.46647, 0.37937, 0.56791, 0.64858, 0.7293, 0.56556, 0.50627, 0.50178, 0.48217, 0.49293, 0.50587, 0.58755, 0.58879, 0.58493, 0.50599, 0.66896, 0.55348, 0.54065, 0.46268, 0.45504, 0.65943, 0.43132, 0.45867, 0.33909, 0.31465, 0.50088, 0.45477, 0.46921, 0.56399, 0.45768, 0.48081, 0.34755, 0.42775, 0.67951, 0.51008, 0.42716, 0.5177, 0.42722, 0.37594, 0.56949, 0.60351, 0.30477, 0.39458, 0.46739, 0.52274, 0.64805, 0.4842, 0.44692, 0.54758, 0.48429, 0.53977, 0.62587, 0.40179, 0.30771, 0.34843, 0.30399, 0.43781, 0.46868, 0.54801, 0.41124, 0.40664, 0.44816, 0.35052, 0.29063, 0.56671, 0.62863, 0.52061, 0.54388, 0.63622, 0.45882, 0.43943, 0.59487, 0.39298, 0.48811, 0.41234, 0.36213, 0.56998, 0.40297, 0.41786, 0.42096, 0.3892, 0.53692, 0.33179, 0.40055, 0.57087, 0.34942, 0.29644, 0.24353, 0.37784, 0.56008, 0.44843, 0.35907, 0.47177, 0.4837, 0.41344, 0.76332, 0.70512, 0.58903, 0.56624, 0.52533, 0.45896, 0.43043, 0.57048, 0.53952, 0.54861, 0.36596, 0.47585, 0.54222, 0.55371, 0.3804, 0.4309, 0.48471, 0.52073, 0.35441, 0.34905, 0.61802, 0.50806, 0.4506, 0.50533, 0.44464, 0.40622, 0.62406, 0.40543, 0.58435, 0.77074, 0.61624, 0.65747, 0.68012, 0.39517, 0.42023, 0.54123, 0.42259, 0.60804, 0.48299, 0.48014, 0.51293, 0.50956, 0.52409, 0.42788, 0.5931, 0.50582, 0.38437, 0.51683, 0.48064, 0.24383, 0.31976, 0.5483, 0.52625, 0.54874, 0.57304, 0.57072, 0.56363, 0.6298, 0.57511, 0.54256, 0.46327, 0.49657, 0.75188, 0.66475, 0.49087, 0.46031, 0.51439, 0.52794, 0.53111, 0.23057, 0.38613, 0.44066, 0.52518, 0.68568, 0.59788, 0.50468, 0.47578, 0.36525, 0.37571, 0.42694, 0.29444, 0.4264, 0.50872, 0.45128, 0.4868, 0.42921, 0.56061, 0.57305, 0.43587, 0.29617, 0.32585, 0.51328, 0.73916, 0.69533, 0.4861, 0.48, 0.51926, 0.50172, 0.48344, 0.58265, 0.39406, 0.29544, 0.47321, 0.61048, 0.46179, 0.49577, 0.70476, 0.6013, 0.56714, 0.53967, 0.53292, 0.60547, 0.50987, 0.43305, 0.25261, 0.30535, 0.45357, 0.54786, 0.4868, 0.55036, 0.43848, 0.40959, 0.65679, 0.6548, 0.571, 0.54856, 0.19585, 0.4396, 0.54116, 0.52552, 0.61307, 0.55074, 0.30677, 0.45732, 0.52616, 0.38576, 0.47995, 0.74826, 0.50203, 0.54131, 0.33566, 0.5354, 0.60865, 0.43115, 0.38056, 0.40844, 0.22222, 0.4007, 0.46255, 0.5238, 0.65882, 0.46765, 0.48198, 0.55874, 0.52683, 0.40895, 0.65987, 0.48876, 0.58295, 0.56103, 0.58832, 0.46861, 0.5112, 0.29691, 0.45028, 0.55641, 0.49102, 0.54187, 0.54572, 0.41877, 0.34969, 0.22563, 0.46108, 0.54558, 0.41347, 0.30696, 0.44758, 0.35096, 0.43453, 0.5853, 0.56303, 0.37114, 0.37268, 0.3967, 0.4117, 0.60335, 0.44825, 0.4539, 0.63542, 0.63526, 0.51615, 0.50271, 0.29137, 0.3477, 0.381, 0.33671, 0.47824, 0.56312, 0.42887, 0.50428, 0.39846, 0.33082, 0.19142, 0.26946, 0.32395, 0.65076, 0.54433, 0.3549, 0.37264, 0.51643, 0.56975, 0.58572, 0.48019, 0.47967, 0.41847, 0.51959, 0.41673, 0.46329, 0.42768, 0.45156, 0.39953, 0.35952, 0.44094, 0.22197, 0.36585, 0.59442, 0.53639, 0.48827, 0.53925, 0.48179, 0.45036, 0.43822, 0.37494, 0.37271, 0.39106, 0.50691, 0.65391, 0.66718, 0.43821, 0.51775, 0.53172, 0.53733, 0.43292, 0.56934, 0.44265, 0.25231, 0.4376, 0.4312, 0.37626, 0.4598, 0.51692, 0.49731, 0.42241, 0.67203, 0.3971, 0.44165, 0.50603, 0.55955, 0.56713, 0.61886, 0.50862, 0.40968, 0.46807, 0.47281, 0.59807, 0.71731, 0.6943, 0.56905, 0.64352, 0.40721, 0.49334, 0.45811, 0.45147, 0.37338, 0.53468, 0.42837, 0.34867, 0.37054, 0.3708, 0.54221, 0.67081, 0.69614, 0.68785, 0.47278, 0.67346, 0.62813, 0.42192, 0.19463, 0.40501, 0.70959, 0.65254, 0.52858, 0.34529, 0.63487, 0.65359, 0.81198, 0.70286, 0.4516, 0.37527, 0.44982, 0.45333, 0.39726, 0.15332, 0.55358, 0.56012, 0.28151, 0.54041, 0.59113, 0.47915, 0.48892, 0.48075, 0.56831, 0.50316, 0.54025, 0.40123, 0.45302, 0.50903, 0.34961, 0.41829, 0.55114, 0.6969, 0.65683, 0.47124, 0.44921, 0.51129, 0.56369, 0.67984, 0.38097, 0.39986, 0.49114, 0.47635, 0.43221, 0.43133, 0.44088, 0.51628, 0.49306, 0.5426, 0.67289, 0.51425, 0.46853, 0.43853, 0.21824, 0.37724, 0.5511, 0.52275, 0.42322, 0.57584, 0.49476, 0.56358, 0.45605, 0.42736, 0.65097, 0.63981, 0.69443, 0.53868, 0.31718, 0.3171, 0.60928, 0.51831, 0.45408, 0.42608, 0.4857, 0.57507, 0.43493, 0.58276, 0.41388, 0.34206, 0.5892, 0.38288, 0.23393, 0.42381, 0.52298, 0.29581, 0.31291, 0.58276, 0.53808, 0.46107, 0.56053, 0.5775, 0.60905, 0.46677, 0.3488, 0.57833, 0.63862, 0.59263, 0.45703, 0.53992, 0.51847, 0.60696, 0.71648, 0.62357, 0.54443, 0.46301, 0.44221, 0.5026, 0.59484, 0.30802, 0.50096, 0.52901, 0.43236, 0.37223, 0.47204, 0.59162, 0.40769, 0.40562, 0.53781, 0.47862, 0.57939, 0.44334, 0.55004, 0.71268, 0.65098, 0.37761, 0.35643, 0.48582, 0.4869, 0.33773, 0.64542, 0.44274, 0.3724, 0.40081, 0.61416, 0.62413, 0.55417, 0.46731, 0.56586, 0.59679, 0.50743, 0.5316, 0.41587, 0.49437, 0.54593, 0.52411, 0.56875, 0.4604, 0.39112, 0.3187, 0.44724, 0.48682, 0.44143, 0.56194, 0.73057, 0.53366, 0.42621, 0.46871, 0.52643, 0.57385, 0.66023, 0.76614, 0.44068, 0.19377, 0.35843, 0.52879, 0.54442, 0.67852, 0.61972, 0.51303, 0.43663, 0.42326, 0.33372, 0.54431, 0.39476, 0.41071, 0.41706, 0.52146, 0.60328, 0.51826, 0.44949, 0.56341, 0.40952, 0.46189, 0.48231, 0.5776, 0.55661, 0.60806, 0.61621, 0.4837, 0.49643, 0.68936, 0.70421, 0.52976, 0.43136, 0.41023, 0.42258, 0.45506, 0.3592, 0.48884, 0.42962, 0.29846, 0.55631, 0.53487, 0.44995, 0.38745, 0.41179, 0.3982, 0.51386, 0.3155, 0.41075, 0.51384, 0.54007, 0.40546, 0.47758, 0.57292, 0.515, 0.45298, 0.42034, 0.40538, 0.48419, 0.4072, 0.5719, 0.64624, 0.59427, 0.50879, 0.5867, 0.65505, 0.51736, 0.28194, 0.49981, 0.59611, 0.44042, 0.56926, 0.4465, 0.54921, 0.58208, 0.65321, 0.63447, 0.64659, 0.26723, 0.36674, 0.38503, 0.43036, 0.57351, 0.61572, 0.74022, 0.64851, 0.37387, 0.44773, 0.55193, 0.57496, 0.49548, 0.6617, 0.64126, 0.44937, 0.54276, 0.56854, 0.57496, 0.55723, 0.37406, 0.43754, 0.64139, 0.56748, 0.64479, 0.47498, 0.41153, 0.40193, 0.46378, 0.43987, 0.46882, 0.41617, 0.5482, 0.64616, 0.5329, 0.65106, 0.52677, 0.57347, 0.64582, 0.49551, 0.55295, 0.76225, 0.53418, 0.58372, 0.54313, 0.39634, 0.26458, 0.46491, 0.47998, 0.46548, 0.51035, 0.40874, 0.43478, 0.57788, 0.47044, 0.50055, 0.38069, 0.37436, 0.24398, 0.32417, 0.45217, 0.36138, 0.63235, 0.50687, 0.41843, 0.47132, 0.46794, 0.18205, 0.34924, 0.3527, 0.58394, 0.62084, 0.63555, 0.58268, 0.60553, 0.4343, 0.42659, 0.33318, 0.2164, 0.33646, 0.59135, 0.67225, 0.80087, 0.64911, 0.53642, 0.39593, 0.56005, 0.56264, 0.37676, 0.43174, 0.65886, 0.61918, 0.53752, 0.64483, 0.50822, 0.49593, 0.52254, 0.47327, 0.29436, 0.21322, 0.43878, 0.66475, 0.33701, 0.50734, 0.49318, 0.42452, 0.4929, 0.57036, 0.65007, 0.59636, 0.44513, 0.52524, 0.59328, 0.66266, 0.60396, 0.71738, 0.55214, 0.48853, 0.46978, 0.4851, 0.55099, 0.64335, 0.59602, 0.66686, 0.58734, 0.39366, 0.34852, 0.54573, 0.45546, 0.39122, 0.53699, 0.63517, 0.58061, 0.45847, 0.54589, 0.52069, 0.41536, 0.42707, 0.55741, 0.69261, 0.54028, 0.48082, 0.51251, 0.4153, 0.43188, 0.55447, 0.51062, 0.5456, 0.53072, 0.53203, 0.57411, 0.49911, 0.32271, 0.32187, 0.60513, 0.45813, 0.44991, 0.63402, 0.63278, 0.52174, 0.4327, 0.53011, 0.55208, 0.60521, 0.55794, 0.64066, 0.55576, 0.49868, 0.45712, 0.65303, 0.55937, 0.46189, 0.65117, 0.60047, 0.51431, 0.26424, 0.34892, 0.50658, 0.61152, 0.43113, 0.57626, 0.4533, 0.56103, 0.49615, 0.4889, 0.65687, 0.66111, 0.61517, 0.45665, 0.67215, 0.62036, 0.3919, 0.26425, 0.46866, 0.6377, 0.49874, 0.53886, 0.56783, 0.4812, 0.37025, 0.43353, 0.46639, 0.53823, 0.66413, 0.60518, 0.58089, 0.5181, 0.52207, 0.52761, 0.46713, 0.29683, 0.35307, 0.46523, 0.56176, 0.33933, 0.7229, 0.68917, 0.49864, 0.502, 0.2408, 0.55321, 0.5925, 0.45128, 0.23512, 0.49784, 0.72979, 0.46895, 0.63122, 0.34932, 0.42801, 0.39915, 0.35954, 0.52449, 0.53546, 0.51605, 0.71679, 0.60214, 0.38193, 0.68108, 0.46412, 0.60574, 0.51114, 0.20655};
    float output[961];
    CNN_AdaptiveAveragePool(1, 50, 50, 31, 31, input, output);
    for (size_t i=0;i<961;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_AdaptiveAveragePool3(){
    const float input[] = {0.99673, 0.65344, 0.06811, 0.13796, 0.95438, 0.85975, 0.82446, 0.4007, 0.53337, 0.92955, 0.72641, 0.40234, 0.36735, 0.13188, 0.60957, 0.53266, 0.27882, 0.90549, 0.87151, 0.22238, 0.39778, 0.73703, 0.95603, 0.80757, 0.11968, 0.8509, 0.21041, 0.94993, 0.23169, 0.59644, 0.9187, 0.39748, 0.3951, 0.99079, 0.36708, 0.44136, 0.3042, 0.99454, 0.23568, 0.82552, 0.28106, 0.79026, 0.6864, 0.88805, 0.9394, 0.4187, 0.33027, 0.52679, 0.95342, 0.40549, 0.62346, 0.31752, 0.53175, 0.93852, 0.59519, 0.8566, 0.84924, 0.43338, 0.84715, 0.9744, 0.27778, 0.41824, 0.73753, 0.58179, 0.34779, 0.46546, 0.97068, 0.12737, 0.58684, 0.78095, 0.61403, 0.46156, 0.27858, 0.23963, 0.96456, 0.57921, 0.76036, 0.13373, 0.79488, 0.73905, 0.16944, 0.45877, 0.31097, 0.53291, 0.35859, 0.51488, 0.9658, 0.50203, 0.77665, 0.38722, 0.0944, 0.13612, 0.05687, 0.75249, 0.09842, 0.80373, 0.51782, 0.51784, 0.27934, 0.9069};
    float output[49];
    CNN_AdaptiveAveragePool(1, 10, 10, 7, 7, input, output);
    float expectedOutput[] = {0.69473, 0.37281, 0.37821, 0.73909, 0.63365, 0.67777, 0.6392, 0.56589, 0.61569, 0.49868, 0.5282, 0.62137, 0.73965, 0.48051, 0.58705, 0.66038, 0.68334, 0.52285, 0.55857, 0.64867, 0.54137, 0.50307, 0.58148, 0.76322, 0.70247, 0.56916, 0.69018, 0.79511, 0.4521, 0.45753, 0.57948, 0.6348, 0.57511, 0.48722, 0.78721, 0.42595, 0.37747, 0.44754, 0.60431, 0.576, 0.55182, 0.67445, 0.21469, 0.24068, 0.35171, 0.4439, 0.63702, 0.51897, 0.58753};
    for (size_t i=0;i<49;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Permute3(){
    float outputReg[] = {0.03958, -0.02145, -0.11738, -0.0546, 0.1054, -0.01394, -0.18461, -0.1425, -0.18486, -0.1617, -0.14764, 0.19548, -0.11502, -0.12154, 0.21121, 0.48961, -0.05304, -0.05735, 0.13819, 0.47433, -0.12459, -0.0593, 0.22122, 0.73505};
    float output[24];
    CNN_Permute(1, 6, 4, 0, 2, 1, outputReg, output);
    float expectedOutput[] = {0.03958, 0.1054, -0.18486, -0.11502, -0.05304, -0.12459, -0.02145, -0.01394, -0.1617, -0.12154, -0.05735, -0.0593, -0.11738, -0.18461, -0.14764, 0.21121, 0.13819, 0.22122, -0.0546, -0.1425, 0.19548, 0.48961, 0.47433, 0.73505};
    for (size_t i=0;i<24;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_Permute4(){
    float outputProb[] = {0.00044, 0.99956, 0.00296, 0.99704, 0.09634, 0.90366, 0.99613, 0.00387, 0.99804, 0.00196, 0.99771, 0.00229};
    float output[12];
    CNN_Permute(1, 6, 2, 0, 2, 1, outputProb, output);
    float expectedOutput[] = {0.00044, 0.00296, 0.09634, 0.99613, 0.99804, 0.99771, 0.99956, 0.99704, 0.90366, 0.00387, 0.00196, 0.00229};
    for (size_t i=0;i<12;++i){
        printf("Output [%d]: %f\n", i, output[i]);
        assert(equalFloatDefault(output[i], expectedOutput[i]));
    }
}

void CNNTest_BoxNmsIdx0(){
    const float boxes[] = {50.0, 50.0, 70.0, 70.0, 10.0, 10.0, 20.0, 20.0, 40.0, 40.0, 70.0, 70.0, 60.0, 60.0, 80.0, 80.0};
    const float scores[] = {0.8, 0.1, 0.1, 0.1};
    int output[4];
    int outputLen = CNN_BoxNmsIdx(4, boxes, scores, 0.3, output);
    assert(outputLen == 3);
    int expectedOutput[] = {0, 1, 3};
    for (size_t i=0;i<3;++i){
        printf("Output [%d]: %d\n", i, output[i]);
        assert(output[i] == expectedOutput[i]);
    }
}

#pragma clang diagnostic pop