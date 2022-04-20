#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include "MLPP/UniLinReg/UniLinReg.hpp"
#include "MLPP/LinReg/LinReg.hpp"
#include "MLPP/LogReg/LogReg.hpp"
#include "MLPP/CLogLogReg/CLogLogReg.hpp"
#include "MLPP/ExpReg/ExpReg.hpp"
#include "MLPP/ProbitReg/ProbitReg.hpp"
#include "MLPP/SoftmaxReg/SoftmaxReg.hpp"
#include "MLPP/TanhReg/TanhReg.hpp"
#include "MLPP/MLP/MLP.hpp"
#include "MLPP/SoftmaxNet/SoftmaxNet.hpp"
#include "MLPP/AutoEncoder/AutoEncoder.hpp"
//#include "MLPP/ANN/ANN.hpp"
//#include "MLPP/MANN/MANN.hpp"
#include "MLPP/MultinomialNB/MultinomialNB.hpp"
#include "MLPP/BernoulliNB/BernoulliNB.hpp"
#include "MLPP/GaussianNB/GaussianNB.hpp"
#include "MLPP/KMeans/KMeans.hpp"
#include "MLPP/kNN/kNN.hpp"
#include "MLPP/PCA/PCA.hpp"
#include "MLPP/OutlierFinder/OutlierFinder.hpp"
#include "MLPP/Stat/Stat.hpp"
#include "MLPP/LinAlg/LinAlg.hpp"
#include "MLPP/Activation/Activation.hpp"
#include "MLPP/Cost/Cost.hpp"
#include "MLPP/Data/Data.hpp"
#include "MLPP/Convolutions/Convolutions.hpp"
#include "MLPP/SVC/SVC.hpp"
#include "MLPP/NumericalAnalysis/NumericalAnalysis.hpp"
#include "MLPP/DualSVC/DualSVC.hpp"
//#include "MLPP/GAN/GAN.hpp"
//#include "MLPP/WGAN/WGAN.hpp"
#include "MLPP/Transforms/Transforms.hpp"

using namespace MLPP;


int main() {

    // // OBJECTS
    Stat stat;
    LinAlg alg;
    Activation avn;
    Cost cost;
    Data data; 
    Convolutions conv; 

     // SUPPORT VECTOR CLASSIFICATION
     auto [inputSet, outputSet] = data.loadBreastCancerSVC();
     SVC model(inputSet, outputSet, 1);
     model.SGD(0.00001, 100000, 1);
     alg.printVector(model.modelSetTest(inputSet));
     std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

    return 0;
}

