#include <stdio.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "ml.h"

void mlp(cv::Mat& trainingData, std::vector<int>& index)
{
    int input_neurons = 8;
    int hidden_neurons = 100;
    int output_neurons = 12;
    cv::Mat layerSizes = cv::Mat(3, 1, CV_32SC1);
    layerSizes.row(0) = cv::Scalar(input_neurons);
    layerSizes.row(1) = cv::Scalar(hidden_neurons);
    layerSizes.row(2) = cv::Scalar(output_neurons);

    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
    mlp->setLayerSizes(layerSizes);
    mlp->setTrainMethod(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.00001f));
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP,0.1f,0.1f);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);

    cv::Mat trainClasses;

    std::cout << "Poker" << std::endl;
    trainClasses = cv::Mat::zeros(trainingData.rows, 12, CV_32FC1);
    //trainClasses.create(trainingData.rows, 12, CV_32FC1);
    for (int i = 0; i < trainClasses.rows; i++)
    {
        trainClasses.at<float>(i, index[i]) = 1.f;
    }

    std::cout << "Row of trainClass: " << trainClasses.cols << std::endl;
    std::cout << "Row of trainData: " << trainingData.cols << std::endl;
    std::cout << "Koker" << std::endl;
    cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, trainClasses);
    mlp->train(td);
    std::cout << "Training Done" << std::endl;

    mlp->save("neural_network.xml");

}



cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples, const cv::Mat& trainResponses)
{
  int networkInputSize = trainSamples.cols;
  int networkOutputSize = trainResponses.cols;
  cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
  std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2, networkOutputSize };
  mlp->setLayerSizes(layerSizes);
  mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
  mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
  return mlp;
}


int main(int argc, char** argv )
{
	cv::Mat trainSamples;
  	cv::Mat trainResponses;
	
	//cv::ml::ANN_MLP *mlp;
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);

	//cv::ANN_MLP_TrainParams *params;

/*class Ann : public cv::ml::ANN_MLP{
	void setTrainMethod(int method, double param1 = 0, double param2 = 0){}
	int getTrainMethod() const{}
	void setActivationFunction(int type, double param1 = 0, double param2 = 0){}
	void setLayerSizes(cv::InputArray _layer_sizes){}
	cv::Mat getLayerSizes() const{}
	cv::TermCriteria getTermCriteria() const{}
	void setTermCriteria(cv::TermCriteria val){}
	double getBackpropWeightScale() const{}
	void setBackpropWeightScale(double val){}
	double getBackpropMomentumScale() const{}
	void setBackpropMomentumScale(double val){}
	double getRpropDW0() const{}
	void setRpropDW0(double val){}
	double getRpropDWPlus() const{}
	void setRpropDWPlus(double val){}
	double getRpropDWMinus() const{}
	void setRpropDWMinus(double val){}
	double getRpropDWMin() const{}
	void setRpropDWMin(double val){}
	double getRpropDWMax() const{}
	void setRpropDWMax(double val){}
	cv::Mat getWeights(int layerIdx) const{}

	int getVarCount() const{}
	bool isTrained() const{}
	bool isClassifier() const{}
	float predict( cv::InputArray samples, cv::OutputArray results=cv::noArray(), int flags=0 ) const{}
};
cv::Mat mt(600,11,CV_8UC1);
Ann *mlp = cv::ml::ANN_MLP::create();//mt, 1, 0, 0);*/


//Ann mlp(const Mat& layerSizes, int activateFunc=CvANN_MLP::SIGMOID_SYM, double fparam1=0, double fparam2=0 );
/*CvANN_MLP mlp;
CvANN_MLP::create();
CvANN_MLP::CvANN_MLP();
CvANN_MLP_TrainParams::CvANN_MLP_TrainParams()
{
    term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01 );
    train_method = RPROP;
    bp_dw_scale = bp_moment_scale = 0.1;
    rp_dw0 = 0.1; rp_dw_plus = 1.2; rp_dw_minus = 0.5;
    rp_dw_min = FLT_EPSILON; rp_dw_max = 50.;
}*/


   

    return 0;
}
