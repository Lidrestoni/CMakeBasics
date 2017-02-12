#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "ml.h"


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
	cv::ml::ANN_MLP *mlp;
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
