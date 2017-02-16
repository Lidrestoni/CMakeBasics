#include <stdio.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "ml.h"

cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat_<float>& trainSamples, const cv::Mat_<float>& trainResponses)
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
	cv::Mat_<float> trainSamples(1, 10, CV_32FC1);
	cv::Mat_<float> trainResponses(1, 1, CV_32FC1);
	trainSamples.reserve(4000);
	trainResponses.reserve(4000);
	float in[11], out;
	cv::Mat_<float> mat;
	while(scanf("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", &in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6], &in[7], &in[8], &in[9], &out)!=EOF){
			mat = cv::Mat_<float>(1, 10, CV_32FC1);
			mat.reserve(2);
			for(int i=0; i<10;i++)
				mat.at<float>(0,i) = in[i];
			trainSamples.push_back(mat);
			mat = cv::Mat_<float>(1, 1, CV_32FC1);
			mat.reserve(2);
			mat.at<float>(0,0) = out;
			trainResponses.push_back(mat);
	}
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
	cv::Mat_<float> temp(1, 1, CV_32FC1);
	 mlp->predict(trainSamples, temp);

	std::cout << temp;
    return 0;
}
