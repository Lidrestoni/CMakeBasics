/*
* Acadêmicos : Andrey Vinicius Fagundes e Guilherme Antunes da Silva
*/
#include <stdio.h>
#include <vector>
#include <iostream>
/*Bibliotecas OpenCV: */
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "ml.h"
/*Bibliotecas usadas na shuffleMatRows: */
#include <ctime>  
#include <cstdlib> 

double abs(double x){
	return x<0? -x : x;
}

double outputSuccessRate(const cv::Mat &are,const cv::Mat &shouldBe){
	int s=0;	
	for (int i = 0; i < are.rows; i++)
		s+=(abs(shouldBe.at<float>(i,0)-are.at<float>(i,0))<0.6); 
	return (double) s/are.rows;
}


cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat_<float>& trainSamples, const cv::Mat_<float>& trainResponses){
	int networkInputSize = trainSamples.cols; 
	int networkOutputSize = trainResponses.cols;
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2, networkOutputSize };
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	return mlp;
}


void FILEtoMAT(char *name, cv::Mat_<float>& input, cv::Mat_<float>& output){
	input = cv::Mat_<float>(1, 10, CV_32FC1);
	output = cv::Mat_<float>(1, 1, CV_32FC1);	
	FILE *file = fopen(name, "r");	
	input.reserve(4000);
	output.reserve(4000);
	float in[11], out;
	cv::Mat_<float> mat;
	while(fscanf(file,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", &in[0], &in[1], &in[2], &in[3], &in[4], &in[5], &in[6], &in[7], &in[8], &in[9], &out)!=EOF){
			mat = cv::Mat_<float>(1, 10, CV_32FC1);
			mat.reserve(2);
			for(int i=0; i<10;i++)
				mat.at<float>(0,i) = in[i];
			input.push_back(mat);
			mat = cv::Mat_<float>(1, 1, CV_32FC1);
			mat.reserve(2);
			mat.at<float>(0,0) = out;
			output.push_back(mat);
	}
	fclose(file);
}

void shuffleMatRows(const cv::Mat &origInput, const cv::Mat &origOutput, cv::Mat &shuffInput, cv::Mat &shuffOutput){
/*Essa função cria um vetor, no qual são colocados números de bilhetes de 0 a n-1 (onde n é o número de linha de uma das matrizes de origem, pois ambas devem ter o mesmo número de linhas).
* Então esses números são embaralhados, e as matrizes de saída são montadas seguindo essa ordem de bilhetes sorteados
*/
	std::vector <int> tickets;
	for (int i = 0; i < origInput.rows; i++)
		tickets.push_back(i);
	std::srand ( unsigned ( std::time(0) ) );
	std::random_shuffle ( tickets.begin(), tickets.end() );
	for (int i = 0; i < origInput.rows; i++){
		shuffInput.push_back(origInput.row(tickets[i]));
		shuffOutput.push_back(origOutput.row(tickets[i]));
	}
}

int main(int argc, char** argv ){	
	cv::Mat_<float> originalInput, originalOutput, trainingInput, trainingOutput;
	FILEtoMAT((char *)"data", originalInput, originalOutput);
	shuffleMatRows(originalInput, originalOutput, trainingInput, trainingOutput);
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainingInput, trainingOutput);
	cv::Mat_<float> temp(1, 1, CV_32FC1);
	mlp->predict(originalInput, temp);
	printf(">>%.2lf%% de acertos<<\n", outputSuccessRate(temp, originalOutput)*100);

	return 0;
}
