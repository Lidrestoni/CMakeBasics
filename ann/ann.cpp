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

void shuffleMatRows(cv::Mat &Input, cv::Mat &Output){
/*Essa função cria um vetor, no qual são colocados números de bilhetes de 0 a n-1 (onde n é o número de linha de uma das matrizes de origem, pois ambas devem ter o mesmo número de linhas).
* Então esses números são embaralhados, e as matrizes de saída são montadas seguindo essa ordem de bilhetes sorteados
*/
	std::vector <int> tickets;
	cv::Mat_<float>input2(1, 10, CV_32FC1);
	cv::Mat_<float>output2(1, 1, CV_32FC1);
	for (int i = 0; i < Input.rows; i++)
		tickets.push_back(i);
	std::srand ( unsigned ( std::time(0) ) );
	std::random_shuffle ( tickets.begin(), tickets.end() );
	for (int i = 0; i < Input.rows; i++){
		input2.push_back(Input.row(tickets[i]));
		output2.push_back(Output.row(tickets[i]));
	}
	Input = input2.clone();
	Output = output2.clone();
}

int main(int argc, char** argv ){	
	cv::Mat_<float> bestInput, bestOutput, trainingInput, trainingOutput,forPredictionInput, forPredictionOutput, predictedOutput;
	cv::Ptr<cv::ml::ANN_MLP> mlp;
	double predictedBOSR,bestOSR = 0;
	FILEtoMAT((char *)"data", trainingInput, trainingOutput);
	int i=3000;
	while(i--){	
		shuffleMatRows(trainingInput, trainingOutput);
		mlp = getTrainedNeuralNetwork(trainingInput, trainingOutput);
		forPredictionInput = trainingInput.clone();
		forPredictionOutput = trainingOutput.clone();
		shuffleMatRows(forPredictionInput,forPredictionOutput);
		if(i<1500){		
			predictedOutput = cv::Mat_<float>(1, 1, CV_32FC1);
			mlp->predict(forPredictionInput, predictedOutput);
			predictedBOSR = outputSuccessRate(predictedOutput, forPredictionOutput);
			if(predictedBOSR>bestOSR){
				bestOSR = predictedBOSR;
				bestInput = trainingInput.clone();
				bestOutput = trainingOutput.clone();	
				printf(">>%.2lf%% de acertos<<\n", bestOSR*100);
			}
		}
	}

	return 0;
}
