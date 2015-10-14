/*******************************************************************************
* Copyright (c) 2015 IBM Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include <iostream>
#include <fstream>
#include <time.h>
#include <limits.h>

#include <opencv2/opencv.hpp>
#include <svm.h>
#include <linear.h>

#include "RecogHandGestureConstants.h"
#include "MyDatasetConstants.h"
#include "GTEAConstants.h"
#include "GTEAGazePlusConstants.h"
#include "CMUKitchenConstants.h"
#include "PcaWrapper.h"
#include "FisherVectorWrapper.h"
#include "SVMUtils.h"
#include "FileUtils.h"
#include "StringUtils.h"

using namespace std;
using namespace cv;

static const int NUMBER_REPEAT_TEST = 5;

void saveParametersToOutputCSV(ofstream &ofsCSV)
{
	if (ofsCSV) {
		ofsCSV << "#NUMBER_REPEAT_TEST," << NUMBER_REPEAT_TEST << endl;
		ofsCSV << "#USE_HAND_MASK_KEYPOINT_REMOVE," << USE_HAND_MASK_KEYPOINT_REMOVE << endl;
		ofsCSV << endl;

		ofsCSV << "#HAND_DETECT_FEATURE_SET," << HAND_DETECT_FEATURE_SET << endl;
		ofsCSV << "#HAND_DETECT_NUM_MODELS_TO_AVERAGE," << HAND_DETECT_NUM_MODELS_TO_AVERAGE << endl;
		ofsCSV << "#HAND_DETECT_STEP_SIZE," << HAND_DETECT_STEP_SIZE << endl;
		ofsCSV << "#HAND_DETECT_PROB_THRES," << HAND_DETECT_PROB_THRES << endl;
		ofsCSV << "#HAND_DETECT_AVERAGE_PROBABILITY_TIME_LENGTH," << HAND_DETECT_AVERAGE_PROBABILITY_TIME_LENGTH << endl;
		ofsCSV << endl;

		ofsCSV << "#USE_HAND_HOG," << USE_HAND_HOG << endl;
		ofsCSV << "#HAND_HOG_BLOCK_SIZE," << HAND_HOG_BLOCK_SIZE << endl;
		ofsCSV << "#HAND_HOG_STRIDE_SIZE," << HAND_HOG_STRIDE_SIZE << endl;
		ofsCSV << "#HAND_HOG_CELL_SIZE," << HAND_HOG_CELL_SIZE << endl;
		ofsCSV << "#HAND_HOG_RAD_BINS," << HAND_HOG_RAD_BINS << endl;
		ofsCSV << endl;

		ofsCSV << "#DENSE_HAND_HOG_KEYPOINT_STEP," << DENSE_HAND_HOG_KEYPOINT_STEP << endl;
		ofsCSV << "#DENSE_HAND_HOG_GRID_2D," << DENSE_HAND_HOG_GRID_2D << endl;
		ofsCSV << "#DENSE_HAND_HOG_NEIGHBOR_SIZE," << DENSE_HAND_HOG_NEIGHBOR_SIZE << endl;
		ofsCSV << "#DENSE_HAND_HOG_BIN_NUM," << DENSE_HAND_HOG_BIN_NUM << endl;
		ofsCSV << "#DENSE_HAND_HOG_USE_ZERO_MAGNITUDE_BIN," << DENSE_HAND_HOG_USE_ZERO_MAGNITUDE_BIN << endl;
		ofsCSV << "#DENSE_HAND_HOG_FULL_RADIAN," << DENSE_HAND_HOG_FULL_RADIAN << endl;
		ofsCSV << endl;

		ofsCSV << "#USE_CROSS_VALIDATION_TRAIN_SVM," << USE_CROSS_VALIDATION_TRAIN_SVM << endl;
		ofsCSV << "#DEFAULT_TRAIN_SVM_C," << DEFAULT_TRAIN_SVM_C << endl;
		ofsCSV << "#SCALE_FEATURE_BEFORE_SVM_MIN," << SCALE_FEATURE_BEFORE_SVM_MIN << endl;
		ofsCSV << "#SCALE_FEATURE_BEFORE_SVM_MAX," << SCALE_FEATURE_BEFORE_SVM_MAX << endl;
		ofsCSV << "#PCA_TRAIN_FEATURE_NUM," << PCA_TRAIN_FEATURE_NUM << endl;
		ofsCSV << "#PCA_TRAIN_FEATURE_NUM_PER_IMAGE," << PCA_TRAIN_FEATURE_NUM_PER_IMAGE << endl;
		ofsCSV << "#FV_TRAIN_FEATURE_NUM," << FV_TRAIN_FEATURE_NUM << endl;
		ofsCSV << "#FV_TRAIN_FEATURE_NUM_PER_IMAGE," << FV_TRAIN_FEATURE_NUM_PER_IMAGE << endl;
		ofsCSV << "#FV_K," << FV_K << endl;
		ofsCSV << "#FV_PCA_DIM," << FV_PCA_DIM << endl;
		ofsCSV << "#NORM_FISHER_VECTOR_FEATURE_TYPE," << NORM_FISHER_VECTOR_FEATURE_TYPE << endl;
		ofsCSV << endl;

		ofsCSV << "#MYDATASET_MOVIE_FRAME_WIDTH," << MYDATASET_MOVIE_FRAME_WIDTH << endl;
		ofsCSV << "#MYDATASET_MOVIE_FRAME_HEIGHT," << MYDATASET_MOVIE_FRAME_HEIGHT << endl;
		ofsCSV << endl;

		ofsCSV << "#MYDATASET_RECOG_GESTURE_FRAME_LENGTH," << MYDATASET_RECOG_GESTURE_FRAME_LENGTH << endl;
		ofsCSV << "#MYDATASET_RECOG_GESTURE_FRAME_STEP," << MYDATASET_RECOG_GESTURE_FRAME_STEP << endl;
		ofsCSV << endl;

		ofsCSV << "#MYDATASET_HAND_DETECT_MODEL_LIST_FILE," << MYDATASET_HAND_DETECT_MODEL_LIST_FILE << endl;
		ofsCSV << "#MYDATASET_HAND_DETECT_GLOBAL_FEATURE_LIST_FILE," << MYDATASET_HAND_DETECT_GLOBAL_FEATURE_LIST_FILE << endl;
		ofsCSV << endl;

		ofsCSV << "#GTEAGAZE_MOVIE_FRAME_WIDTH," << GTEAGAZE_MOVIE_FRAME_WIDTH << endl;
		ofsCSV << "#GTEAGAZE_MOVIE_FRAME_HEIGHT," << GTEAGAZE_MOVIE_FRAME_HEIGHT << endl;
		ofsCSV << endl;

		ofsCSV << "#GTEAGAZE_ACTION_VERB_RECOG_GESTURE_FRAME_LENGTH," << GTEAGAZE_ACTION_VERB_RECOG_GESTURE_FRAME_LENGTH << endl;
		ofsCSV << "#GTEAGAZE_ACTION_VERB_OBJECT_RECOG_GESTURE_FRAME_LENGTH," << GTEAGAZE_ACTION_VERB_OBJECT_RECOG_GESTURE_FRAME_LENGTH << endl;
		ofsCSV << "#GTEAGAZE_RECOG_GESTURE_FRAME_STEP," << GTEAGAZE_RECOG_GESTURE_FRAME_STEP << endl;
		ofsCSV << endl;
	}
}

vector<vector<cv::Mat> > readMovieFeature(const string &movieFeatureFile, const string &delimFeatureFile)
{
	vector<vector<cv::Mat> > movieFetures;

    ifstream file(movieFeatureFile.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

	string line;
	while (getline(file,line)) {
		if (line.size()>0) {
			vector<string> tokens = StringUtils::splitString(line, delimFeatureFile);
			CV_Assert(tokens.size()>=2);

			vector<Mat> frameFeatures;
			for (int i=0; i<tokens.size()-1; i++) { // last token is timestamp
				Mat feature;
				//FileUtils::readMat(tokens[i].c_str(), feature);
				FileUtils::readMatBin(tokens[i].c_str(), feature);
				frameFeatures.push_back(feature);
			}
			movieFetures.push_back(frameFeatures);
		}
	}

	return movieFetures;
}

vector<cv::Mat> getRandomTrainFeatures(const vector<string> &movieFeatureFiles, const string &delimFeatureFile, int trainFeatureNum, int trainFeatureNumPerMovie)
{
	CV_Assert(movieFeatureFiles.size()>0);

	vector<vector<Mat> > feature = readMovieFeature(movieFeatureFiles[0], delimFeatureFile);
	CV_Assert(feature.size()>0 && feature[0].size()>0);

	int featureTypeNum = feature[0].size();

	// init random number
	srand((unsigned int)time(NULL));

	int trainMovieNum = trainFeatureNum/trainFeatureNumPerMovie;

	vector<cv::Mat> features;
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		int featureDim = feature[0][ftype].cols;
		features.push_back(cv::Mat::zeros(trainFeatureNumPerMovie*trainMovieNum, featureDim, CV_32F));
	}

	#pragma omp parallel for
	for (int i=0; i<trainMovieNum; i++) {
		float randMovie = static_cast<float>(rand())/static_cast<float>(RAND_MAX+1);
		vector<vector<Mat> > descriptors = readMovieFeature(movieFeatureFiles[movieFeatureFiles.size()*randMovie], delimFeatureFile);

		if (descriptors.size()==0) {
			continue;
		}
		bool movieDescZero = true;
		for (int j=0; j<descriptors.size(); j++) {
			bool frameDescAllNonZero = true;
			for (int k=0; k<descriptors[j].size(); k++) {
				if (countNonZero(descriptors[j][k])==0) {
					frameDescAllNonZero = false;
					break;
				}
			}
			if (frameDescAllNonZero) {
				movieDescZero = false;
				break;
			}
		}
		if (movieDescZero) {
			continue;
		}

		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			for (int j=0; j<trainFeatureNumPerMovie; ) {
				float randFrame = static_cast<float>(rand())/static_cast<float>(RAND_MAX+1);
				int f = descriptors.size()*randFrame;

				float randFeature = static_cast<float>(rand())/static_cast<float>(RAND_MAX+1);
				int k = descriptors[f][ftype].rows*randFeature;

				if (norm(descriptors[f][ftype].row(k))>0) {
					for (int d=0; d<descriptors[f][ftype].cols; d++) {
						CV_Assert(i*trainFeatureNumPerMovie + j<trainFeatureNumPerMovie*trainMovieNum);
						CV_Assert(f<descriptors.size());
						CV_Assert(k<descriptors[f][ftype].rows);

						features[ftype].at<float>(i*trainFeatureNumPerMovie + j, d) = descriptors[f][ftype].at<float>(k,d);
					}
					j++;
				} else {
					cout << "skip select feature" << endl;
				}
			}
		}

		cout << "Number of gestures selected for random training : " << i << " / " << trainMovieNum << endl;
	}

	return features;
}

void calcSVMTrainFeature(PcaWrapper **pcaWrappers, FisherVectorWrapper **fvWrappers, const vector<string> &movieFeatureFiles, const string &delimFeatureFile, 
						int recogFrameLength, int recogFrameStep, const vector<string> &movieLabels, vector<float> &minFeatures, vector<float> &maxFeatures, 
						cv::Mat &D, vector<string> &labels)
{
	labels.clear();

	cout << "Start extract Fisher Vector features for training...." << endl;
	vector<Mat> trainFeatures;
	for (int i=0; i<movieFeatureFiles.size(); i++) {
		vector<vector<cv::Mat> > movieFeatures = readMovieFeature(movieFeatureFiles[i], delimFeatureFile);

		for (int t=0; t<movieFeatures.size()-recogFrameLength; t+=recogFrameStep) {
			Mat trainFeature;

			for (int ftype=0; ftype<movieFeatures[t].size(); ftype++) {
				Mat feature;
				for (int j=t; j<t+recogFrameLength; j++) {
					feature.push_back(movieFeatures[j][ftype]);
				}
				CV_Assert(!feature.empty());

				cout << "start calc pca" << endl;
				Mat pcaFeature = pcaWrappers[ftype]->calcPcaProject(feature);
				cout << "end calc pca" << endl;

				cout << "start calc fisher vector" << endl;
				Mat fvFeature = fvWrappers[ftype]->calcFisherVector(pcaFeature);
				cout << "end calc fisher vector" << endl;

				if (ftype==0) {
					trainFeature = fvFeature;
				} else {
					vconcat(trainFeature, fvFeature, trainFeature);
				}
			}

			trainFeatures.push_back(trainFeature);
			labels.push_back(movieLabels[i]);
		}
	}
	cout << "End extract Fisher Vector features for training. Dimension of Fisher Vector feature = " << trainFeatures[0].rows << std::endl;

	// convert input data to Matrix for SVM
	D = cv::Mat(trainFeatures.size(), trainFeatures[0].rows, CV_32F);
	for (int i=0; i<trainFeatures.size(); i++) {
		for (int j=0; j<trainFeatures[i].rows; j++) {
			D.at<float>(i, j) = trainFeatures[i].at<double>(j);
		}
	}
	// scale data before training SVM
	SVMUtils::prepareScaleData(D, minFeatures, maxFeatures);
	SVMUtils::scaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, D);
}

void calcSVMTestFeature(PcaWrapper **pcaWrappers, FisherVectorWrapper **fvWrappers, const string &movieFeatureFile, const string &delimFeatureFile,
						int recogFrameLength, int recogFrameStep, vector<float> &minFeatures, vector<float> &maxFeatures, vector<cv::Mat> &testDs)
{
	vector<vector<cv::Mat> > movieFeatures = readMovieFeature(movieFeatureFile, delimFeatureFile);

	for (int t=0; t<movieFeatures.size()-recogFrameLength; t+=recogFrameStep) {
		cv::Mat fvFeatures;

		for (int ftype=0; ftype<movieFeatures[t].size(); ftype++) {
			cout << "start calc fisher vector" << endl;
			Mat feature;
			for (int j=t; j<t+recogFrameLength; j++) {
				feature.push_back(movieFeatures[j][ftype]);
			}
			CV_Assert(!feature.empty());

			cout << "start calc pca" << endl;
			Mat pcaFeature = pcaWrappers[ftype]->calcPcaProject(feature);
			cout << "end calc pca" << endl;

			cout << "start calc fisher vector" << endl;
			Mat fvFeature = fvWrappers[ftype]->calcFisherVector(pcaFeature);
			cout << "end calc fisher vector" << endl;

			if (ftype==0) {
				fvFeatures = fvFeature;
			} else {
				vconcat(fvFeatures, fvFeature, fvFeatures);
			}
		}

		// convert input data to Matrix for SVM
		cv::Mat testD(1, fvFeatures.rows, CV_32F);
		for (int j=0; j<fvFeatures.rows; j++) {
			testD.at<float>(0, j) = fvFeatures.at<double>(j);
		}
		// scale data before testing SVM
		SVMUtils::scaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, testD);

		testDs.push_back(testD);
	}
}

// LibSVM
void trainSvmLibSVM(PcaWrapper **pcaWrappers, FisherVectorWrapper **fvWrappers, const vector<string> &movieFeatureFiles, const string &delimFeatureFile, 
					int recogFrameLength, int recogFrameStep, const vector<string> &movieClassIDs, const set<string> &uniqueLabels, string outputModelFile, 
					vector<float> &minFeatures, vector<float> &maxFeatures)
{
    svm_problem prob;
	{
		cv::Mat D;
		vector<string> labels;
		calcSVMTrainFeature(pcaWrappers, fvWrappers, movieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, movieClassIDs, 
							minFeatures, maxFeatures, D, labels);

		prob.l = labels.size();
		prob.y = new double[prob.l];
		for (int i=0; i<labels.size(); i++) {
			prob.y[i] = std::distance(uniqueLabels.begin(), uniqueLabels.find(labels[i]));
		}

		prob.x = new svm_node*[prob.l];
		for (int i=0; i<D.rows; i++) {
			prob.x[i] = new svm_node[D.cols+1];
			for (int j=0; j<D.cols; j++) {
				prob.x[i][j].index = j+1;
				prob.x[i][j].value = D.at<float>(i, j);
			}
			prob.x[i][D.cols].index = -1;
		}
	}

	svm_parameter param;
	param.svm_type = C_SVC;
	param.kernel_type = LINEAR;
	param.C = DEFAULT_TRAIN_SVM_C;
	param.gamma = 1.0;
	param.coef0 = 0;
	param.cache_size = 100;
	param.eps = 1e-3;
	param.shrinking = 1;
	param.probability = 1;
	param.degree = 0.0;
	param.nu = 0.0;
	param.p = 0.0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	const char *error_msg = svm_check_parameter(&prob,&param);

	if (USE_CROSS_VALIDATION_TRAIN_SVM) {
		cout << "Start SVM train cross validation...." << endl;
		double bestC = 0.01;
		double bestAcc = -1;
		double *target = new double[prob.l];
		for (double c=0.01; c<10000.0; c*=10.0) {
			param.C = c;
			svm_cross_validation(&prob, &param, 10, target);

			int correct = 0;
			for(int i=0;i<prob.l;i++) {
				if(target[i] == prob.y[i]) {
					++correct;
				}
			}
			double acc = 100.0*correct/prob.l;
			if (acc>bestAcc) {
				bestAcc = acc;
				bestC = c;
				cout << "Cross Validation improved, best accuracy = " << bestAcc << ", best parameter C = " << bestC << endl;
			} else {
				cout << "Cross Validation not improved, accuracy = " << acc << ", parameter C = " << c 
					<< ", best accuracy = " << bestAcc << ", best parameter C = " << bestC << endl;
			}
		}
		delete[] target;

		param.C = bestC;
		cout << "End SVM train cross validation." << endl;
	}

	cout << "Start SVM train ...." << endl;
	svm_model *svmModel = svm_train(&prob, &param);
	cout << "End SVM train." << endl;
	
	cout << "Start save SVM model ...." << endl;
	svm_save_model(outputModelFile.c_str(), svmModel);
	cout << "End save SVM model." << endl;

	delete[] prob.y;
    for (int i=0; i<prob.l; i++) {
	    delete[] prob.x[i];
	}
    delete[] prob.x;
}
// LibLinear
void trainSvmLibLinear(PcaWrapper **pcaWrappers, FisherVectorWrapper **fvWrappers, const vector<string> &movieFeatureFiles, const string &delimFeatureFile, 
					int recogFrameLength, int recogFrameStep, const vector<string> &movieClassIDs, const set<string> &uniqueLabels, string outputModelFile, 
					vector<float> &minFeatures, vector<float> &maxFeatures)
{
    problem prob;
	{
		cv::Mat D;
		vector<string> labels;
		calcSVMTrainFeature(pcaWrappers, fvWrappers, movieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, movieClassIDs, 
							minFeatures, maxFeatures, D, labels);

		prob.l = labels.size();
		prob.bias = -1;
		prob.n = D.cols;
		prob.y = new double[prob.l];
		for (int i=0; i<labels.size(); i++) {
			prob.y[i] = std::distance(uniqueLabels.begin(), uniqueLabels.find(labels[i]));
		}

		prob.x = new feature_node*[prob.l];
		for (int i=0; i<D.rows; i++) {
			prob.x[i] = new feature_node[D.cols+1];
			for (int j=0; j<D.cols; j++) {
				prob.x[i][j].index = j+1;
				prob.x[i][j].value = D.at<float>(i, j);
			}
			prob.x[i][D.cols].index = -1;
		}
	}

	parameter param;
	param.solver_type = L2R_L2LOSS_SVC; // default parameter is L2R_L2LOSS_SVC_DUAL
	param.C = DEFAULT_TRAIN_SVM_C;
	param.eps = 1e-3;
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	const char *error_msg = check_parameter(&prob,&param);

	if (USE_CROSS_VALIDATION_TRAIN_SVM) {
		cout << "Start SVM train cross validation...." << endl;
		double bestC = 0.01;
		double bestAcc = -1;
		double *target = new double[prob.l];
		for (double c=0.01; c<10000.0; c*=10.0) {
			param.C = c;
			cross_validation(&prob, &param, 10, target);

			int correct = 0;
			for(int i=0;i<prob.l;i++) {
				if(target[i] == prob.y[i]) {
					++correct;
				}
			}
			double acc = 100.0*correct/prob.l;
			if (acc>bestAcc) {
				bestAcc = acc;
				bestC = c;
				cout << "Cross Validation improved, best accuracy = " << bestAcc << ", best parameter C = " << bestC << endl;
			} else {
				cout << "Cross Validation not improved, accuracy = " << acc << ", parameter C = " << c 
					<< ", best accuracy = " << bestAcc << ", best parameter C = " << bestC << endl;
			}
		}
		delete[] target;

		param.C = bestC;
		cout << "End SVM train cross validation." << endl;
	}

	cout << "Start SVM train ...." << endl;
	model *svmModel = train(&prob, &param);
	cout << "End SVM train." << endl;
	
	cout << "Start save SVM model ...." << endl;
	save_model(outputModelFile.c_str(), svmModel);
	cout << "End save SVM model." << endl;

	delete[] prob.y;
    for (int i=0; i<prob.l; i++) {
	    delete[] prob.x[i];
	}
    delete[] prob.x;
}

// LibSVM
double testSvmLibSVM(PcaWrapper **pcaWrappers, FisherVectorWrapper **fvWrappers, const vector<string> &movieFeatureFiles, const string &delimFeatureFile, 
					int recogFrameLength, int recogFrameStep, const vector<string> &movieClassIDs, const set<string> &uniqueLabels, 
					vector<float> &minFeatures, vector<float> &maxFeatures, const svm_model *svmModel, 
					ofstream &ofsLogCSV, ofstream &ofsEstProbCSV, ofstream &ofsEstLabelCSV)
{
	int correct = 0;
	int wrong = 0;
	vector<int> correctByClass(uniqueLabels.size(), 0);
	vector<int> wrongByClass(uniqueLabels.size(), 0);
	vector<vector<double> > estimateProbs;
	vector<int> estimateLabels;
	vector<int> trueLabels;

	for (int i=0; i<movieFeatureFiles.size(); i++) {
		int testMEMBER = std::distance(uniqueLabels.begin(), uniqueLabels.find(movieClassIDs[i]));
		vector<cv::Mat> testDs;
		calcSVMTestFeature(pcaWrappers, fvWrappers, movieFeatureFiles[i], delimFeatureFile, recogFrameLength, recogFrameStep, minFeatures, maxFeatures, testDs);
		cout << "Number of images extracted test Fisher Vector feature : " << i << " / " << movieFeatureFiles.size() << endl;

		for (int t=0; t<testDs.size(); t++) {
			// classify
			svm_node* test = new svm_node[testDs[t].cols+1];
			for (int j=0; j<testDs[t].cols; j++) {
				test[j].index = j+1;
				test[j].value = testDs[t].at<float>(j);
			}
			test[testDs[t].cols].index = -1;
			double* prob = new double[uniqueLabels.size()];
			const auto classResult = static_cast<int>(svm_predict_probability(svmModel, test, prob));

			// save correct/wrong number
			if (classResult == testMEMBER){
				cout << "correct answer, ground truth=" << testMEMBER << ", answer=" << classResult << endl;
				correct++;
				correctByClass[testMEMBER]++;
			} else {
				cout << "wrong answer, ground truth=" << testMEMBER << ", answer=" << classResult << endl;
				wrong++;
				wrongByClass[testMEMBER]++;
			}

			// save estimated probability
			vector<double> probvec;
			for (int j=0; j<uniqueLabels.size(); j++) {
				probvec.push_back(prob[j]);
			}
			estimateProbs.push_back(probvec);

			// save estimate label and true label
			estimateLabels.push_back(classResult);
			trueLabels.push_back(testMEMBER);

			delete[] test;
			delete[] prob;
		}
	}

	// output correct/wrong number and statistics
	for (set<string>::iterator it=uniqueLabels.begin(); it!=uniqueLabels.end(); ++it) {
		int member = std::distance(uniqueLabels.begin(), uniqueLabels.find(*it));
		cout << "[result of " << *it << "] correct = " << correctByClass[member] << ", wrong = " << wrongByClass[member] << endl;
		if (ofsLogCSV) {
			ofsLogCSV << "\"" << *it << "\"," << correctByClass[member] << "," << wrongByClass[member] << "," << correctByClass[member]/(double)(correctByClass[member]+wrongByClass[member]) << endl;
		}
	}
	cout << "[total] correct = " << correct << ", wrong = " << wrong << endl;

	double precision = correct/(double)(correct+wrong);
	if (ofsLogCSV) {
		ofsLogCSV << "\"#Correct Num\"," << correct << endl;
		ofsLogCSV << "\"#Wrong Num\"," << wrong << endl;
		ofsLogCSV << "\"#Precision\"," << precision << endl;
	}

	// output estimated probability
	for (int i=0; i<trueLabels.size(); i++) {
		ofsEstProbCSV << trueLabels[i];
		for (int j=0; j<estimateProbs[i].size(); j++) {
			ofsEstProbCSV << "," << estimateProbs[i][j];
		}
		ofsEstProbCSV << endl;
	}

	// output estimated class ID
	for (int i=0; i<trueLabels.size(); i++) {
		ofsEstLabelCSV << trueLabels[i] << "," << estimateLabels[i] << endl;
	}

	return precision;
}
// LibLinear
double testSvmLibLinear(PcaWrapper **pcaWrappers, FisherVectorWrapper **fvWrappers, const vector<string> &movieFeatureFiles, const string &delimFeatureFile, 
						int recogFrameLength, int recogFrameStep, const vector<string> &movieClassIDs, const set<string> &uniqueLabels, 
						vector<float> &minFeatures, vector<float> &maxFeatures, const model *svmModel, ofstream &ofsCSV)
{
	int correct = 0;
	int wrong = 0;
	vector<int> correctByClass(uniqueLabels.size(), 0);
	vector<int> wrongByClass(uniqueLabels.size(), 0);

	for (int i=0; i<movieFeatureFiles.size(); i++) {
		int testMEMBER = std::distance(uniqueLabels.begin(), uniqueLabels.find(movieClassIDs[i]));
		vector<cv::Mat> testDs;
		calcSVMTestFeature(pcaWrappers, fvWrappers, movieFeatureFiles[i], delimFeatureFile, recogFrameLength, recogFrameStep, minFeatures, maxFeatures, testDs);
		cout << "Number of images extracted test Fisher Vector feature : " << i << " / " << movieFeatureFiles.size() << endl;

		for (int t=0; t<testDs.size(); t++) {
			// classify
			feature_node* test = new feature_node[testDs[t].cols+1];
			for (int j=0; j<testDs[t].cols; j++) {
				test[j].index = j+1;
				test[j].value = testDs[t].at<float>(j);
			}
			test[testDs[t].cols].index = -1;
			const auto classResult = static_cast<int>(predict(svmModel, test));
			if (classResult == testMEMBER){
				cout << "correct answer, ground truth=" << testMEMBER << ", answer=" << classResult << endl;
				correct++;
				correctByClass[testMEMBER]++;
			} else {
				cout << "wrong answer, ground truth=" << testMEMBER << ", answer=" << classResult << endl;
				wrong++;
				wrongByClass[testMEMBER]++;
			}
			delete[] test;
		}
	}
	for (set<string>::iterator it=uniqueLabels.begin(); it!=uniqueLabels.end(); ++it) {
		int member = std::distance(uniqueLabels.begin(), uniqueLabels.find(*it));
		cout << "[result of " << *it << "] correct = " << correctByClass[member] << ", wrong = " << wrongByClass[member] << endl;
		if (ofsCSV) {
			ofsCSV << "\"" << *it << "\"," << correctByClass[member] << "," << wrongByClass[member] << "," << correctByClass[member]/(double)(correctByClass[member]+wrongByClass[member]) << endl;
		}
	}
	cout << "[total] correct = " << correct << ", wrong = " << wrong << endl;

	double precision = correct/(double)(correct+wrong);
	if (ofsCSV) {
		ofsCSV << "\"#Correct Num\"," << correct << endl;
		ofsCSV << "\"#Wrong Num\"," << wrong << endl;
		ofsCSV << "\"#Precision\"," << precision << endl;
	}
	return precision;
}

int testMine(string movieFeatureListFile, int recogFrameLength, int recogFrameStep, string outputDir) {
	const string delimFeatureFile = ",";

	vector<string> allMovieFeatureFiles;
	vector<string> allMovieGestureIDs;

	ifstream fs;
	fs.open(movieFeatureListFile);
	string line;
	while (fs>>line) {
		if (line.size()>0) {
			vector<string> tokens = StringUtils::splitString(line, delimFeatureFile);
			CV_Assert(tokens.size()==2);

			vector<vector<cv::Mat> > movieFeatures = readMovieFeature(tokens[0], delimFeatureFile);
			if (movieFeatures.size()>recogFrameLength) {
				allMovieFeatureFiles.push_back(tokens[0]);
				allMovieGestureIDs.push_back(tokens[1]);
			} else {
				cout << "Skip read too short movie : " << tokens[0] << endl;
			}
		}
	}
	CV_Assert(allMovieFeatureFiles.size()>0);

	set<string> uniqueMovieGestureIDs;
	for (int i=0; i<allMovieGestureIDs.size(); i++) {
		set<string>::iterator uniqueMovieGestureIDsIter = uniqueMovieGestureIDs.find(allMovieGestureIDs[i]);
		if(uniqueMovieGestureIDsIter == uniqueMovieGestureIDs.end()) {
			uniqueMovieGestureIDs.insert(allMovieGestureIDs[i]);
		}
	}

	int featureTypeNum;
	{
		vector<vector<cv::Mat> > movieFeature = readMovieFeature(allMovieFeatureFiles[0], delimFeatureFile);
		CV_Assert(movieFeature.size()>0);
		featureTypeNum = movieFeature[0].size();
	}
	CV_Assert(featureTypeNum==1 || featureTypeNum==2);
	cout << "number of movies : " << allMovieFeatureFiles.size() << endl;
	cout << "number of unique gesture IDs : " << uniqueMovieGestureIDs.size() << endl;
	cout << "number of feature types : " << featureTypeNum << endl;

	stringstream ss;
	ss << outputDir << "\\" << "all-estimate-prob.txt";
	ofstream ofsEstProbCSV(ss.str(), ios_base::out);

	ss.str("");
	ss << outputDir << "\\" << "all-estimate-label.txt";
	ofstream ofsEstLabelCSV(ss.str(), ios_base::out);

	vector<double> precisions;
	for (int trialCount=0; trialCount<NUMBER_REPEAT_TEST; trialCount++) {
		cout << "Start " << trialCount << " the test" << endl;

		vector<string> trainMovieFeatureFiles, testMovieFeatureFiles;
		vector<string> trainMovieGestureIDs, testMovieGestureIDs;
		for(set<string>::iterator iter=uniqueMovieGestureIDs.begin(); iter!=uniqueMovieGestureIDs.end(); ++iter){
			string uniqueID = *iter;

			int numMoviePerClass = 0;
			for (int i=0; i<allMovieGestureIDs.size(); i++) {
				if (allMovieGestureIDs[i]==uniqueID) numMoviePerClass++;
			}
			int numTestMoviePerClass = numMoviePerClass/NUMBER_REPEAT_TEST;

			int countUniqueID = 0;
			int countTrainMovie = 0;
			int countTestMovie = 0;
			for (int i=0; i<allMovieFeatureFiles.size(); i++) {
				if (allMovieGestureIDs[i]==uniqueID) {
					if (countUniqueID>=numTestMoviePerClass*trialCount && countUniqueID<numTestMoviePerClass*(trialCount+1)) {
						testMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
						testMovieGestureIDs.push_back(allMovieGestureIDs[i]);
						countTestMovie++;
					} else {
						trainMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
						trainMovieGestureIDs.push_back(allMovieGestureIDs[i]);
						countTrainMovie++;
					}
					countUniqueID++;
				}
			}
			cout << "For class ID " << uniqueID << ", number of train movie files : " << countTrainMovie << endl;
			cout << "For class ID " << uniqueID << ", number of test movie files : " << countTestMovie << endl;
		}
		cout << "Total number of train movie files : " << trainMovieFeatureFiles.size() << endl;
		cout << "Total number of test movie files : " << testMovieFeatureFiles.size() << endl;

		ss.str("");
		ss << outputDir << "\\" << trialCount << "-scale.txt";
		string outputScaleFile = ss.str();

		ss.str("");
		ss << outputDir << "\\" << trialCount << "-svm.txt";
		string outputModelFile = ss.str();

		ss.str("");
		ss << outputDir << "\\" << trialCount << "-log.txt";
		string outputCSV = ss.str();

		cout << "Start getting features to train PCA...." << endl;
		vector<cv::Mat> pcaTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, PCA_TRAIN_FEATURE_NUM, PCA_TRAIN_FEATURE_NUM_PER_IMAGE);
		CV_Assert(pcaTrainFeatures.size()>0);
		cout << "End getting features to train PCA. Training feature matrix size is (" << pcaTrainFeatures[0].rows << " x " << pcaTrainFeatures[0].cols << ")" << endl;

		cout << "Start train PCA...." << endl;
		PcaWrapper **pcaWrappers = new PcaWrapper*[featureTypeNum];
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start train PCA : feature type " << ftype << " ...." << endl;
			pcaWrappers[ftype] = new PcaWrapper(pcaTrainFeatures[ftype], FV_PCA_DIM);
			cout << "End train PCA : feature type " << ftype << "." << endl;
		}
		cout << "End train PCA." << endl;

		cout << "Start save PCA model...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start save PCA model : feature type " << ftype << "...." << endl;
			ss.str("");
			ss << outputDir << "\\" << trialCount << "-pca-" << ftype << ".txt";
			string outputPcaFile = ss.str();

			FileStorage fsWrite(outputPcaFile, FileStorage::WRITE);
			pcaWrappers[ftype]->write(fsWrite);
			fsWrite.release();
			cout << "End save PCA model : feature type " << ftype << "." << endl;
		}
		cout << "End save PCA model." << endl;

		cout << "Reload PCA model from file...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Reload PCA model from file : feature type " << ftype << " ...." << endl;
			delete pcaWrappers[ftype];

			ss.str("");
			ss << outputDir << "\\" << trialCount << "-pca-" << ftype << ".txt";
			string outputPcaFile = ss.str();

			FileStorage fsRead(outputPcaFile, FileStorage::READ);
			pcaWrappers[ftype] = new PcaWrapper();
			pcaWrappers[ftype]->read(fsRead.root());
			fsRead.release();
			cout << "End reload PCA model from file : feature type " << ftype << "." << endl;
		}
		cout << "End reload PCA model from file." << endl;

		cout << "Start getting features to train Fisher Vector...." << endl;
		vector<cv::Mat> fvTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, FV_TRAIN_FEATURE_NUM, FV_TRAIN_FEATURE_NUM_PER_IMAGE);
		CV_Assert(fvTrainFeatures.size()>0);
		cout << "End getting features to train Fisher Vector. Training feature matrix size is (" << fvTrainFeatures[0].rows << " x " << fvTrainFeatures[0].cols << ")" << endl;

		cout << "Start PCA projection of train Fisher Vector...." << endl;
		vector<cv::Mat> pcaFvTrainFeatures(featureTypeNum);
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start PCA projection of train Fisher Vector : feature type " << ftype 
				<< ", feature dim : " << fvTrainFeatures[ftype].cols << ", pca dim : " << FV_PCA_DIM << " ...." << endl;
			pcaFvTrainFeatures[ftype] = pcaWrappers[ftype]->calcPcaProject(fvTrainFeatures[ftype]);
			cout << "End PCA projection of train Fisher Vector : feature type " << ftype << " ...." << endl;
		}
		cout << "End PCA projection of train Fisher Vector...." << endl;

		cout << "Start train Fisher Vector...." << endl;
		FisherVectorWrapper **fvWrappers = new FisherVectorWrapper*[featureTypeNum];
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start train Fisher Vector : feature type " << ftype << " ...." << endl;
			fvWrappers[ftype] = new FisherVectorWrapper(pcaFvTrainFeatures[ftype], FV_K, NORM_FISHER_VECTOR_FEATURE_TYPE);
			cout << "End train Fisher Vector : feature type " << ftype << "." << endl;
		}
		cout << "End train Fisher Vector." << endl;

		cout << "Start save Fisher Vector model...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start save Fisher Vector model : feature type " << ftype << " ...." << endl;
			ss.str("");
			ss << outputDir << "\\" << trialCount << "-fisher-" << ftype << ".txt";
			string outputFisherFile = ss.str();

			FileStorage fsWrite(outputFisherFile, FileStorage::WRITE);
			fvWrappers[ftype]->write(fsWrite);
			fsWrite.release();
			cout << "End save Fisher Vector model : feature type " << ftype << "." << endl;
		}
		cout << "End save Fisher Vector model." << endl;

		cout << "Reload Fisher Vector model from file...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Reload Fisher Vector model from file : feature type " << ftype << " ...." << endl;
			delete fvWrappers[ftype];

			ss.str("");
			ss << outputDir << "\\" << trialCount << "-fisher-" << ftype << ".txt";
			string outputFisherFile = ss.str();

			FileStorage fsRead(outputFisherFile, FileStorage::READ);
			fvWrappers[ftype] = new FisherVectorWrapper();
			fvWrappers[ftype]->read(fsRead.root());
			fsRead.release();
			cout << "End reload Fisher Vector model from file : feature type " << ftype << "." << endl;
		}
		cout << "End reload Fisher Vector model from file." << endl;

		cout << "Start train SVM...." << endl;
		vector<float> minFeatures;
		vector<float> maxFeatures;
		// LibSVM
		trainSvmLibSVM(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
					uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
		SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
		svm_model* svm = svm_load_model(outputModelFile.c_str());
		// LibLinear
		/*
		trainSvmLibLinear(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
						uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
		SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
		model* svm = load_model(outputModelFile.c_str());
		*/
		cout << "End train SVM." << endl;

		cout << "Start classify test images..." << endl;
		ofstream ofsLogCSV(outputCSV, ios_base::out);
		saveParametersToOutputCSV(ofsLogCSV);
		ofsLogCSV << "\"Category Name\",\"Correct\",\"Wrong\",\"Recognition Rate\"" << endl;
		// LibSVM
		double precision = testSvmLibSVM(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
										uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV, ofsEstProbCSV, ofsEstLabelCSV);
		// LibLinear
		/*
		double precision = testSvmLibLinear(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
											uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV);
		*/
		cout << "End classify test images." << endl;
		precisions.push_back(precision);
	}
	double mean = 0.0;
	for (int i=0; i<precisions.size(); i++) {
		mean += precisions[i];
	}
	mean =  mean/(double)precisions.size();
	double stdev = 0.0;
	for (int i=0; i<precisions.size(); i++) {
		stdev += (precisions[i]-mean)*(precisions[i]-mean);
	}
	stdev = sqrt(stdev/(double)(precisions.size()-1));

	ss.str("");
	ss << outputDir << "\\all-log.txt";
	ofstream ofsAllLogCSV(ss.str(), ios_base::out);
	saveParametersToOutputCSV(ofsAllLogCSV);
	ofsAllLogCSV << "\"Trial\",\"Precision\"" << endl;
	for (int t=0; t<precisions.size(); t++) {
		ofsAllLogCSV << (t+1) << "," << precisions[t] << endl;
	}
	ofsAllLogCSV << "#Mean Precision," << mean << endl;
	ofsAllLogCSV << "#Stdev Precision," << stdev << endl;

	return 1;
}

int testGtea(string movieFeatureListFile, int recogFrameLength, int recogFrameStep, string outputDir) {
	const string delimFeatureFile = "\t";

	vector<string> allMovieFeatureFiles;
	vector<string> allMovieGestureIDs;
	vector<string> allMovieSubjects;
	{
		vector<vector<string> > tokens;
		FileUtils::readTSV(movieFeatureListFile, tokens);

		vector<string> movieFeatureFiles;
		vector<string> movieFiles;
		vector<string> movieSubjects;
		vector<string> movieFoods;
		vector<string> movieActions;
		vector<string> movieObjects;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==7);

			movieFeatureFiles.push_back(tokens[i][0]);
			movieFiles.push_back(tokens[i][1]);
			movieSubjects.push_back(tokens[i][2]);
			movieFoods.push_back(tokens[i][3]);
			movieActions.push_back(tokens[i][4]);
			movieObjects.push_back(tokens[i][5]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][6]));
		}

		for (int i=0; i<movieFeatureFiles.size(); i++) {
			vector<vector<cv::Mat> > movieFeatures = readMovieFeature(movieFeatureFiles[i], delimFeatureFile);
			if (movieFeatures.size()>recogFrameLength) {
				allMovieFeatureFiles.push_back(movieFeatureFiles[i]);
				allMovieGestureIDs.push_back(movieActions[i]);
				allMovieSubjects.push_back(movieSubjects[i]);
			} else {
				cout << "Skip read too short movie : " << movieFeatureFiles[i] << endl;
			}
		}
	}
	CV_Assert(allMovieFeatureFiles.size()>0);

	set<string> uniqueMovieGestureIDs;
	for (int i=0; i<allMovieGestureIDs.size(); i++) {
		set<string>::iterator uniqueMovieGestureIDsIter = uniqueMovieGestureIDs.find(allMovieGestureIDs[i]);
		if(uniqueMovieGestureIDsIter == uniqueMovieGestureIDs.end()) {
			uniqueMovieGestureIDs.insert(allMovieGestureIDs[i]);
		}
	}

	int featureTypeNum;
	{
		vector<vector<cv::Mat> > movieFeature = readMovieFeature(allMovieFeatureFiles[0], delimFeatureFile);
		CV_Assert(movieFeature.size()>0);
		featureTypeNum = movieFeature[0].size();
	}

	set<string> uniqueSubjects;
	for (int i=0; i<allMovieSubjects.size(); i++) {
		set<string>::iterator uniqueSubjectsIter = uniqueSubjects.find(allMovieSubjects[i]);
		if(uniqueSubjectsIter == uniqueSubjects.end()) {
			uniqueSubjects.insert(allMovieSubjects[i]);
		}
	}
	cout << "number of movies : " << allMovieFeatureFiles.size() << endl;
	cout << "number of unique gesture IDs : " << uniqueMovieGestureIDs.size() << endl;
	cout << "number of feature types : " << featureTypeNum << endl;
	cout << "number of unique subjects : " << uniqueSubjects.size() << endl;

	stringstream ss;
	ss << outputDir << "\\" << "all-estimate-prob.txt";
	ofstream ofsEstProbCSV(ss.str(), ios_base::out);

	ss.str("");
	ss << outputDir << "\\" << "all-estimate-label.txt";
	ofstream ofsEstLabelCSV(ss.str(), ios_base::out);

	string testSubject = GTEA_TEST_SUBJECT;
	cout << "Start test for subject : " << testSubject << endl;

	vector<string> trainMovieFeatureFiles, testMovieFeatureFiles;
	vector<string> trainMovieGestureIDs, testMovieGestureIDs;
	for (int i=0; i<allMovieFeatureFiles.size(); i++) {
		if (allMovieSubjects[i]==testSubject) {
			testMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
			testMovieGestureIDs.push_back(allMovieGestureIDs[i]);
		} else {
			trainMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
			trainMovieGestureIDs.push_back(allMovieGestureIDs[i]);
		}
	}
	cout << "Total number of train movie files : " << trainMovieFeatureFiles.size() << endl;
	cout << "Total number of test movie files : " << testMovieFeatureFiles.size() << endl;

	ss.str("");
	ss << outputDir << "\\" << testSubject << "-scale.txt";
	string outputScaleFile = ss.str();

	ss.str("");
	ss << outputDir << "\\" << testSubject << "-svm.txt";
	string outputModelFile = ss.str();

	ss.str("");
	ss << outputDir << "\\" << testSubject << "-log.txt";
	string outputCSV = ss.str();

	cout << "Start getting features to train PCA...." << endl;
	vector<cv::Mat> pcaTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, PCA_TRAIN_FEATURE_NUM, PCA_TRAIN_FEATURE_NUM_PER_IMAGE);
	CV_Assert(pcaTrainFeatures.size()>0);
	cout << "End getting features to train PCA. Training feature matrix size is (" << pcaTrainFeatures[0].rows << " x " << pcaTrainFeatures[0].cols << ")" << endl;

	cout << "Start train PCA...." << endl;
	PcaWrapper **pcaWrappers = new PcaWrapper*[featureTypeNum];
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		cout << "Start train PCA : feature type " << ftype << " ...." << endl;
		pcaWrappers[ftype] = new PcaWrapper(pcaTrainFeatures[ftype], FV_PCA_DIM);
		cout << "End train PCA : feature type " << ftype << "." << endl;
	}
	cout << "End train PCA." << endl;

	cout << "Start save PCA model...." << endl;
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		cout << "Start save PCA model : feature type " << ftype << "...." << endl;
		ss.str("");
		ss << outputDir << "\\" << testSubject << "-pca-" << ftype << ".txt";
		string outputPcaFile = ss.str();

		FileStorage fsWrite(outputPcaFile, FileStorage::WRITE);
		pcaWrappers[ftype]->write(fsWrite);
		fsWrite.release();
		cout << "End save PCA model : feature type " << ftype << "." << endl;
	}
	cout << "End save PCA model." << endl;

	cout << "Reload PCA model from file...." << endl;
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		cout << "Reload PCA model from file : feature type " << ftype << " ...." << endl;
		delete pcaWrappers[ftype];

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-pca-" << ftype << ".txt";
		string outputPcaFile = ss.str();

		FileStorage fsRead(outputPcaFile, FileStorage::READ);
		pcaWrappers[ftype] = new PcaWrapper();
		pcaWrappers[ftype]->read(fsRead.root());
		fsRead.release();
		cout << "End reload PCA model from file : feature type " << ftype << "." << endl;
	}
	cout << "End reload PCA model from file." << endl;

	cout << "Start getting features to train Fisher Vector...." << endl;
	vector<cv::Mat> fvTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, FV_TRAIN_FEATURE_NUM, FV_TRAIN_FEATURE_NUM_PER_IMAGE);
	CV_Assert(fvTrainFeatures.size()>0);
	cout << "End getting features to train Fisher Vector. Training feature matrix size is (" << fvTrainFeatures[0].rows << " x " << fvTrainFeatures[0].cols << ")" << endl;

	cout << "Start PCA projection of train Fisher Vector...." << endl;
	vector<cv::Mat> pcaFvTrainFeatures(featureTypeNum);
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		cout << "Start PCA projection of train Fisher Vector : feature type " << ftype << " ...." << endl;
		pcaFvTrainFeatures[ftype] = pcaWrappers[ftype]->calcPcaProject(fvTrainFeatures[ftype]);
		cout << "End PCA projection of train Fisher Vector : feature type " << ftype << " ...." << endl;
	}
	cout << "End PCA projection of train Fisher Vector...." << endl;

	cout << "Start train Fisher Vector...." << endl;
	FisherVectorWrapper **fvWrappers = new FisherVectorWrapper*[featureTypeNum];
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		cout << "Start train Fisher Vector : feature type " << ftype << " ...." << endl;
		fvWrappers[ftype] = new FisherVectorWrapper(pcaFvTrainFeatures[ftype], FV_K, NORM_FISHER_VECTOR_FEATURE_TYPE);
		cout << "End train Fisher Vector : feature type " << ftype << "." << endl;
	}
	cout << "End train Fisher Vector." << endl;

	cout << "Start save Fisher Vector model...." << endl;
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		cout << "Start save Fisher Vector model : feature type " << ftype << " ...." << endl;
		ss.str("");
		ss << outputDir << "\\" << testSubject << "-fisher-" << ftype << ".txt";
		string outputFisherFile = ss.str();

		FileStorage fsWrite(outputFisherFile, FileStorage::WRITE);
		fvWrappers[ftype]->write(fsWrite);
		fsWrite.release();
		cout << "End save Fisher Vector model : feature type " << ftype << "." << endl;
	}
	cout << "End save Fisher Vector model." << endl;

	cout << "Reload Fisher Vector model from file...." << endl;
	for (int ftype=0; ftype<featureTypeNum; ftype++) {
		cout << "Reload Fisher Vector model from file : feature type " << ftype << " ...." << endl;
		delete fvWrappers[ftype];

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-fisher-" << ftype << ".txt";
		string outputFisherFile = ss.str();

		FileStorage fsRead(outputFisherFile, FileStorage::READ);
		fvWrappers[ftype] = new FisherVectorWrapper();
		fvWrappers[ftype]->read(fsRead.root());
		fsRead.release();
		cout << "End reload Fisher Vector model from file : feature type " << ftype << "." << endl;
	}
	cout << "End reload Fisher Vector model from file." << endl;

	cout << "Start train SVM...." << endl;
	vector<float> minFeatures;
	vector<float> maxFeatures;
	// LibSVM
	trainSvmLibSVM(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
				uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
	SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
	svm_model* svm = svm_load_model(outputModelFile.c_str());
	// LibLinear
	/*
	trainSvmLibLinear(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
					uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
	SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
	model* svm = load_model(outputModelFile.c_str());
	*/
	cout << "End train SVM." << endl;

	cout << "Start classify test images..." << endl;
	ofstream ofsLogCSV(outputCSV, ios_base::out);
	saveParametersToOutputCSV(ofsLogCSV);
	ofsLogCSV << "\"Category Name\",\"Correct\",\"Wrong\",\"Recognition Rate\"" << endl;
	// LibSVM
	double precision = testSvmLibSVM(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
									uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV, ofsEstProbCSV, ofsEstLabelCSV);
	// LibLinear
	/*
	double precision = testSvmLibLinear(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
										uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV);
	*/
	cout << "End classify test images." << endl;

	return 1;
}

int testGteaGaze(string movieFeatureListFile, int recogFrameLength, int recogFrameStep, string outputDir, bool testActionVerbObject=false) {
	const string delimFeatureFile = "\t";

	vector<string> allMovieFeatureFiles;
	vector<string> allMovieGestureIDs;
	vector<string> allMovieSubjects;
	{
		vector<vector<string> > tokens;
		FileUtils::readTSV(movieFeatureListFile, tokens);

		vector<string> movieFeatureFiles;
		vector<string> movieFiles;
		vector<string> movieSubjects;
		vector<string> movieFoods;
		vector<string> movieActionVerbs;
		vector<string> movieActionVerbObjects;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==7);

			movieFeatureFiles.push_back(tokens[i][0]);
			movieFiles.push_back(tokens[i][1]);
			movieSubjects.push_back(tokens[i][2]);
			movieFoods.push_back(tokens[i][3]);
			movieActionVerbs.push_back(tokens[i][4]);
			movieActionVerbObjects.push_back(tokens[i][5]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][6]));
		}

		for (int i=0; i<movieFeatureFiles.size(); i++) {
			vector<vector<cv::Mat> > movieFeatures = readMovieFeature(movieFeatureFiles[i], delimFeatureFile);
			if (movieFeatures.size()>recogFrameLength) {
				if (testActionVerbObject) {
					if ("none"==movieActionVerbObjects[i]) {
						// skip 
					} else {
						allMovieFeatureFiles.push_back(movieFeatureFiles[i]);
						allMovieGestureIDs.push_back(movieActionVerbObjects[i]);
						allMovieSubjects.push_back(movieSubjects[i]);
					}
				} else {
					allMovieFeatureFiles.push_back(movieFeatureFiles[i]);
					allMovieGestureIDs.push_back(movieActionVerbs[i]);
					allMovieSubjects.push_back(movieSubjects[i]);
				}
			} else {
				cout << "Skip read too short movie : " << movieFeatureFiles[i] << endl;
			}
		}
	}
	CV_Assert(allMovieFeatureFiles.size()>0);

	set<string> uniqueMovieGestureIDs;
	for (int i=0; i<allMovieGestureIDs.size(); i++) {
		set<string>::iterator uniqueMovieGestureIDsIter = uniqueMovieGestureIDs.find(allMovieGestureIDs[i]);
		if(uniqueMovieGestureIDsIter == uniqueMovieGestureIDs.end()) {
			uniqueMovieGestureIDs.insert(allMovieGestureIDs[i]);
		}
	}

	int featureTypeNum;
	{
		vector<vector<cv::Mat> > movieFeature = readMovieFeature(allMovieFeatureFiles[0], delimFeatureFile);
		CV_Assert(movieFeature.size()>0);
		featureTypeNum = movieFeature[0].size();
	}

	set<string> uniqueSubjects;
	for (int i=0; i<allMovieSubjects.size(); i++) {
		set<string>::iterator uniqueSubjectsIter = uniqueSubjects.find(allMovieSubjects[i]);
		if(uniqueSubjectsIter == uniqueSubjects.end()) {
			uniqueSubjects.insert(allMovieSubjects[i]);
		}
	}
	cout << "number of movies : " << allMovieFeatureFiles.size() << endl;
	cout << "number of unique gesture IDs : " << uniqueMovieGestureIDs.size() << endl;
	cout << "number of feature types : " << featureTypeNum << endl;
	cout << "number of unique subjects : " << uniqueSubjects.size() << endl;

	stringstream ss;
	ss << outputDir << "\\" << "all-estimate-prob.txt";
	ofstream ofsEstProbCSV(ss.str(), ios_base::out);

	ss.str("");
	ss << outputDir << "\\" << "all-estimate-label.txt";
	ofstream ofsEstLabelCSV(ss.str(), ios_base::out);

	vector<double> precisions;
	for(set<string>::iterator iter=uniqueSubjects.begin(); iter!=uniqueSubjects.end(); ++iter){
		string testSubject = *iter;
		cout << "Start test for subject : " << testSubject << endl;

		vector<string> trainMovieFeatureFiles, testMovieFeatureFiles;
		vector<string> trainMovieGestureIDs, testMovieGestureIDs;
		for (int i=0; i<allMovieFeatureFiles.size(); i++) {
			if (allMovieSubjects[i]==testSubject) {
				testMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
				testMovieGestureIDs.push_back(allMovieGestureIDs[i]);
			} else {
				trainMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
				trainMovieGestureIDs.push_back(allMovieGestureIDs[i]);
			}
		}
		cout << "Total number of train movie files : " << trainMovieFeatureFiles.size() << endl;
		cout << "Total number of test movie files : " << testMovieFeatureFiles.size() << endl;

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-scale.txt";
		string outputScaleFile = ss.str();

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-svm.txt";
		string outputModelFile = ss.str();

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-log.txt";
		string outputCSV = ss.str();

		cout << "Start getting features to train PCA...." << endl;
		vector<cv::Mat> pcaTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, PCA_TRAIN_FEATURE_NUM, PCA_TRAIN_FEATURE_NUM_PER_IMAGE);
		CV_Assert(pcaTrainFeatures.size()>0);
		cout << "End getting features to train PCA. Training feature matrix size is (" << pcaTrainFeatures[0].rows << " x " << pcaTrainFeatures[0].cols << ")" << endl;

		cout << "Start train PCA...." << endl;
		PcaWrapper **pcaWrappers = new PcaWrapper*[featureTypeNum];
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start train PCA : feature type " << ftype << " ...." << endl;
			pcaWrappers[ftype] = new PcaWrapper(pcaTrainFeatures[ftype], FV_PCA_DIM);
			cout << "End train PCA : feature type " << ftype << "." << endl;
		}
		cout << "End train PCA." << endl;

		cout << "Start save PCA model...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start save PCA model : feature type " << ftype << "...." << endl;
			ss.str("");
			ss << outputDir << "\\" << testSubject << "-pca-" << ftype << ".txt";
			string outputPcaFile = ss.str();

			FileStorage fsWrite(outputPcaFile, FileStorage::WRITE);
			pcaWrappers[ftype]->write(fsWrite);
			fsWrite.release();
			cout << "End save PCA model : feature type " << ftype << "." << endl;
		}
		cout << "End save PCA model." << endl;

		cout << "Reload PCA model from file...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Reload PCA model from file : feature type " << ftype << " ...." << endl;
			delete pcaWrappers[ftype];

			ss.str("");
			ss << outputDir << "\\" << testSubject << "-pca-" << ftype << ".txt";
			string outputPcaFile = ss.str();

			FileStorage fsRead(outputPcaFile, FileStorage::READ);
			pcaWrappers[ftype] = new PcaWrapper();
			pcaWrappers[ftype]->read(fsRead.root());
			fsRead.release();
			cout << "End reload PCA model from file : feature type " << ftype << "." << endl;
		}
		cout << "End reload PCA model from file." << endl;

		cout << "Start getting features to train Fisher Vector...." << endl;
		vector<cv::Mat> fvTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, FV_TRAIN_FEATURE_NUM, FV_TRAIN_FEATURE_NUM_PER_IMAGE);
		CV_Assert(fvTrainFeatures.size()>0);
		cout << "End getting features to train Fisher Vector. Training feature matrix size is (" << fvTrainFeatures[0].rows << " x " << fvTrainFeatures[0].cols << ")" << endl;

		cout << "Start PCA projection of train Fisher Vector...." << endl;
		vector<cv::Mat> pcaFvTrainFeatures(featureTypeNum);
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start PCA projection of train Fisher Vector : feature type " << ftype << " ...." << endl;
			pcaFvTrainFeatures[ftype] = pcaWrappers[ftype]->calcPcaProject(fvTrainFeatures[ftype]);
			cout << "End PCA projection of train Fisher Vector : feature type " << ftype << " ...." << endl;
		}
		cout << "End PCA projection of train Fisher Vector...." << endl;

		cout << "Start train Fisher Vector...." << endl;
		FisherVectorWrapper **fvWrappers = new FisherVectorWrapper*[featureTypeNum];
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start train Fisher Vector : feature type " << ftype << " ...." << endl;
			fvWrappers[ftype] = new FisherVectorWrapper(pcaFvTrainFeatures[ftype], FV_K, NORM_FISHER_VECTOR_FEATURE_TYPE);
			cout << "End train Fisher Vector : feature type " << ftype << "." << endl;
		}
		cout << "End train Fisher Vector." << endl;

		cout << "Start save Fisher Vector model...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start save Fisher Vector model : feature type " << ftype << " ...." << endl;
			ss.str("");
			ss << outputDir << "\\" << testSubject << "-fisher-" << ftype << ".txt";
			string outputFisherFile = ss.str();

			FileStorage fsWrite(outputFisherFile, FileStorage::WRITE);
			fvWrappers[ftype]->write(fsWrite);
			fsWrite.release();
			cout << "End save Fisher Vector model : feature type " << ftype << "." << endl;
		}
		cout << "End save Fisher Vector model." << endl;

		cout << "Reload Fisher Vector model from file...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Reload Fisher Vector model from file : feature type " << ftype << " ...." << endl;
			delete fvWrappers[ftype];

			ss.str("");
			ss << outputDir << "\\" << testSubject << "-fisher-" << ftype << ".txt";
			string outputFisherFile = ss.str();

			FileStorage fsRead(outputFisherFile, FileStorage::READ);
			fvWrappers[ftype] = new FisherVectorWrapper();
			fvWrappers[ftype]->read(fsRead.root());
			fsRead.release();
			cout << "End reload Fisher Vector model from file : feature type " << ftype << "." << endl;
		}
		cout << "End reload Fisher Vector model from file." << endl;

		cout << "Start train SVM...." << endl;
		vector<float> minFeatures;
		vector<float> maxFeatures;
		// LibSVM
		trainSvmLibSVM(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
					uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
		SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
		svm_model* svm = svm_load_model(outputModelFile.c_str());
		// LibLinear
		/*
		trainSvmLibLinear(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
						uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
		SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
		model* svm = load_model(outputModelFile.c_str());
		*/
		cout << "End train SVM." << endl;

		cout << "Start classify test images..." << endl;
		ofstream ofsLogCSV(outputCSV, ios_base::out);
		saveParametersToOutputCSV(ofsLogCSV);
		ofsLogCSV << "\"Category Name\",\"Correct\",\"Wrong\",\"Recognition Rate\"" << endl;
		// LibSVM
		double precision = testSvmLibSVM(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
										uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV, ofsEstProbCSV, ofsEstLabelCSV);
		// LibLinear
		/*
		double precision = testSvmLibLinear(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
											uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV);
		*/
		cout << "End classify test images." << endl;
		precisions.push_back(precision);
	}
	double mean = 0.0;
	for (int i=0; i<precisions.size(); i++) {
		mean += precisions[i];
	}
	mean =  mean/(double)precisions.size();
	double stdev = 0.0;
	for (int i=0; i<precisions.size(); i++) {
		stdev += (precisions[i]-mean)*(precisions[i]-mean);
	}
	stdev = sqrt(stdev/(double)(precisions.size()-1));

	ss.str("");
	ss << outputDir << "\\all-log.txt";
	ofstream ofsAllLogCSV(ss.str(), ios_base::out);
	saveParametersToOutputCSV(ofsAllLogCSV);
	ofsAllLogCSV << "\"Trial\",\"Precision\"" << endl;
	for (int t=0; t<precisions.size(); t++) {
		ofsAllLogCSV << (t+1) << "," << precisions[t] << endl;
	}
	ofsAllLogCSV << "#Mean Precision," << mean << endl;
	ofsAllLogCSV << "#Stdev Precision," << stdev << endl;

	return 1;
}

int testCMUKitchen(string movieFeatureListFile, int recogFrameLength, int recogFrameStep, string outputDir) {
	const string delimFeatureFile = "\t";

	vector<string> allMovieFeatureFiles;
	vector<string> allMovieGestureIDs;
	vector<string> allMovieSubjects;
	{
		vector<vector<string> > tokens;
		FileUtils::readTSV(movieFeatureListFile, tokens);

		vector<string> movieFeatureFiles;
		vector<string> movieFiles;
		vector<string> movieSubjects;
		vector<string> movieFoods;
		vector<string> movieActions;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==6);

			movieFeatureFiles.push_back(tokens[i][0]);
			movieFiles.push_back(tokens[i][1]);
			movieSubjects.push_back(tokens[i][2]);
			movieFoods.push_back(tokens[i][3]);
			movieActions.push_back(tokens[i][4]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][5]));
		}

		for (int i=0; i<movieFeatureFiles.size(); i++) {
			vector<vector<cv::Mat> > movieFeatures = readMovieFeature(movieFeatureFiles[i], delimFeatureFile);
			if (movieFeatures.size()>recogFrameLength) {
				allMovieFeatureFiles.push_back(movieFeatureFiles[i]);
				allMovieGestureIDs.push_back(movieActions[i]);
				allMovieSubjects.push_back(movieSubjects[i]);
			} else {
				cout << "Skip read too short movie : " << movieFeatureFiles[i] << endl;
			}
		}
	}
	CV_Assert(allMovieFeatureFiles.size()>0);

	set<string> uniqueMovieGestureIDs;
	for (int i=0; i<allMovieGestureIDs.size(); i++) {
		set<string>::iterator uniqueMovieGestureIDsIter = uniqueMovieGestureIDs.find(allMovieGestureIDs[i]);
		if(uniqueMovieGestureIDsIter == uniqueMovieGestureIDs.end()) {
			uniqueMovieGestureIDs.insert(allMovieGestureIDs[i]);
		}
	}

	int featureTypeNum;
	{
		vector<vector<cv::Mat> > movieFeature = readMovieFeature(allMovieFeatureFiles[0], delimFeatureFile);
		CV_Assert(movieFeature.size()>0);
		featureTypeNum = movieFeature[0].size();
	}

	set<string> uniqueSubjects;
	for (int i=0; i<allMovieSubjects.size(); i++) {
		set<string>::iterator uniqueSubjectsIter = uniqueSubjects.find(allMovieSubjects[i]);
		if(uniqueSubjectsIter == uniqueSubjects.end()) {
			uniqueSubjects.insert(allMovieSubjects[i]);
		}
	}
	cout << "number of movies : " << allMovieFeatureFiles.size() << endl;
	cout << "number of unique gesture IDs : " << uniqueMovieGestureIDs.size() << endl;
	cout << "number of feature types : " << featureTypeNum << endl;
	cout << "number of unique subjects : " << uniqueSubjects.size() << endl;

	stringstream ss;
	ss << outputDir << "\\" << "all-estimate-prob.txt";
	ofstream ofsEstProbCSV(ss.str(), ios_base::out);

	ss.str("");
	ss << outputDir << "\\" << "all-estimate-label.txt";
	ofstream ofsEstLabelCSV(ss.str(), ios_base::out);

	vector<double> precisions;
	for(set<string>::iterator iter=uniqueSubjects.begin(); iter!=uniqueSubjects.end(); ++iter){
		string testSubject = *iter;
		cout << "Start test for subject : " << testSubject << endl;

		vector<string> trainMovieFeatureFiles, testMovieFeatureFiles;
		vector<string> trainMovieGestureIDs, testMovieGestureIDs;
		for (int i=0; i<allMovieFeatureFiles.size(); i++) {
			if (allMovieSubjects[i]==testSubject) {
				testMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
				testMovieGestureIDs.push_back(allMovieGestureIDs[i]);
			} else {
				trainMovieFeatureFiles.push_back(allMovieFeatureFiles[i]);
				trainMovieGestureIDs.push_back(allMovieGestureIDs[i]);
			}
		}
		cout << "Total number of train movie files : " << trainMovieFeatureFiles.size() << endl;
		cout << "Total number of test movie files : " << testMovieFeatureFiles.size() << endl;

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-scale.txt";
		string outputScaleFile = ss.str();

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-svm.txt";
		string outputModelFile = ss.str();

		ss.str("");
		ss << outputDir << "\\" << testSubject << "-log.txt";
		string outputCSV = ss.str();

		cout << "Start getting features to train PCA...." << endl;
		vector<cv::Mat> pcaTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, PCA_TRAIN_FEATURE_NUM, PCA_TRAIN_FEATURE_NUM_PER_IMAGE);
		CV_Assert(pcaTrainFeatures.size()>0);
		cout << "End getting features to train PCA. Training feature matrix size is (" << pcaTrainFeatures[0].rows << " x " << pcaTrainFeatures[0].cols << ")" << endl;

		cout << "Start train PCA...." << endl;
		PcaWrapper **pcaWrappers = new PcaWrapper*[featureTypeNum];
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start train PCA : feature type " << ftype << " ...." << endl;
			pcaWrappers[ftype] = new PcaWrapper(pcaTrainFeatures[ftype], FV_PCA_DIM);
			cout << "End train PCA : feature type " << ftype << "." << endl;
		}
		cout << "End train PCA." << endl;

		cout << "Start save PCA model...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start save PCA model : feature type " << ftype << "...." << endl;
			ss.str("");
			ss << outputDir << "\\" << testSubject << "-pca-" << ftype << ".txt";
			string outputPcaFile = ss.str();

			FileStorage fsWrite(outputPcaFile, FileStorage::WRITE);
			pcaWrappers[ftype]->write(fsWrite);
			fsWrite.release();
			cout << "End save PCA model : feature type " << ftype << "." << endl;
		}
		cout << "End save PCA model." << endl;

		cout << "Reload PCA model from file...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Reload PCA model from file : feature type " << ftype << " ...." << endl;
			delete pcaWrappers[ftype];

			ss.str("");
			ss << outputDir << "\\" << testSubject << "-pca-" << ftype << ".txt";
			string outputPcaFile = ss.str();

			FileStorage fsRead(outputPcaFile, FileStorage::READ);
			pcaWrappers[ftype] = new PcaWrapper();
			pcaWrappers[ftype]->read(fsRead.root());
			fsRead.release();
			cout << "End reload PCA model from file : feature type " << ftype << "." << endl;
		}
		cout << "End reload PCA model from file." << endl;

		cout << "Start getting features to train Fisher Vector...." << endl;
		vector<cv::Mat> fvTrainFeatures = getRandomTrainFeatures(trainMovieFeatureFiles, delimFeatureFile, FV_TRAIN_FEATURE_NUM, FV_TRAIN_FEATURE_NUM_PER_IMAGE);
		CV_Assert(fvTrainFeatures.size()>0);
		cout << "End getting features to train Fisher Vector. Training feature matrix size is (" << fvTrainFeatures[0].rows << " x " << fvTrainFeatures[0].cols << ")" << endl;

		cout << "Start PCA projection of train Fisher Vector...." << endl;
		vector<cv::Mat> pcaFvTrainFeatures(featureTypeNum);
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start PCA projection of train Fisher Vector : feature type " << ftype << " ...." << endl;
			pcaFvTrainFeatures[ftype] = pcaWrappers[ftype]->calcPcaProject(fvTrainFeatures[ftype]);
			cout << "End PCA projection of train Fisher Vector : feature type " << ftype << " ...." << endl;
		}
		cout << "End PCA projection of train Fisher Vector...." << endl;

		cout << "Start train Fisher Vector...." << endl;
		FisherVectorWrapper **fvWrappers = new FisherVectorWrapper*[featureTypeNum];
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start train Fisher Vector : feature type " << ftype << " ...." << endl;
			fvWrappers[ftype] = new FisherVectorWrapper(pcaFvTrainFeatures[ftype], FV_K, NORM_FISHER_VECTOR_FEATURE_TYPE);
			cout << "End train Fisher Vector : feature type " << ftype << "." << endl;
		}
		cout << "End train Fisher Vector." << endl;

		cout << "Start save Fisher Vector model...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Start save Fisher Vector model : feature type " << ftype << " ...." << endl;
			ss.str("");
			ss << outputDir << "\\" << testSubject << "-fisher-" << ftype << ".txt";
			string outputFisherFile = ss.str();

			FileStorage fsWrite(outputFisherFile, FileStorage::WRITE);
			fvWrappers[ftype]->write(fsWrite);
			fsWrite.release();
			cout << "End save Fisher Vector model : feature type " << ftype << "." << endl;
		}
		cout << "End save Fisher Vector model." << endl;

		cout << "Reload Fisher Vector model from file...." << endl;
		for (int ftype=0; ftype<featureTypeNum; ftype++) {
			cout << "Reload Fisher Vector model from file : feature type " << ftype << " ...." << endl;
			delete fvWrappers[ftype];

			ss.str("");
			ss << outputDir << "\\" << testSubject << "-fisher-" << ftype << ".txt";
			string outputFisherFile = ss.str();

			FileStorage fsRead(outputFisherFile, FileStorage::READ);
			fvWrappers[ftype] = new FisherVectorWrapper();
			fvWrappers[ftype]->read(fsRead.root());
			fsRead.release();
			cout << "End reload Fisher Vector model from file : feature type " << ftype << "." << endl;
		}
		cout << "End reload Fisher Vector model from file." << endl;

		cout << "Start train SVM...." << endl;
		vector<float> minFeatures;
		vector<float> maxFeatures;
		// LibSVM
		trainSvmLibSVM(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
					uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
		SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
		svm_model* svm = svm_load_model(outputModelFile.c_str());
		// LibLinear
		/*
		trainSvmLibLinear(pcaWrappers, fvWrappers, trainMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, trainMovieGestureIDs, 
						uniqueMovieGestureIDs, outputModelFile, minFeatures, maxFeatures);
		SVMUtils::saveScaleData(SCALE_FEATURE_BEFORE_SVM_MIN, SCALE_FEATURE_BEFORE_SVM_MAX, minFeatures, maxFeatures, outputScaleFile);
		model* svm = load_model(outputModelFile.c_str());
		*/
		cout << "End train SVM." << endl;

		cout << "Start classify test images..." << endl;
		ofstream ofsLogCSV(outputCSV, ios_base::out);
		saveParametersToOutputCSV(ofsLogCSV);
		ofsLogCSV << "\"Category Name\",\"Correct\",\"Wrong\",\"Recognition Rate\"" << endl;
		// LibSVM
		double precision = testSvmLibSVM(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
										uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV, ofsEstProbCSV, ofsEstLabelCSV);
		// LibLinear
		/*
		double precision = testSvmLibLinear(pcaWrappers, fvWrappers, testMovieFeatureFiles, delimFeatureFile, recogFrameLength, recogFrameStep, testMovieGestureIDs, 
											uniqueMovieGestureIDs, minFeatures, maxFeatures, svm, ofsLogCSV);
		*/
		cout << "End classify test images." << endl;
		precisions.push_back(precision);
	}
	double mean = 0.0;
	for (int i=0; i<precisions.size(); i++) {
		mean += precisions[i];
	}
	mean =  mean/(double)precisions.size();
	double stdev = 0.0;
	for (int i=0; i<precisions.size(); i++) {
		stdev += (precisions[i]-mean)*(precisions[i]-mean);
	}
	stdev = sqrt(stdev/(double)(precisions.size()-1));

	ss.str("");
	ss << outputDir << "\\all-log.txt";
	ofstream ofsAllLogCSV(ss.str(), ios_base::out);
	saveParametersToOutputCSV(ofsAllLogCSV);
	ofsAllLogCSV << "\"Trial\",\"Precision\"" << endl;
	for (int t=0; t<precisions.size(); t++) {
		ofsAllLogCSV << (t+1) << "," << precisions[t] << endl;
	}
	ofsAllLogCSV << "#Mean Precision," << mean << endl;
	ofsAllLogCSV << "#Stdev Precision," << stdev << endl;

	return 1;
}

int main (int argc, char * const argv[]) 
{
	if (argc!=3 && argc!=4) {
		cout << "[Usage] ./RecogGestureFisherVector [input movie feature list file] [output dir] [(optional) input database type]" << endl;
		return -1;
	}
	string movieFeatureListFile = argv[1];
	string outputDir = argv[2];
	string inputDatabaseType = "mine";
	if (argc==4) {
		inputDatabaseType = argv[3];
	}
	CV_Assert(inputDatabaseType=="mine" || inputDatabaseType=="gtea" || inputDatabaseType=="gteagaze-verb" || inputDatabaseType=="gteagaze-verb-object" 
			|| inputDatabaseType=="cmu-kitchen");

	if (inputDatabaseType=="mine") {
		testMine(movieFeatureListFile, MYDATASET_RECOG_GESTURE_FRAME_LENGTH, MYDATASET_RECOG_GESTURE_FRAME_STEP, outputDir);
	} else if (inputDatabaseType=="gtea") {
		testGtea(movieFeatureListFile, GTEA_RECOG_GESTURE_FRAME_LENGTH, GTEA_RECOG_GESTURE_FRAME_STEP, outputDir);
	} else if (inputDatabaseType=="gteagaze-verb") {
		testGteaGaze(movieFeatureListFile, GTEAGAZE_ACTION_VERB_RECOG_GESTURE_FRAME_LENGTH, GTEAGAZE_RECOG_GESTURE_FRAME_STEP, outputDir);
	} else if (inputDatabaseType=="gteagaze-verb-object") {
		testGteaGaze(movieFeatureListFile, GTEAGAZE_ACTION_VERB_OBJECT_RECOG_GESTURE_FRAME_LENGTH, GTEAGAZE_RECOG_GESTURE_FRAME_STEP, outputDir, true);
	} else if (inputDatabaseType=="cmu-kitchen") {
		testCMUKitchen(movieFeatureListFile, CMUKITCHEN_RECOG_GESTURE_FRAME_LENGTH, CMUKITCHEN_RECOG_GESTURE_FRAME_STEP, outputDir);
	}
}