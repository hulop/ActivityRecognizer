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
#include <iterator>

#include <opencv2/opencv.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/video/tracking.hpp>

#include "RecogHandGestureConstants.h"
#include "MyDatasetConstants.h"
#include "GTEAConstants.h"
#include "GTEAGazePlusConstants.h"
#include "GTEAGazePlusUtils.h"
#include "CMUKitchenConstants.h"
#include "DenseTrajectory.h"
#include "DenseTrajectoryTrackPoint.h"
#include "VideoStabilizer.h"
#include "HandDetectorWrapper.h"
#include "FileUtils.h"
#include "StringUtils.h"
#include "TimerUtils.h"

using namespace std;
using namespace cv;

static const bool SAVE_VIDEO = false;
static const char* INPUT_VIDEO = "input.avi";
static const char* DENSE_TRAJECTORY_VIDEO = "dense-trajectory.avi";
static const char* HAND_PROBABILITY_VIDEO = "hand-probability.avi";
static const char* HAND_HOG_VIDEO = "hand-hog.avi";

string processMovie(string movieFile, int movieFrameWidth, int movieFrameHeight, int timeIntervalDetectKeypoint, int denseKeyPointStep, string outFeatureDir, string delimFeatureFile, string subject="", 
					map<string, string> handModelMap=map<string, string>(), map<string, string> handGlobalFeatMap=map<string, string>(), int maxFrame=-1)
{
	VideoCapture capture(movieFile);
	if (!capture.isOpened()) {
		cout << "Failed to open video : " << movieFile << endl;
		return "";
	}

	cv::VideoWriter inputVideoWriter;
	cv::VideoWriter dtVideoWriter;
	cv::VideoWriter handProbVideoWriter;
	cv::VideoWriter handHogVideoWriter;

	string movieFileName = FileUtils::getFileName(movieFile);
	string outFeatureFile = outFeatureDir + "\\" + movieFileName + "-dfeat.txt";
	ofstream ofs(outFeatureFile, ios_base::out);

	Size capSize = Size(movieFrameWidth, movieFrameHeight);
	DenseTrajectory denseTrajectory(dtUseTraj, dtUseHOG, dtUseHOF, dtUseMBH, capSize, timeIntervalDetectKeypoint, denseKeyPointStep, dtTrackMedianFilterSize, 
									dtPyramidLevels, dtMaxTimeLength, dtPyramidScale, dtDescGrid2D, dtDescGridTime, dtDescNeighborSize);

	HandDetectorWrapper handDetector;
	VideoStabilizer videoStabilizer;
	if (USE_HAND_MASK_KEYPOINT_REMOVE || USE_HAND_HOG || USE_VIDEO_STABILIZATION) {
		if (subject.empty()) {
			handDetector.testInitialize(MYDATASET_HAND_DETECT_MODEL_LIST_FILE, MYDATASET_HAND_DETECT_GLOBAL_FEATURE_LIST_FILE, HAND_DETECT_FEATURE_SET, 
										HAND_DETECT_NUM_MODELS_TO_AVERAGE, movieFrameWidth);
		} else {
			string handDetectModel = handModelMap[subject];
			string handDetectGlobalFeat = handGlobalFeatMap[subject];
			CV_Assert(!handDetectModel.empty() && !handDetectGlobalFeat.empty());

			handDetector.testInitialize(handDetectModel, handDetectGlobalFeat, HAND_DETECT_FEATURE_SET, 
										HAND_DETECT_NUM_MODELS_TO_AVERAGE, movieFrameWidth);
		}
	}

	Ptr<DescriptorExtractor> siftExtractor = DescriptorExtractor::create("SIFT");
	Ptr<FeatureDetector> siftDetector = FeatureDetector::create("SIFT");
	if (USE_SIFT) {
		siftExtractor = DescriptorExtractor::create("SIFT");
		siftDetector = FeatureDetector::create("SIFT");
	}

	int time = 0;
	int denseTrajDim = 0;
	int handHogDim = 0;
	if (USE_HAND_HOG) {
		handHogDim = handDetector.getDenseHandHOGDim(HAND_HOG_BLOCK_SIZE, HAND_HOG_STRIDE_SIZE, HAND_HOG_CELL_SIZE, HAND_HOG_RAD_BINS);
	}
	int siftDim = 0;
	Mat prevCapFrame;
	vector<Mat> handProbHist;
	Mat handProbAverage;
	Mat handMask;
	while (true) {
		Mat capFrame;
		capture >> capFrame;
		if (capFrame.empty() || (maxFrame>0 && time>maxFrame)) {
			break;
		}
		cv::resize(capFrame, capFrame, capSize);
		imshow("Input Video", capFrame);

		Mat vizHandProb;
		if (USE_HAND_MASK_KEYPOINT_REMOVE || USE_HAND_HOG || USE_VIDEO_STABILIZATION) {
			// process hand detection
			TimerUtils::getInstance().start();

			Mat handDetectFrame = capFrame.clone();
			handDetector.test(handDetectFrame, HAND_DETECT_NUM_MODELS_TO_AVERAGE, HAND_DETECT_STEP_SIZE);

			cout << "time to detect hand : " << TimerUtils::getInstance().stop() << " secs." << endl;

			Mat handProbBlur = handDetector.getResponseImage();
			GaussianBlur(handProbBlur, handProbBlur, cv::Size(3,3), 0, 0, BORDER_REFLECT);
			handProbHist.push_back(handProbBlur);
			if (handProbHist.size()>HAND_DETECT_AVERAGE_PROBABILITY_TIME_LENGTH) {
				handProbHist.erase(handProbHist.begin());
			}
			handProbAverage = Mat::zeros(handProbBlur.size(), handProbBlur.type());
			for (int i=0; i<handProbHist.size(); i++) {
				handProbAverage += handProbHist[i];
			}
			handProbAverage /= handProbHist.size();
			cv::resize(handProbAverage, handProbAverage, capSize);

			handMask = handDetector.postprocess(handProbAverage, HAND_DETECT_PROB_THRES);
			handMask.convertTo(handMask, CV_8U);

			imshow("Hand Probability", handDetector.getResponseImage());
			imshow("Hand Probability Average", handProbAverage);
			imshow("Hand Mask", handMask);

			handDetector.colormap(handProbAverage, vizHandProb, 1);
		}

		Mat frame;
		if (USE_VIDEO_STABILIZATION && !prevCapFrame.empty() && !handMask.empty()) {
			Mat invHandMask = 255 - handMask;
			frame = videoStabilizer.calcStabilizedFrame(capFrame, prevCapFrame, invHandMask);
			imshow("Stabilized Input Video", frame);
		} else {
			frame = capFrame.clone();
		}

		Mat dtFrame;
		if (USE_HAND_MASK_KEYPOINT_REMOVE) {
			dtFrame = denseTrajectory.update(frame, handMask);
		} else {
			dtFrame = denseTrajectory.update(frame, Mat());
		}

		TimerUtils::getInstance().start();

		Mat descriptors = denseTrajectory.getDesc();
		CV_Assert(checkRange(descriptors));
		/*
		if (!checkRange(descriptors)) {
			descriptors = Mat();
		}
		*/

		Mat handHogDescs;
		Mat vizHandHog;
		if (USE_HAND_HOG) {
			// Dense Hand HOG for hand region
			Mat handMask = handDetector.postprocess(handProbAverage, HAND_DETECT_PROB_THRES);
			handMask.convertTo(handMask, CV_8U);
			imshow("Hand Mask for Hand HOG", handMask);

			vizHandHog = handDetector.computeDenseHandHOG(frame, handProbAverage, HAND_HOG_BLOCK_SIZE, HAND_HOG_STRIDE_SIZE, 
														HAND_HOG_CELL_SIZE, HAND_HOG_RAD_BINS, handMask, handHogDescs);
			CV_Assert(checkRange(handHogDescs));
			/*
			if (!checkRange(descriptors)) {
				descriptors = Mat();
			}
			*/
		}

		Mat siftDescs;
		if (USE_SIFT) {
			Mat grayImage;
			cvtColor(frame, grayImage, CV_BGR2GRAY);

			vector<cv::KeyPoint> keypoints;
			siftDetector->detect(grayImage, keypoints);

			Mat keypointImage = frame.clone();
			std::vector<cv::KeyPoint>::iterator itk;
			for (itk = keypoints.begin(); itk != keypoints.end(); ++itk) {
				cv::circle(keypointImage, itk->pt, itk->size, cv::Scalar(0,255,255), 1, CV_AA);
				cv::circle(keypointImage, itk->pt, 1, cv::Scalar(0,255,0), -1);
			}
			cv::imshow("SIFT keypoints", keypointImage);

			siftExtractor->compute(grayImage, keypoints, siftDescs);
			CV_Assert(checkRange(siftDescs));
		}

		cout << "time to get descriptors : " << TimerUtils::getInstance().stop() << " secs." << endl;

		cout << "movie time frame : " << time << ", descriptor num=" << descriptors.rows << ", descriptor dim=" << descriptors.cols << endl;

		if (descriptors.empty() && denseTrajDim>0) {
			descriptors = Mat::zeros(1, denseTrajDim, CV_32F);
		}
		if (USE_HAND_HOG && handHogDescs.empty() && handHogDim>0) {
			handHogDescs = Mat::zeros(1, handHogDim, CV_32F);
		}
		if (USE_SIFT && siftDescs.empty() && siftDim>0) {
			siftDescs = Mat::zeros(1, siftDim, CV_32F);
		}

		if (time>=dtMaxTimeLength && (!USE_HAND_HOG && !descriptors.empty()) || (USE_HAND_HOG && !descriptors.empty() && !handHogDescs.empty())) {
			CV_Assert(denseTrajDim==0 || denseTrajDim==descriptors.cols);
			
			stringstream ss;
			ss << time;
			string denseTrajDescMatFile = outFeatureDir + "\\" + movieFileName + "-dfeat-" + ss.str() + ".yml";
			//FileUtils::saveMat(denseTrajDescMatFile.c_str(), descriptors);
			FileUtils::saveMatBin(denseTrajDescMatFile.c_str(), descriptors);
			ofs << denseTrajDescMatFile << delimFeatureFile;
			
			if (USE_HAND_HOG) {
				CV_Assert(handHogDim==0 || handHogDim==handHogDescs.cols);

				string handHogDescMatFile = outFeatureDir + "\\" + movieFileName + "-handhog-" + ss.str() + ".yml";
				//FileUtils::saveMat(handHogDescMatFile.c_str(), handHogDescs);
				FileUtils::saveMatBin(handHogDescMatFile.c_str(), handHogDescs);
				ofs << handHogDescMatFile << delimFeatureFile;
			}

			if (USE_SIFT) {
				string siftDescMatFile = outFeatureDir + "\\" + movieFileName + "-sift-" + ss.str() + ".yml";
				FileUtils::saveMatBin(siftDescMatFile.c_str(), siftDescs);
				ofs << siftDescMatFile << delimFeatureFile;
			}

			ofs << time << endl;

			denseTrajDim = descriptors.cols;
			handHogDim = handHogDescs.cols;
			siftDim = siftDescs.cols;
		}

		if (SAVE_VIDEO) {
			int codec = CV_FOURCC('M', 'J', 'P', 'G');
			if (!inputVideoWriter.isOpened()) {
				inputVideoWriter.open(outFeatureDir + "\\" + INPUT_VIDEO, codec, 15.0, frame.size(), true);
			}
			if (!dtVideoWriter.isOpened()) {
				dtVideoWriter.open(outFeatureDir + "\\" + DENSE_TRAJECTORY_VIDEO, codec, 15.0, dtFrame.size(), true);
			}
			if (!handProbVideoWriter.isOpened()) {
				handProbVideoWriter.open(outFeatureDir + "\\" + HAND_PROBABILITY_VIDEO, codec, 15.0, vizHandProb.size(), true);
			}
			if (!handHogVideoWriter.isOpened()) {
				handHogVideoWriter.open(outFeatureDir + "\\" + HAND_HOG_VIDEO, codec, 15.0, vizHandHog.size(), true);
			}

			inputVideoWriter << frame;
			dtVideoWriter << dtFrame;
			handProbVideoWriter << vizHandProb;
			handHogVideoWriter << vizHandHog;
		}

		time++;
		prevCapFrame = capFrame.clone();
		waitKey(1);
	}
	return outFeatureFile;
}

int main (int argc, char * const argv[]) 
{
	if (argc!=4 && argc!=5) {
		cout << "[Usage] ./DenseTrajectory [input movie list file] [output feature directory] [output feature list file] [(optional) input database type]" << endl;
		return -1;
	}
	string movieListFile = argv[1];
	string outFeatureDir = argv[2];
	string outFeatureListFile = argv[3];
	string inputDatabaseType = "mine";
	if (argc==5) {
		inputDatabaseType = argv[4];
	}
	CV_Assert(inputDatabaseType=="mine" || inputDatabaseType=="gtea" || inputDatabaseType=="gteagaze-verb" || inputDatabaseType=="gteagaze-verb-object" || inputDatabaseType=="cmu-kitchen");

	if (inputDatabaseType=="mine") {
		vector<string> movieFiles;
		vector<int> movieClassIDs;

		ifstream fs;
		fs.open(movieListFile);
		string line;
		string tokenDelim = ",";
		while (fs>>line) {
			if (line.size()>0) {
				vector<string> tokens = StringUtils::splitString(line, tokenDelim);
				CV_Assert(tokens.size()==2);

				movieFiles.push_back(tokens[0]);
				movieClassIDs.push_back(std::stoi(tokens[1]));
			}
		}

		ofstream ofs(outFeatureListFile, ios_base::out);
		for (int i=0; i<movieFiles.size(); i++) {
			string featureFile = processMovie(movieFiles[i], MYDATASET_MOVIE_FRAME_WIDTH, MYDATASET_MOVIE_FRAME_HEIGHT, 
											MYDATASET_TIME_INTERVAL_DETECT_KEYPOINT, MYDATASET_DENSE_KEYPOINT_STEP, outFeatureDir, ",");
			if (featureFile.size()>0) {
				ofs << featureFile << "," << movieClassIDs[i] << endl;
			}
		}
	} else if (inputDatabaseType=="gtea") {
		vector<vector<string> > tokens;
		FileUtils::readTSV(movieListFile, tokens);

		vector<string> movieFiles;
		vector<string> movieSubjects;
		vector<string> movieFoods;
		vector<string> movieActions;
		vector<string> movieObjects;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==6);

			movieFiles.push_back(tokens[i][0]);
			movieSubjects.push_back(tokens[i][1]);
			movieFoods.push_back(tokens[i][2]);
			movieActions.push_back(tokens[i][3]);
			movieObjects.push_back(tokens[i][4]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][5]));
		}

		int countNone = 0;
		ofstream ofs(outFeatureListFile, ios_base::out);
		for (int i=0; i<movieFiles.size(); i++) {
			if (find(GTEA_TARGET_FOODS.begin(), GTEA_TARGET_FOODS.end(), movieFoods[i])!=GTEA_TARGET_FOODS.end()
				&& find(GTEA_TARGET_ACTIONS.begin(), GTEA_TARGET_ACTIONS.end(), movieActions[i])!=GTEA_TARGET_ACTIONS.end()) {
				cout << "extract feature from the movie : " << movieFiles[i] << endl;

				string featureFile;
				if ("none"==movieActions[i]) {
					if (countNone>=GTEA_ACTION_SKIP_NUM_EVERY_NONE) {
						featureFile = processMovie(movieFiles[i], GTEA_MOVIE_FRAME_WIDTH, GTEA_MOVIE_FRAME_HEIGHT, 
												GTEA_TIME_INTERVAL_DETECT_KEYPOINT, GTEA_DENSE_KEYPOINT_STEP, 
												outFeatureDir, "\t", movieSubjects[i], GTEA_HAND_DETECT_MODEL_LIST_FILES, 
												GTEA_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES, GTEA_MAX_FRAME_NONE_ACTION);
						countNone=0;
					} else {
						countNone++;
					}
				} else {
					featureFile = processMovie(movieFiles[i], GTEA_MOVIE_FRAME_WIDTH, GTEA_MOVIE_FRAME_HEIGHT, 
											GTEA_TIME_INTERVAL_DETECT_KEYPOINT, GTEA_DENSE_KEYPOINT_STEP, 
											outFeatureDir, "\t", movieSubjects[i], GTEA_HAND_DETECT_MODEL_LIST_FILES, 
											GTEA_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES);
				}

				if (featureFile.size()>0) {
					ofs << featureFile << "\t" << movieFiles[i] << "\t" << movieSubjects[i] << "\t" << movieFoods[i] 
						<< "\t" << movieActions[i] << "\t" << movieObjects[i] << "\t" << movieFrameIndexs[i] << endl;
				}
			} else {
				cout << "skip the movie : " << movieFiles[i] << endl;
			}
		}
	} else if (inputDatabaseType=="gteagaze-verb" || inputDatabaseType=="gteagaze-verb-object") {
		map<string, string> normVerbObjMap = GTEAGazePlusUtils::getGteaGazeNormalizeVerbObjectMap();

		vector<vector<string> > tokens;
		FileUtils::readTSV(movieListFile, tokens);

		vector<string> movieFiles;
		vector<string> movieSubjects;
		vector<string> movieFoods;
		vector<string> movieActionVerbs;
		vector<string> movieActionVerbObjects;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==6);

			movieFiles.push_back(tokens[i][0]);
			movieSubjects.push_back(tokens[i][1]);
			movieFoods.push_back(tokens[i][2]);
			movieActionVerbs.push_back(tokens[i][3]);
			movieActionVerbObjects.push_back(tokens[i][4]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][5]));
		}

		int countNone = 0;
		ofstream ofs(outFeatureListFile, ios_base::out);
		for (int i=0; i<movieFiles.size(); i++) {
			if (inputDatabaseType=="gteagaze-verb") {
				if (find(GTEAGAZE_TARGET_FOODS.begin(), GTEAGAZE_TARGET_FOODS.end(), movieFoods[i])!=GTEAGAZE_TARGET_FOODS.end()
					&& find(GTEAGAZE_TARGET_VERBS.begin(), GTEAGAZE_TARGET_VERBS.end(), movieActionVerbs[i])!=GTEAGAZE_TARGET_VERBS.end()) {
					cout << "extract feature from the movie : " << movieFiles[i] << endl;

					string featureFile;
					if ("none"==movieActionVerbs[i]) {
						if (countNone>=GTEAGAZE_ACTION_VERB_SKIP_NUM_EVERY_NONE) {
							featureFile = processMovie(movieFiles[i], GTEAGAZE_MOVIE_FRAME_WIDTH, GTEAGAZE_MOVIE_FRAME_HEIGHT, 
													GTEAGAZE_TIME_INTERVAL_DETECT_KEYPOINT, GTEAGAZE_DENSE_KEYPOINT_STEP, 
													outFeatureDir, "\t", movieSubjects[i], GTEAGAZE_HAND_DETECT_MODEL_LIST_FILES, 
													GTEAGAZE_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES, GTEAGAZE_MAX_FRAME_NONE_ACTION);
							countNone=0;
						} else {
							countNone++;
						}
					} else {
						featureFile = processMovie(movieFiles[i], GTEAGAZE_MOVIE_FRAME_WIDTH, GTEAGAZE_MOVIE_FRAME_HEIGHT, 
												GTEAGAZE_TIME_INTERVAL_DETECT_KEYPOINT, GTEAGAZE_DENSE_KEYPOINT_STEP, 
												outFeatureDir, "\t", movieSubjects[i], GTEAGAZE_HAND_DETECT_MODEL_LIST_FILES, 
												GTEAGAZE_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES);
					}

					if (featureFile.size()>0) {
						ofs << featureFile << "\t" << movieFiles[i] << "\t" << movieSubjects[i] << "\t" << movieFoods[i] 
							<< "\t" << movieActionVerbs[i] << "\t" << movieActionVerbObjects[i] << "\t" << movieFrameIndexs[i] << endl;
					}
				} else {
					cout << "skip the movie : " << movieFiles[i] << endl;
				}
			} else if (inputDatabaseType=="gteagaze-verb-object") {
				string featureFile;
				string normVerbObj;
				if ("none"==movieActionVerbs[i]) {
					// skip
				} else {
					normVerbObj = normVerbObjMap[movieActionVerbObjects[i]];
					if (normVerbObj.size()>0) {
						cout << "extract feature from the movie : " << movieFiles[i] << endl;
						featureFile = processMovie(movieFiles[i], GTEAGAZE_MOVIE_FRAME_WIDTH, GTEAGAZE_MOVIE_FRAME_HEIGHT, 
												GTEAGAZE_TIME_INTERVAL_DETECT_KEYPOINT, GTEAGAZE_DENSE_KEYPOINT_STEP, outFeatureDir, "\t");
					}
				}

				if (featureFile.size()>0) {
					ofs << featureFile << "\t" << movieFiles[i] << "\t" << movieSubjects[i] << "\t" << movieFoods[i] 
						<< "\t" << movieActionVerbs[i] << "\t" << normVerbObj << "\t" << movieFrameIndexs[i] << endl;
				} else {
					cout << "skip the movie : " << movieFiles[i] << endl;
				}
			}
		}
	} else if (inputDatabaseType=="cmu-kitchen") {
		vector<vector<string> > tokens;
		FileUtils::readTSV(movieListFile, tokens);

		vector<string> movieFiles;
		vector<string> movieSubjects;
		vector<string> movieFoods;
		vector<string> movieActions;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==5);

			movieFiles.push_back(tokens[i][0]);
			movieSubjects.push_back(tokens[i][1]);
			movieFoods.push_back(tokens[i][2]);
			movieActions.push_back(tokens[i][3]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][4]));
		}

		int countNone = 0;
		ofstream ofs(outFeatureListFile, ios_base::out);
		for (int i=0; i<movieFiles.size(); i++) {
			if (find(CMUKITCHEN_TARGET_FOODS.begin(), CMUKITCHEN_TARGET_FOODS.end(), movieFoods[i])!=CMUKITCHEN_TARGET_FOODS.end()
				&& find(CMUKITCHEN_TARGET_ACTIONS.begin(), CMUKITCHEN_TARGET_ACTIONS.end(), movieActions[i])!=CMUKITCHEN_TARGET_ACTIONS.end()) {
				cout << "extract feature from the movie : " << movieFiles[i] << endl;

				string featureFile;
				if ("none---"==movieActions[i]) {
					if (countNone>=CMUKITCHEN_ACTION_SKIP_NUM_EVERY_NONE) {
						featureFile = processMovie(movieFiles[i], CMUKITCHEN_MOVIE_FRAME_WIDTH, CMUKITCHEN_MOVIE_FRAME_HEIGHT, 
												CMUKITCHEN_TIME_INTERVAL_DETECT_KEYPOINT, CMUKITCHEN_DENSE_KEYPOINT_STEP, 
												outFeatureDir, "\t", movieSubjects[i], CMUKITCHEN_HAND_DETECT_MODEL_LIST_FILES, 
												CMUKITCHEN_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES, CMUKITCHEN_MAX_FRAME_NONE_ACTION);
						countNone=0;
					} else {
						countNone++;
					}
				} else {
					featureFile = processMovie(movieFiles[i], CMUKITCHEN_MOVIE_FRAME_WIDTH, CMUKITCHEN_MOVIE_FRAME_HEIGHT, 
											CMUKITCHEN_TIME_INTERVAL_DETECT_KEYPOINT, CMUKITCHEN_DENSE_KEYPOINT_STEP, 
											outFeatureDir, "\t", movieSubjects[i], CMUKITCHEN_HAND_DETECT_MODEL_LIST_FILES, 
											CMUKITCHEN_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES);
				}

				if (featureFile.size()>0) {
					ofs << featureFile << "\t" << movieFiles[i] << "\t" << movieSubjects[i] << "\t" << movieFoods[i] 
						<< "\t" << movieActions[i] << "\t" << movieFrameIndexs[i] << endl;
				}
			} else {
				cout << "skip the movie : " << movieFiles[i] << endl;
			}
		}
	}

	return 0;
}