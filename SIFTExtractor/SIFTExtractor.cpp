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
#include <opencv2/nonfree/nonfree.hpp>

#include "RecogHandGestureConstants.h"
#include "MyDatasetConstants.h"
#include "GTEAConstants.h"
#include "GTEAGazePlusConstants.h"
#include "CMUKitchenConstants.h"
#include "GTEAGazePlusUtils.h"
#include "DenseTrajectory.h"
#include "DenseTrajectoryTrackPoint.h"
#include "FileUtils.h"
#include "StringUtils.h"
#include "TimerUtils.h"

using namespace std;
using namespace cv;

string processMovie(string movieFile, int movieFrameWidth, int movieFrameHeight, string outFeatureDir, string delimFeatureFile, int maxFrame=-1)
{
	VideoCapture capture(movieFile);
	if (!capture.isOpened()) {
		cout << "Failed to open video : " << movieFile << endl;
		return "";
	}

	string movieFileName = FileUtils::getFileName(movieFile);
	string outFeatureFile = outFeatureDir + "\\" + movieFileName + "-sift.txt";
	ofstream ofs(outFeatureFile, ios_base::out);

	Size capSize = Size(movieFrameWidth, movieFrameHeight);

	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");

	int time = 0;
	int siftDim = 0;
	while (true) {
		Mat frame;
		capture >> frame;
		if (frame.empty() || (maxFrame>0 && time>maxFrame)) {
			break;
		}
		cv::resize(frame, frame, capSize);
		imshow("Input Video", frame);

		TimerUtils::getInstance().start();

		Mat grayImage;
		cvtColor(frame, grayImage, CV_BGR2GRAY);

		vector<cv::KeyPoint> keypoints;
		detector->detect(grayImage, keypoints);

		Mat keypointImage = frame.clone();
		std::vector<cv::KeyPoint>::iterator itk;
		for (itk = keypoints.begin(); itk != keypoints.end(); ++itk) {
			cv::circle(keypointImage, itk->pt, itk->size, cv::Scalar(0,255,255), 1, CV_AA);
			cv::circle(keypointImage, itk->pt, 1, cv::Scalar(0,255,0), -1);
		}
		cv::imshow("SIFT keypoints", keypointImage);

		Mat descriptors;
		extractor->compute(grayImage, keypoints, descriptors);
		CV_Assert(checkRange(descriptors));

		cout << "time to get descriptors : " << TimerUtils::getInstance().stop() << " secs." << endl;

		cout << "movie time frame : " << time << ", descriptor num=" << descriptors.rows << ", descriptor dim=" << descriptors.cols << endl;

		if (descriptors.empty() && siftDim>0) {
			descriptors = Mat::zeros(1, siftDim, CV_32F);
		}

		if (time>=dtMaxTimeLength && !descriptors.empty()) { // start save SIFT feature from the same time with dense trajectory for performance comparison
			CV_Assert(siftDim==0 || siftDim==descriptors.cols);
			
			stringstream ss;
			ss << time;
			string siftDescMatFile = outFeatureDir + "\\" + movieFileName + "-sift-" + ss.str() + ".yml";
			//FileUtils::saveMat(siftDescMatFile.c_str(), descriptors);
			FileUtils::saveMatBin(siftDescMatFile.c_str(), descriptors);
			ofs << siftDescMatFile << delimFeatureFile;

			ofs << time << endl;

			siftDim = descriptors.cols;
		}

		time++;
		waitKey(1);
	}
	return outFeatureFile;
}

int main (int argc, char * const argv[]) 
{
	if (argc!=4 && argc!=5) {
		cout << "[Usage] ./SIFTExtractor [input movie list file] [output feature directory] [output feature list file] [(optional) input database type]" << endl;
		return -1;
	}
	cv::initModule_nonfree();

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
			string featureFile = processMovie(movieFiles[i], MYDATASET_MOVIE_FRAME_WIDTH, MYDATASET_MOVIE_FRAME_HEIGHT, outFeatureDir, ",");
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
		vector<string> movieAction;
		vector<string> movieObject;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==6);

			movieFiles.push_back(tokens[i][0]);
			movieSubjects.push_back(tokens[i][1]);
			movieFoods.push_back(tokens[i][2]);
			movieAction.push_back(tokens[i][3]);
			movieObject.push_back(tokens[i][4]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][5]));
		}

		int countNone = 0;
		ofstream ofs(outFeatureListFile, ios_base::out);
		for (int i=0; i<movieFiles.size(); i++) {
			if (find(GTEA_TARGET_FOODS.begin(), GTEA_TARGET_FOODS.end(), movieFoods[i])!=GTEA_TARGET_FOODS.end()
				&& find(GTEA_TARGET_ACTIONS.begin(), GTEA_TARGET_ACTIONS.end(), movieAction[i])!=GTEA_TARGET_ACTIONS.end()) {
				cout << "extract feature from the movie : " << movieFiles[i] << endl;

				string featureFile;
				if ("none"==movieAction[i]) {
					if (countNone>=GTEA_ACTION_SKIP_NUM_EVERY_NONE) {
						featureFile = processMovie(movieFiles[i], GTEA_MOVIE_FRAME_WIDTH, GTEA_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t", GTEA_MAX_FRAME_NONE_ACTION);
						countNone=0;
					} else {
						countNone++;
					}
				} else {
					featureFile = processMovie(movieFiles[i], GTEA_MOVIE_FRAME_WIDTH, GTEA_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t");
				}

				if (featureFile.size()>0) {
					ofs << featureFile << "\t" << movieFiles[i] << "\t" << movieSubjects[i] << "\t" << movieFoods[i] 
						<< "\t" << movieAction[i] << "\t" << movieObject[i] << "\t" << movieFrameIndexs[i] << endl;
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
							featureFile = processMovie(movieFiles[i], GTEAGAZE_MOVIE_FRAME_WIDTH, GTEAGAZE_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t", GTEAGAZE_MAX_FRAME_NONE_ACTION);
							countNone=0;
						} else {
							countNone++;
						}
					} else {
						featureFile = processMovie(movieFiles[i], GTEAGAZE_MOVIE_FRAME_WIDTH, GTEAGAZE_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t");
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
					normVerbObj = movieActionVerbObjects[i];
					if (countNone>=GTEAGAZE_ACTION_VERB_OBJECT_SKIP_NUM_EVERY_NONE) {
						featureFile = processMovie(movieFiles[i], GTEAGAZE_MOVIE_FRAME_WIDTH, GTEAGAZE_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t", GTEAGAZE_MAX_FRAME_NONE_ACTION);
						countNone=0;
					} else {
						countNone++;
					}
				} else {
					normVerbObj = normVerbObjMap[movieActionVerbObjects[i]];
					if (normVerbObj.size()>0) {
						cout << "extract feature from the movie : " << movieFiles[i] << endl;
						featureFile = processMovie(movieFiles[i], GTEAGAZE_MOVIE_FRAME_WIDTH, GTEAGAZE_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t");
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
		vector<string> movieAction;
		vector<int> movieFrameIndexs;
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==5);

			movieFiles.push_back(tokens[i][0]);
			movieSubjects.push_back(tokens[i][1]);
			movieFoods.push_back(tokens[i][2]);
			movieAction.push_back(tokens[i][3]);
			movieFrameIndexs.push_back(std::stoi(tokens[i][4]));
		}

		int countNone = 0;
		ofstream ofs(outFeatureListFile, ios_base::out);
		for (int i=0; i<movieFiles.size(); i++) {
			if (find(CMUKITCHEN_TARGET_FOODS.begin(), CMUKITCHEN_TARGET_FOODS.end(), movieFoods[i])!=CMUKITCHEN_TARGET_FOODS.end()
				&& find(CMUKITCHEN_TARGET_ACTIONS.begin(), CMUKITCHEN_TARGET_ACTIONS.end(), movieAction[i])!=CMUKITCHEN_TARGET_ACTIONS.end()) {
				cout << "extract feature from the movie : " << movieFiles[i] << endl;

				string featureFile;
				if ("none---"==movieAction[i]) {
					if (countNone>=CMUKITCHEN_ACTION_SKIP_NUM_EVERY_NONE) {
						featureFile = processMovie(movieFiles[i], CMUKITCHEN_MOVIE_FRAME_WIDTH, CMUKITCHEN_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t", 
												CMUKITCHEN_MAX_FRAME_NONE_ACTION);
						countNone=0;
					} else {
						countNone++;
					}
				} else {
					featureFile = processMovie(movieFiles[i], CMUKITCHEN_MOVIE_FRAME_WIDTH, CMUKITCHEN_MOVIE_FRAME_HEIGHT, outFeatureDir, "\t");
				}

				if (featureFile.size()>0) {
					ofs << featureFile << "\t" << movieFiles[i] << "\t" << movieSubjects[i] << "\t" << movieFoods[i] 
						<< "\t" << movieAction[i] << "\t" << movieFrameIndexs[i] << endl;
				}
			} else {
				cout << "skip the movie : " << movieFiles[i] << endl;
			}
		}
	}

	return 0;
}