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
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "FileUtils.h"
#include "StringUtils.h"
#include "MyDatasetConstants.h"
#include "GTEAConstants.h"
#include "GTEAGazePlusConstants.h"
#include "CMUKitchenConstants.h"

using namespace std;
using namespace cv;

static const bool SAVE_NONE_ACTION = true;

/*
	parse string in following format : <action><object>
	example : <scoop><coffee,spoon>
*/
void parseGTEAActionObject(const string& token, string& action, string& object) {
	CV_Assert(token[0]=='<' && token[token.length()-1]=='>');

	action = "";
	object = "";

	bool bInParen = false;
	stringstream ss;
	for (int i=0; i<token.length(); i++) {
		if (token[i]=='<') {
			bInParen = true;
		} else if (token[i]=='>') {
			bInParen = false;

			if (action.length()==0) {
				action = ss.str();
			} else {
				object = ss.str();
			}
			ss.str("");
		} else if (token[i]!='<' && token[i]!='>' && bInParen) {
			ss << token[i];
		}
	}
}

/*
	parse string in following format : (startFrame-endFrame)
	example : (237-276)
*/
void parseGTEAFrameNum(const string& token, int& startFrame, int& endFrame) {
	CV_Assert(token[0]=='(' && token[token.length()-1]==')');

	stringstream ss;
	for (int i=0; i<token.length(); i++) {
		if (token[i]=='(') {
		} else if (token[i]=='-') {
			startFrame = atoi(ss.str().c_str());
			ss.str("");
		} else if (token[i]==')') {
			endFrame = atoi(ss.str().c_str());
		} else {
			ss << token[i];
		}
	}
}

/*
	convert timestamp format string (hh:mm:ss.fff) to frame number
*/
int convertTimestamp2FrameNumber(const string& timestamp, int fps) {
	vector<string> tokens;

    string token;
    stringstream stream(timestamp);
    while (getline(stream, token, ':')) {
		tokens.push_back(token);
	}
	CV_Assert(tokens.size()==3);

	int hour = atoi(tokens[0].c_str());
	int min = atoi(tokens[1].c_str());
	double secDouble = atof(tokens[2].c_str());
	int sec = floor(secDouble);
	int dec = (secDouble - sec) / (1.0/(double)fps);
	int frame = hour * 60 * 60 * fps;
	frame += min * 60 * fps;
	frame += sec * fps;
	frame += dec;

	return frame;
}

int parseGTEADataset(const vector<string>& inputMovieFiles, const vector<string>& inputMovieFilesSubject, const vector<string>& inputMovieFilesFood, 
					const vector<string>& inputAnnotateFiles, const string outputMovieDir, const string outputMovieListFile)
{
	ofstream ofs(outputMovieListFile, ios_base::out);
	for (int i=0; i<inputMovieFiles.size(); i++) {
		// open movie
		VideoCapture capture(inputMovieFiles[i]);
		if (!capture.isOpened()) {
			cout << "Failed to open video : " << inputMovieFiles[i] << endl;
			return -1;
		}

		// open annotation
		vector<int> startFrameNums;
		vector<int> endFrameNums;
		vector<string> actions;
		vector<string> objects;
		{
			vector<vector<string> > tokens;
			FileUtils::readSSV(inputAnnotateFiles[i], tokens);
			for (int j=0; j<tokens.size(); j++) {
				CV_Assert(tokens[j].size()==2 || tokens[j].size()==3);

				if (tokens[j].size()==3) { // action annotation
					int startFrame = -1;
					int endFrame = -1;
					string action;
					string object;
					for (int k=0; k<tokens[j].size(); k++) {
						if (k==0) {
							parseGTEAActionObject(tokens[j][k], action, object);
						} else if (k==1) {
							parseGTEAFrameNum(tokens[j][k], startFrame, endFrame);
						} else if (k==2) {
							// skip this token
						}
					}
					actions.push_back(action);
					objects.push_back(object);
					startFrameNums.push_back(startFrame);
					endFrameNums.push_back(endFrame);
				} if (tokens[j].size()==2) { // object annotation
					cout << "Skip object annotation : " << tokens[j][0] << " " << tokens[j][1] << endl;
				}
			}
		}

		// process movie
		int count = 0;
		string curAction = "";
		string curObject = "";
		int curActionStartFrame = 0;
		int curActionEndFrame = 0;
		cv::VideoWriter videoWriter;
		while(true) {
			Mat frame;
			capture >> frame;
			if(frame.empty()) {
				break;
			}
			cv::resize(frame, frame, cv::Size(), GTEA_PARSE_SCALE_MOVIE, GTEA_PARSE_SCALE_MOVIE);

			vector<int>::iterator iterStartFrameNum = find(startFrameNums.begin(), startFrameNums.end(), count);
			if (iterStartFrameNum!=startFrameNums.end()) { // if current frame is start of action
				int pos = iterStartFrameNum - startFrameNums.begin();
				curAction = actions[pos];
				curObject = objects[pos];
				curActionStartFrame = startFrameNums[pos];

				stringstream ss;
				ss << outputMovieDir << "\\" << inputMovieFilesSubject[i] << "-" << inputMovieFilesFood[i] << "-" << curAction << "-" << curObject << "-" << count << ".avi";

				if (videoWriter.isOpened()) {
					videoWriter.release();
				}
				int codec = CV_FOURCC('M', 'J', 'P', 'G');
				Size capSize = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH)*GTEA_PARSE_SCALE_MOVIE, (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT)*GTEA_PARSE_SCALE_MOVIE);
				videoWriter.open(ss.str(), codec, (double)GTEA_FPS, capSize, true);

				ofs << ss.str() << "\t" << inputMovieFilesSubject[i] << "\t" << inputMovieFilesFood[i] << "\t" << curAction << "\t" << curObject << "\t" << count << endl;
			}
			vector<int>::iterator iterEndFrameNum = find(endFrameNums.begin(), endFrameNums.end(), count);
			if (curAction=="" || iterEndFrameNum!=endFrameNums.end()) { // if action is not annotated, or current frame is end of action
				curAction = "none";
				curObject = "none";

				if (videoWriter.isOpened()) {
					videoWriter.release();
				}

				if (SAVE_NONE_ACTION) {
					stringstream ss;
					ss << outputMovieDir << "\\" << inputMovieFilesSubject[i] << "-" << inputMovieFilesFood[i] << "-" << curAction << "-" << curObject << "-" << count << ".avi";

					int codec = CV_FOURCC('M', 'J', 'P', 'G');
					Size capSize = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH)*GTEA_PARSE_SCALE_MOVIE, (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT)*GTEA_PARSE_SCALE_MOVIE);
					videoWriter.open(ss.str(), codec, (double)GTEA_FPS, capSize, true);

					ofs << ss.str() << "\t" << inputMovieFilesSubject[i] << "\t" << inputMovieFilesFood[i] << "\t" << curAction << "\t" << curObject << "\t" << count << endl;
				}
			}
			if (videoWriter.isOpened()) {
				videoWriter << frame;
			}

			imshow("Input Video", frame);
			waitKey(1);
			count++;
		}
	}
}

int parseGTEAGazeDataset(const vector<string>& inputMovieFiles, const vector<string>& inputMovieFilesSubject, const vector<string>& inputMovieFilesFood, 
						const vector<string>& inputAnnotateFiles, const string outputMovieDir, const string outputMovieListFile)
{
	ofstream ofs(outputMovieListFile, ios_base::out);
	for (int i=0; i<inputMovieFiles.size(); i++) {
		// open movie
		VideoCapture capture(inputMovieFiles[i]);
		if (!capture.isOpened()) {
			cout << "Failed to open video : " << inputMovieFiles[i] << endl;
			return -1;
		}

		// open annotation
		vector<string> actionVerbs;
		vector<int> startFrameNums;
		vector<int> endFrameNums;
		vector<string> actions;
		{
			vector<vector<string> > tokens;
			FileUtils::readTSV(inputAnnotateFiles[i], tokens);
			for (int j=0; j<tokens.size(); j++) {
				CV_Assert(tokens[j].size()==4);

				int startFrame = -1;
				int endFrame = -1;
				string actionVerb;
				string action;
				for (int k=0; k<tokens[j].size(); k++) {
					if (k==0) {
						actionVerb = StringUtils::trim(tokens[j][k]);
						replace(actionVerb.begin(), actionVerb.end(), '/', ' ');
					} else if (k==1) {
						startFrame = convertTimestamp2FrameNumber(StringUtils::trim(tokens[j][k]), GTEAGAZE_FPS);
					} else if (k==2) {
						endFrame = convertTimestamp2FrameNumber(StringUtils::trim(tokens[j][k]), GTEAGAZE_FPS);
					} else if (k==3) {
						action = StringUtils::trim(tokens[j][k]);
						replace(action.begin(), action.end(), '/', ' ');
					}
				}
				actionVerbs.push_back(actionVerb);
				startFrameNums.push_back(startFrame);
				endFrameNums.push_back(endFrame);
				actions.push_back(action);
			}
		}

		// process movie
		int count = 0;
		string curActionVerb = "";
		string curAction = "";
		int curActionStartFrame = 0;
		int curActionEndFrame = 0;
		cv::VideoWriter videoWriter;
		while(true) {
			Mat frame;
			capture >> frame;
			if(frame.empty()) {
				break;
			}
			cv::resize(frame, frame, cv::Size(), GTEAGAZE_PARSE_SCALE_MOVIE, GTEAGAZE_PARSE_SCALE_MOVIE);

			vector<int>::iterator iterStartFrameNum = find(startFrameNums.begin(), startFrameNums.end(), count);
			if (iterStartFrameNum!=startFrameNums.end()) { // if current frame is start of action
				int pos = iterStartFrameNum - startFrameNums.begin();
				curActionVerb = actionVerbs[pos];
				curActionStartFrame = startFrameNums[pos];
				curAction = actions[pos];

				stringstream ss;
				ss << outputMovieDir << "\\" << inputMovieFilesSubject[i] << "-" << inputMovieFilesFood[i] << "-" << curActionVerb << "-" << curAction << "-" << count << ".avi";

				if (videoWriter.isOpened()) {
					videoWriter.release();
				}
				int codec = CV_FOURCC('M', 'J', 'P', 'G');
				Size capSize = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH)*GTEAGAZE_PARSE_SCALE_MOVIE, (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT)*GTEAGAZE_PARSE_SCALE_MOVIE);
				videoWriter.open(ss.str(), codec, (double)GTEAGAZE_FPS, capSize, true);

				ofs << ss.str() << "\t" << inputMovieFilesSubject[i] << "\t" << inputMovieFilesFood[i] << "\t" << curActionVerb << "\t" << curAction << "\t" << count << endl;
			}
			vector<int>::iterator iterEndFrameNum = find(endFrameNums.begin(), endFrameNums.end(), count);
			if (curActionVerb=="" || iterEndFrameNum!=endFrameNums.end()) { // if action is not annotated, or current frame is end of action
				curActionVerb = "none";
				curAction = "none";

				if (videoWriter.isOpened()) {
					videoWriter.release();
				}

				if (SAVE_NONE_ACTION) {
					stringstream ss;
					ss << outputMovieDir << "\\" << inputMovieFilesSubject[i] << "-" << inputMovieFilesFood[i] << "-" << curActionVerb << "-" << curAction << "-" << count << ".avi";

					int codec = CV_FOURCC('M', 'J', 'P', 'G');
					Size capSize = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH)*GTEAGAZE_PARSE_SCALE_MOVIE, (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT)*GTEAGAZE_PARSE_SCALE_MOVIE);
					videoWriter.open(ss.str(), codec, (double)GTEAGAZE_FPS, capSize, true);

					ofs << ss.str() << "\t" << inputMovieFilesSubject[i] << "\t" << inputMovieFilesFood[i] << "\t" << curActionVerb << "\t" << curAction << "\t" << count << endl;
				}
			}
			if (videoWriter.isOpened()) {
				videoWriter << frame;
			}

			imshow("Input Video", frame);
			waitKey(1);
			count++;
		}
	}
}

int parseCMUKitchenDataset(const vector<string>& inputMovieFiles, const vector<string>& inputMovieFilesSubject, const vector<string>& inputMovieFilesFood, 
						const vector<string>& inputAnnotateFiles, const string outputMovieDir, const string outputMovieListFile)
{
	ofstream ofs(outputMovieListFile, ios_base::out);
	for (int i=0; i<inputMovieFiles.size(); i++) {
		// open movie
		VideoCapture capture(inputMovieFiles[i]);
		if (!capture.isOpened()) {
			cout << "Failed to open video : " << inputMovieFiles[i] << endl;
			return -1;
		}

		// get start/end annotation frame
		int startAnnotationFrameNum = CMUKITCHEN_BROWNIE_MOVIE_START_FRAME[inputMovieFilesSubject[i]];
		int endAnnotationFrameNum = CMUKITCHEN_BROWNIE_MOVIE_END_FRAME[inputMovieFilesSubject[i]];
		CV_Assert(startAnnotationFrameNum>=0 && endAnnotationFrameNum>=0);

		// open annotation
		vector<int> startFrameNums;
		vector<int> endFrameNums;
		vector<string> actions;
		{
			vector<vector<string> > tokens;
			FileUtils::readSSV(inputAnnotateFiles[i], tokens);
			for (int j=0; j<tokens.size(); j++) {
				CV_Assert(tokens[j].size()==3);

				int startFrame = -1;
				int endFrame = -1;
				string action;
				for (int k=0; k<tokens[j].size(); k++) {
					if (k==0) {
						startFrame = startAnnotationFrameNum + atoi(StringUtils::trim(tokens[j][k]).c_str())-1;
					} else if (k==1) {
						endFrame = startAnnotationFrameNum + atoi(StringUtils::trim(tokens[j][k]).c_str())-1;
					} else if (k==2) {
						action = StringUtils::trim(tokens[j][k]);
					}
				}
				startFrameNums.push_back(startFrame);
				endFrameNums.push_back(endFrame);
				actions.push_back(action);
			}
		}

		// process movie
		int count = 0;
		string curAction = "";
		int curActionStartFrame = 0;
		int curActionEndFrame = 0;
		cv::VideoWriter videoWriter;
		while(true) {
			Mat frame;
			capture >> frame;
			if(frame.empty() || count>endAnnotationFrameNum) {
				break;
			}
			cv::resize(frame, frame, cv::Size(), CMUKITCHEN_PARSE_SCALE_MOVIE, CMUKITCHEN_PARSE_SCALE_MOVIE);
			cv::flip(frame, frame, 0); // make upside down

			if (count>=startAnnotationFrameNum) {
				vector<int>::iterator iterStartFrameNum = find(startFrameNums.begin(), startFrameNums.end(), count);
				if (iterStartFrameNum!=startFrameNums.end()) { // if current frame is start of action
					int pos = iterStartFrameNum - startFrameNums.begin();
					curActionStartFrame = startFrameNums[pos];
					curAction = actions[pos];

					stringstream ss;
					ss << outputMovieDir << "\\" << inputMovieFilesSubject[i] << "-" << inputMovieFilesFood[i] << "-" << curAction << "-" << count << ".avi";

					if (videoWriter.isOpened()) {
						videoWriter.release();
					}
					int codec = CV_FOURCC('M', 'J', 'P', 'G');
					Size capSize = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH)*CMUKITCHEN_PARSE_SCALE_MOVIE, (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT)*CMUKITCHEN_PARSE_SCALE_MOVIE);
					videoWriter.open(ss.str(), codec, (double)CMUKITCHEN_FPS, capSize, true);

					ofs << ss.str() << "\t" << inputMovieFilesSubject[i] << "\t" << inputMovieFilesFood[i] << "\t" << curAction << "\t" << count << endl;
				}
				if (videoWriter.isOpened()) {
					videoWriter << frame;
				}
			}
			imshow("Input Video", frame);
			waitKey(1);
			count++;
		}
	}
}

int main (int argc, char * const argv[]) 
{
	if (argc!=5 && argc!=6) {
		cout << "[Usage] ./ParseAnnotatedMovie [input movie list file] [input annotation list file] [output movie directory] [output movie list file] [(optional) input database type]" << endl;
		return -1;
	}
	string inputMovieListFile = argv[1];
	string inputAnnotateListFile = argv[2];
	string outputMovieDir = argv[3];
	string outputMovieListFile = argv[4];
	string inputDatabaseType = "gtea";
	if (argc==6) {
		inputDatabaseType = argv[5];
	}
	CV_Assert(inputDatabaseType=="gtea" || inputDatabaseType=="gtea-gaze" || inputDatabaseType=="cmu-kitchen");

	vector<string> inputMovieFiles;
	vector<string> inputMovieFilesSubject;
	vector<string> inputMovieFilesFood;
	{
		vector<vector<string> > tokens;
		FileUtils::readTSV(inputMovieListFile, tokens);

		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==3);
			inputMovieFiles.push_back(tokens[i][0]);
			inputMovieFilesSubject.push_back(tokens[i][1]);
			inputMovieFilesFood.push_back(tokens[i][2]);
		}
	}
	vector<string> inputAnnotateFiles;
	{
		vector<vector<string> > tokens;
		FileUtils::readTSV(inputAnnotateListFile, tokens);
		for (int i=0; i<tokens.size(); i++) {
			CV_Assert(tokens[i].size()==1);
			inputAnnotateFiles.push_back(tokens[i][0]);
		}
	}
	CV_Assert(inputMovieFiles.size()==inputAnnotateFiles.size());
	
	if (inputDatabaseType=="gtea") {
		parseGTEADataset(inputMovieFiles, inputMovieFilesSubject, inputMovieFilesFood, inputAnnotateFiles, outputMovieDir, outputMovieListFile);
	} else if (inputDatabaseType=="gtea-gaze") {
		parseGTEAGazeDataset(inputMovieFiles, inputMovieFilesSubject, inputMovieFilesFood, inputAnnotateFiles, outputMovieDir, outputMovieListFile);
	} else if (inputDatabaseType=="cmu-kitchen") {
		parseCMUKitchenDataset(inputMovieFiles, inputMovieFilesSubject, inputMovieFilesFood, inputAnnotateFiles, outputMovieDir, outputMovieListFile);
	}

    return 0;
}