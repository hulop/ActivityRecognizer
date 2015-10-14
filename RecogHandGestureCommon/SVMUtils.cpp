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

#include "SVMUtils.h"

void SVMUtils::prepareScaleData(const cv::Mat input, vector<float> &minValues, vector<float> &maxValues)
{
	minValues = vector<float>(input.cols, std::numeric_limits<float>::max());
	maxValues = vector<float>(input.cols, std::numeric_limits<float>::min());

	for (int i=0; i<input.rows; i++) {
		for (int j=0; j<input.cols; j++) {
			if (input.at<float>(i, j)<minValues[j]) {
				minValues[j] = input.at<float>(i, j);
			}
			if (input.at<float>(i, j)>maxValues[j]) {
				maxValues[j] = input.at<float>(i, j);
			}
		}
	}
}

void SVMUtils::scaleData(float minScale, float maxScale, const vector<float> &minValues, const vector<float> &maxValues, cv::Mat &target)
{
	for (int i=0; i<target.rows; i++) {
		for (int j=0; j<target.cols; j++) {
			target.at<float>(i, j) = minScale + (maxScale-minScale)*(target.at<float>(i, j)-minValues[j])/(maxValues[j]-minValues[j]);
		}
	}
}

void SVMUtils::saveScaleData(float minScale, float maxScale, const vector<float> &minValues, const vector<float> &maxValues, string filename)
{
	CV_Assert(minValues.size()==maxValues.size());

	ofstream ofs(filename, ios_base::out);
	ofs << minScale << "," << maxScale << endl;
	for (int i=0; i<minValues.size(); i++) {
		ofs << i << "," << minValues[i] << "," << maxValues[i] << endl;
	}
}

void SVMUtils::loadScaleData(const string filename, vector<float> &minValues, vector<float> &maxValues)
{
	minValues.clear();
	maxValues.clear();

	ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

	// first line is min, max value for scaled data
	string line;
	{
		getline(file, line);
		vector<string> tokens;

		string token;
		istringstream stream(line);
		while (getline(stream, token, ',')) {
			stringstream ss;
			ss << token;

			tokens.push_back(token);
		}
		CV_Assert(tokens.size() == 2);

		float min = atof(tokens[0].c_str());
		float max = atof(tokens[1].c_str());
		cout << "min, max value for scaled data : " << min << ", " << max << endl;
	}
	// read scale value for each dimension
	while(getline(file, line)){
		vector<string> tokens;

		string token;
		istringstream stream(line);
		while (getline(stream, token, ',' )) {
			stringstream ss;
			ss << token;

			tokens.push_back(token);
		}
		CV_Assert(tokens.size() == 3);

		int index = atoi(tokens[0].c_str());
		float min = atof(tokens[1].c_str());
		float max = atof(tokens[2].c_str());
		minValues.push_back(min);
		maxValues.push_back(max);
	}
}