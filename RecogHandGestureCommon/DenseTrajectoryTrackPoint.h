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

#ifndef __DENSE_TRAJECTORY_TRACK_POINT__
#define __DENSE_TRAJECTORY_TRACK_POINT__

#include <opencv2/opencv.hpp>
#include "HogHofMbh.h"

using namespace std;
using namespace cv;

class DenseTrajectoryTrackPoint
{
private:
	int trajTimeLength; // max time length for each track points
	int cubeSize2D; // cube size in 2D
	int cubeSizeTime; // cube size in time length
	int shiftBlockSize; // size to shift HOG/HOF/MBH blocks in 2D
	int timeBlockNum; // block num in time
	vector<Point2f> points;
	vector<vector<float>> hogDesc; // HOG values for each 3D cell
	vector<vector<float>> hofDesc; // HOF values for each 3D cell
	vector<vector<float>> mbhxDesc; // MBHx values for each 3D cell
	vector<vector<float>> mbhyDesc; // MBHy values for each 3D cell

	void updateDescHistory(Point2f point, HogHofMbh& hogHofMbh, vector<Mat>& intImage, vector<vector<float>>& descHistory);

public:
	DenseTrajectoryTrackPoint();
	DenseTrajectoryTrackPoint(Size imageSize, int trajTimeLength, int cubeSize2D, int cubeSizeTime);

	bool markedDelete;

	void addPointToHistory(Point2f point);
	void updateHogHistory(Point2f point, HogHofMbh& hog, vector<Mat>& hogIImage);
	void updateHofHistory(Point2f point, HogHofMbh& hof, vector<Mat>& hofIImage);
	void updateMbhHistory(Point2f point, HogHofMbh& mbh, vector<Mat>& mbhxIImage, vector<Mat>& mbhyIImage);

	const Point2f& getPointHistory(int time);
	int getTimeLength();
	Mat getTrajDesc();
	Mat getHogDesc();
	Mat getHofDesc();
	Mat getMbhDesc();
};

#endif __DENSE_TRAJECTORY_TRACK_POINT__