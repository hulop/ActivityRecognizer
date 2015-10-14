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

#ifndef __RECOG_HAND_GESTURE_COMMON_DENSE_TRAJECTORY__
#define __RECOG_HAND_GESTURE_COMMON_DENSE_TRAJECTORY__

#include <opencv2/opencv.hpp>
#include "DenseTrajectoryTrackPoint.h"
#include "HogHofMbh.h"

using namespace std;
using namespace cv;

class DenseTrajectory
{
private:
	int mDebug;

	bool mUseTraj;
	bool mUseHOG;
	bool mUseHOF;
	bool mUseMBH;

	Size mCapSize;
	int mTimeIntevalDetectKeypoint;
	int mTimeCountDetectKeypoint;

	int mDenseKeypointStep;
	int mTrackMedianFilterSize;
	int mPyramidLevels;
	int mMaxTimeLength;
	float mPyramidScale;
	int mDescGrid2D;
	int mDescGridTime;
	int mDescNeighborSize;

	vector<Mat> mPrevGrayPyramid;
	vector<vector<DenseTrajectoryTrackPoint> > mTrackPointsPyramid;

	HogHofMbh* mHog;
	HogHofMbh* mHof;
	HogHofMbh* mMbh;

	void detectKeypointsPyramid(const vector<Mat>& grayPyramid, const vector<Mat>& maskPyramid, vector<vector<cv::KeyPoint> >& keypointsPyramid);
	void trackKeypointsMedianFilterFlow(const cv::Mat& flow, int pyramidLevel, vector<cv::Point2f>& curPoints);

	Mat visualizeTrackPoints(Mat input);
	void visualizeDenseOpticalFlow(Mat flow);
	void visualizeHogHofMbh(vector<Mat>& intImage, int binNum, string windowName);

public:
	DenseTrajectory(bool useTraj, bool useHOG, bool useHOF, bool useMBH, Size capSize, int timeIntervalDetectKeypoint, int denseKeypointStep, 
					int trackMedianFilterSize, int pyramidLevels, int maxTimeLength, float pyramidScale, int descGrid2D, int descGridTime, int descNeighborSize);
	~DenseTrajectory();
	Mat update(Mat frame, Mat mask);
	Mat getDesc();
};

#endif __RECOG_HAND_GESTURE_COMMON_DENSE_TRAJECTORY__