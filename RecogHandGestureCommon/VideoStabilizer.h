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

#ifndef __RECOG_HAND_GESTURE_COMMON_VIDEO_STABILIZER__
#define __RECOG_HAND_GESTURE_COMMON_VIDEO_STABILIZER__

#include <opencv2/opencv.hpp>
#include "FeatureExtractor.h"

using namespace std;
using namespace cv;

class VideoStabilizer
{
public:
	VideoStabilizer();
	~VideoStabilizer();

	cv::Mat calcStabilizedFrame(cv::Mat &curFrame, cv::Mat &prevFrame, const Mat& mask=Mat());

private:
	FeatureExtractor *mFeatureExtractor;

	void symmetryTest(const vector<vector<DMatch>>& matches1, const vector<vector<DMatch>>& matches2, vector<DMatch>& symMatches);
	int ratioTest(const float ratio, vector<vector<DMatch>>& matches);
	int computeHomography(Mat& image1, Mat& image2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, 
						vector<uchar>& inlierKeypoints1, vector<uchar>& inlierKeypoints2, Mat& H, const Mat& mask=Mat());
};

#endif __RECOG_HAND_GESTURE_COMMON_VIDEO_STABILIZER__