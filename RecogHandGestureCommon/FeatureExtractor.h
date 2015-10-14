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

#ifndef __RECOG_HAND_GESTURE_COMMON_FEATURE_EXTRACTOR__
#define __RECOG_HAND_GESTURE_COMMON_FEATURE_EXTRACTOR__

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;

class FeatureExtractor
{
public:
	enum FeatureType {SIFT=0, SURF = 1, ORB = 2};

	FeatureExtractor(const FeatureType featureType);
	~FeatureExtractor();
	void detectKeypoints(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, const cv::Mat& mask=cv::Mat());
	void computeDescriptors(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
	FeatureType mFeatureType;

	cv::FeatureDetector *detector;
	cv::DescriptorExtractor *extractor;
};

#endif