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

#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor(const FeatureType featureType)
{
	cv::initModule_nonfree();

	this->mFeatureType = featureType;

	switch (mFeatureType) {
		case SIFT:{
			detector = new cv::SIFT();
			extractor = new cv::SIFT();
			break;
		}
		case SURF:{
			detector = new cv::SURF();
			extractor = new cv::SURF();
			break;
		}
		case ORB:{
			detector = new cv::ORB();
			extractor = new cv::ORB();
			break;
		}
		default:{
			CV_Error(CV_StsBadArg, "invalid local feature type");
			break;
		}
	}
}

FeatureExtractor::~FeatureExtractor()
{
	delete detector;
	delete extractor;
}

void FeatureExtractor::detectKeypoints(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, const cv::Mat& mask)
{
	CV_Assert(image.type()==CV_8UC3);

	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, CV_BGR2GRAY);

    detector->detect(grayImage, keypoints, mask);
}

void FeatureExtractor::computeDescriptors(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
	CV_Assert(image.type()==CV_8UC3);

	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, CV_BGR2GRAY);

    extractor->compute(grayImage, keypoints, descriptors);
}