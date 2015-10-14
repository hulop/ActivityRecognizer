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

#include "VideoStabilizer.h"

static const int VIDEO_STABILIZER_INLIER_THRES = 8;
static const float VIDEO_STABILIZER_RATIO_TEST_VALUE = 0.7;

VideoStabilizer::VideoStabilizer()
{
	mFeatureExtractor = new FeatureExtractor(FeatureExtractor::SIFT);
}

VideoStabilizer::~VideoStabilizer()
{
	delete mFeatureExtractor;
}

void VideoStabilizer::symmetryTest(const vector<vector<DMatch>>& matches1, const vector<vector<DMatch>>& matches2, vector<DMatch>& symMatches) {
	for (vector<vector<DMatch>>::const_iterator matchIterator1=matches1.begin(); matchIterator1!=matches1.end(); ++matchIterator1) {
		if (matchIterator1->size() < 2)
			continue;

		for (vector<vector<DMatch>>::const_iterator matchIterator2=matches2.begin(); matchIterator2!=matches2.end(); ++matchIterator2) {
			if (matchIterator2->size() < 2)
				continue;

			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx  && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx, (*matchIterator1)[0].distance));
				break;
			}
		}
	}
}

int VideoStabilizer::ratioTest(const float ratio, vector<vector<DMatch>>& matches) {
	int removed=0;

	for (vector<std::vector<DMatch>>::iterator matchIterator=matches.begin(); matchIterator!=matches.end(); ++matchIterator) {
		if (matchIterator->size() > 1) {
			if ((*matchIterator)[0].distance/(*matchIterator)[1].distance > ratio) {
				matchIterator->clear();
				removed++;
			}
		} else {
			matchIterator->clear();
			removed++;
		}
	}

	return removed;
}

int VideoStabilizer::computeHomography(Mat& image1, Mat& image2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, 
									vector<uchar>& inlierKeypoints1, vector<uchar>& inlierKeypoints2, Mat& H, const Mat& mask)
{
	mFeatureExtractor->detectKeypoints(image1, keypoints1, mask);
	mFeatureExtractor->detectKeypoints(image2, keypoints2, mask);
	if (keypoints1.size()<VIDEO_STABILIZER_INLIER_THRES || keypoints2.size()<VIDEO_STABILIZER_INLIER_THRES) {
		return 0;
	}

	Mat descriptors1;
	mFeatureExtractor->computeDescriptors(image1, keypoints1, descriptors1);
	Mat descriptors2;
	mFeatureExtractor->computeDescriptors(image2, keypoints2, descriptors2);

	vector<vector<DMatch>> matches1, matches2;
	BFMatcher matcher;
	if(descriptors1.type()==CV_32FC1) {
		matcher = BFMatcher(NORM_L2);
	} else {
		matcher = BFMatcher(NORM_HAMMING);
	}
	matcher.knnMatch(descriptors1, descriptors2, matches1, 2);
	matcher.knnMatch(descriptors2, descriptors1, matches2, 2);

	ratioTest(VIDEO_STABILIZER_RATIO_TEST_VALUE, matches1);
	ratioTest(VIDEO_STABILIZER_RATIO_TEST_VALUE, matches2);

	vector<DMatch> symMatches;
	symmetryTest(matches1, matches2, symMatches);
	if (symMatches.size()<VIDEO_STABILIZER_INLIER_THRES) {
		return 0;
	}

	vector<cv::Point2f> points1, points2;
	for (int i=0; i<symMatches.size(); i++) {
		CV_Assert(symMatches[i].queryIdx < keypoints1.size());
		points1.push_back(cvPoint(keypoints1[symMatches[i].queryIdx].pt.x, keypoints1[symMatches[i].queryIdx].pt.y));
		CV_Assert(symMatches[i].trainIdx < keypoints2.size());
		points2.push_back(cvPoint(keypoints2[symMatches[i].trainIdx].pt.x, keypoints2[symMatches[i].trainIdx].pt.y));
	}

	inlierKeypoints1 = vector<uchar>(keypoints1.size(), 0);
	inlierKeypoints2 = vector<uchar>(keypoints2.size(), 0);

	int inliers = 0;
	try {
		std::vector<uchar> inlierMask;
		H = findHomography(points1, points2, cv::RANSAC, 3.0, inlierMask);

		for (int i=0; i<inlierMask.size(); i++) {
			if (inlierMask.at(i)) {
				inlierKeypoints1[symMatches[i].queryIdx] = 1;
				inlierKeypoints2[symMatches[i].trainIdx] = 1;
				inliers++;
			}
		}
	} catch( cv::Exception& e ) {
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
	}
	return inliers;
}

cv::Mat VideoStabilizer::calcStabilizedFrame(cv::Mat &curFrame, cv::Mat &prevFrame, const cv::Mat& mask)
{
	static const int DEBUG = 1;

	cout << "start matching image " << endl;
	vector<KeyPoint> curKeypoints, prevKeypoints;
	vector<uchar> inlierCurKeypoints, inlierPrevKeypoints;
	Mat H;
	int inliers = computeHomography(prevFrame, curFrame, prevKeypoints, curKeypoints, inlierPrevKeypoints, inlierCurKeypoints, H, mask);
	if (DEBUG) {
		cv::Mat showFrame = curFrame.clone();
		for (int k=0; k<curKeypoints.size(); k++) {
			Point roiKeypoint = Point(curKeypoints[k].pt.x, curKeypoints[k].pt.y);
			if (inlierCurKeypoints[k]) {
				circle(showFrame, roiKeypoint, curKeypoints[k].size, cv::Scalar(0,255,0), 1);
			} else {
				circle(showFrame, roiKeypoint, curKeypoints[k].size, cv::Scalar(0,0,255), 1);
			}
		}
		cv::imshow("input frame for video stabilizer", showFrame);
	}
	cout << "end matching image." << endl;

	Mat resultFrame;
	warpPerspective(curFrame, resultFrame, H.inv(DECOMP_SVD), curFrame.size());
	return resultFrame;
}