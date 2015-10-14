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

#ifndef __RECOG_HAND_GESTURE_COMMON_HAND_DETECTOR_WRAPPER__
#define __RECOG_HAND_GESTURE_COMMON_HAND_DETECTOR_WRAPPER__

////////////// TODO Hand Detector 1
#include <HandDetector.hpp>
////////////// TODO Hand Detector 1

#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

class HandDetectorWrapper
{
public:

	void trainModels(string imgListFile, string mskListFile, string model_prefix, string modelListFile, string feat_prefix, string featListFile,
		string feature_set, int max_models, int img_width);

	void testInitialize(string modelListFile, string featListFile, string feature_set, int knn, int width);

	void test(Mat &img, int num_models, int step_size);

	Mat getResponseImage();
	Mat postprocess(Mat &img, vector<Point2f> &pt, float contourThres = 0.04);
	Mat postprocess(Mat &img, float contourThres = 0.04);

	int getDenseHandHOGDim(Size block_size, Size strid_size, Size cell_size, int ori_bins);
	Mat computeDenseHandHOG(Mat &image, Mat &prob, Size block_size, Size strid_size, Size cell_size, int ori_bins,
		Mat &mask, Mat& handHogDescs);

	void computeColorHist_HSV(Mat &src, Mat &hist);
	void colormap(Mat &src, Mat &dst, int do_norm);
	void rasterizeResVec(Mat &img, Mat&res, vector<KeyPoint> &keypts, cv::Size s, int bs);
	Mat normalizeProb(Mat &prob);

private:
	////////////// TODO Hand Detector 2
	HandDetector handDetector;
	////////////// TODO Hand Detector 2

	Mat visual_dense_hog(vector<Mat>& intImage, Mat mask, int binNum, bool fullRadianHogHofMbh, string windowName);
	Mat visual_dense_hog2(vector<Mat>& intImage, Mat mask, int binNum, bool fullRadianHogHofMbh, string windowName);
};

#endif