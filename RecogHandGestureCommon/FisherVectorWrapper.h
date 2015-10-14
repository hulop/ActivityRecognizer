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

#ifndef __RECOG_HAND_GESTURE_COMMON_FISHER_VECTOR_WRAPPER__
#define __RECOG_HAND_GESTURE_COMMON_FISHER_VECTOR_WRAPPER__

#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>

extern "C" {
#include <vl/kmeans.h>
#include <vl/gmm.h>
#include <vl/fisher.h>
}

using namespace std;
using namespace cv;

class FisherVectorWrapper
{
public:
	enum NormFisherVectorFeatureType { UNKNOWN=-1, NONE=0, L2_NORM=1, LP_NORM=2 };

	FisherVectorWrapper();
	FisherVectorWrapper(const cv::Mat &gmmTrainFeatures, int _K, NormFisherVectorFeatureType _NORM_FISHER_VECTOR_FEATURE_TYPE);
	~FisherVectorWrapper();

    void write(FileStorage& fs) const;
    void read(const FileNode& node);

	cv::Mat calcFisherVector(const cv::Mat &descriptors);

private:
	int K;
	NormFisherVectorFeatureType NORM_FISHER_VECTOR_FEATURE_TYPE;

	VlGMM* gmm;

	cv::Mat weights;
	cv::Mat means;
	vector<cv::Mat> covs;
	cv::Mat normWeights;
	cv::Mat logNormWeights;

	VlGMM* calcVLFeatGMM(const cv::Mat& features, const int K, const int maxIter, const double eps, 
						cv::Mat& weights, cv::Mat& means, vector<cv::Mat>& covs);
};

#endif __RECOG_HAND_GESTURE_COMMON_FISHER_VECTOR_WRAPPER__