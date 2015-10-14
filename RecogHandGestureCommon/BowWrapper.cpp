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

#include "BowWrapper.h"

static const int KMEANS_ITERATION = 100;

BowWrapper::BowWrapper()
{
}

BowWrapper::BowWrapper(const cv::Mat &kmeansTrainFeatures, int _K, NormBowFeatureType _NORM_BOW_FEATURE_TYPE)
{
	K = _K;
	NORM_BOW_FEATURE_TYPE = _NORM_BOW_FEATURE_TYPE;

	cout << "Start train kmeans...." << endl;
	centers = trainKMeans(kmeansTrainFeatures);

	flannIndexParams = new cv::flann::KDTreeIndexParams();
	flannIndex = new cv::flann::Index(centers, *flannIndexParams);
	cout << "End train kmeans. Number of cluster is " << K << endl;
}

BowWrapper::~BowWrapper()
{
	delete flannIndexParams;
	delete flannIndex;
}

void BowWrapper::write(FileStorage& fs) const
{
	CV_Assert(NORM_BOW_FEATURE_TYPE>=0);	
    fs << "NORM_BOW_FEATURE_TYPE" << (NORM_BOW_FEATURE_TYPE == NormBowFeatureType::NONE ? string("NONE") :
									NORM_BOW_FEATURE_TYPE == NormBowFeatureType::L2_NORM ? string("L2") :
									NORM_BOW_FEATURE_TYPE == NormBowFeatureType::L1_NORM_SQUARE_ROOT ? string("L1_SQUARE_ROOT") :
									format("unknown_%d", NORM_BOW_FEATURE_TYPE));

	fs << "K" << K;

	fs << "centers" << centers;
}

void BowWrapper::read(const FileNode& node)
{
	string normBowTypeStr;
	node["NORM_BOW_FEATURE_TYPE"] >> normBowTypeStr;
	NORM_BOW_FEATURE_TYPE = normBowTypeStr == "NONE" ? BowWrapper::NONE :
							normBowTypeStr == "L2" ? BowWrapper::L2_NORM :
							normBowTypeStr == "L1_SQUARE_ROOT" ? BowWrapper::L1_NORM_SQUARE_ROOT :
							BowWrapper::UNKNOWN;
	CV_Assert(NORM_BOW_FEATURE_TYPE>=0);

	node["K"] >> K;

	node["centers"] >> centers;

	flannIndexParams = new cv::flann::KDTreeIndexParams();
	flannIndex = new cv::flann::Index(centers, *flannIndexParams);
}

cv::Mat BowWrapper::trainKMeans(const cv::Mat &features)
{
	cv::Mat centers;
	cv::Mat labels;

	int clusterCount = min(K, features.rows);
    cv::kmeans(features, clusterCount, labels, 
		cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, KMEANS_ITERATION, FLT_EPSILON),
		3, cv::KMEANS_PP_CENTERS, centers);

	return centers;
}

cv::Mat BowWrapper::calcBow(const cv::Mat &descriptors)
{
	CV_Assert(K>0);
	
	cv::Mat indices(descriptors.rows, 1, CV_32SC1);
	cv::Mat dists(descriptors.rows, 1, CV_32FC1);
	flannIndex->knnSearch (descriptors, indices, dists, 1, cv::flann::SearchParams());

	cv::Mat bof = cv::Mat::zeros(K, 1, CV_64F);
	for (int i=0; i<indices.rows; i++) {
		int idx = indices.at<int>(i, 0);
		bof.at<double>(idx) = bof.at<double>(idx) + 1.0;
	}
	
	for (int i=0; i<K; i++) {
		bof.at<double>(i) = bof.at<double>(i)/(double)(descriptors.rows);
	}

	switch (NORM_BOW_FEATURE_TYPE) {
		case L2_NORM:{
			double norm = cv::norm(bof);
			if (norm>0.0) {
				for (int i=0; i<K; i++) {
					bof.at<double>(i) = bof.at<double>(i)/norm;
				}
			}
			break;
		}
		case L1_NORM_SQUARE_ROOT:{
			double norm = 0.0;
			for (int i=0; i<K; i++) {
				norm += bof.at<double>(i);
			}
			if (norm>0.0) {
				for (int i=0; i<K; i++) {
					bof.at<double>(i) = sqrt(bof.at<double>(i)/norm);
				}
			}
			break;
		}
		case NONE:{
			break;
		}
		default:{
			CV_Error(CV_StsBadArg, "invalid norm feature type");
			break;
		}
	}

	return bof;
}