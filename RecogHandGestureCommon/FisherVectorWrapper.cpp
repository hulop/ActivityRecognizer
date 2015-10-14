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

#include "FisherVectorWrapper.h"

static const int EM_ITERATION = 100;
static const double EM_EPSILON = DBL_EPSILON;
static const int GMM_INITIALIZE_KMEANS_ITERATION = 100;

FisherVectorWrapper::FisherVectorWrapper()
{
}

FisherVectorWrapper::FisherVectorWrapper(const cv::Mat &gmmTrainFeatures, int _K, NormFisherVectorFeatureType _NORM_FISHER_VECTOR_FEATURE_TYPE)
{
	K = _K;
	NORM_FISHER_VECTOR_FEATURE_TYPE = _NORM_FISHER_VECTOR_FEATURE_TYPE;

	cout << "Start train VLFeatEM...." << endl;
	gmm = calcVLFeatGMM(gmmTrainFeatures, K, EM_ITERATION, EM_EPSILON, weights, means, covs);
	cout << "End train VLFeatEM." << endl;

	normWeights = cv::Mat(1, weights.cols, CV_64F);
	logNormWeights = cv::Mat(1, weights.cols, CV_64F);
	double normW = 0.0;
	for (int i=0; i<weights.cols; i++) {
		normW += exp(weights.at<double>(i));
	}
	for (int i=0; i<weights.cols; i++) {
		normWeights.at<double>(i) = exp(weights.at<double>(i))/normW;
		logNormWeights.at<double>(i) = log(normWeights.at<double>(i));
	}
	cout << "weight : " << weights << endl;
	cout << "normWeights : " << normWeights << endl;
	cout << "logNormWeights : " << logNormWeights << endl;
	cout << "End train EM. Number of cluster is " << K << endl;
}

FisherVectorWrapper::~FisherVectorWrapper()
{
	vl_gmm_delete(gmm);
}

void FisherVectorWrapper::write(FileStorage& fs) const
{
	CV_Assert(NORM_FISHER_VECTOR_FEATURE_TYPE>=0);	
    fs << "NORM_FISHER_VECTOR_FEATURE_TYPE" << (NORM_FISHER_VECTOR_FEATURE_TYPE == FisherVectorWrapper::NONE ? string("NONE") :
												NORM_FISHER_VECTOR_FEATURE_TYPE == FisherVectorWrapper::L2_NORM ? string("L2") :
												NORM_FISHER_VECTOR_FEATURE_TYPE == FisherVectorWrapper::LP_NORM ? string("LP") :
												format("unknown_%d", NORM_FISHER_VECTOR_FEATURE_TYPE));

	fs << "K" << K;

	fs << "weights" << weights;
	fs << "means" << means;
	fs << "covs" << covs;

	fs << "gmm_dimension" << (int)vl_gmm_get_dimension(gmm);
	fs << "gmm_numClusters" << (int)vl_gmm_get_num_clusters(gmm);
	fs << "gmm_maxNumIterations" << (int)vl_gmm_get_max_num_iterations(gmm);
	fs << "gmm_numRepetitions" << (int)vl_gmm_get_num_repetitions(gmm);
	fs << "gmm_verbosity" << vl_gmm_get_verbosity(gmm);
}

void FisherVectorWrapper::read(const FileNode& node)
{
	string normFisherVectorTypeStr;
	node["NORM_FISHER_VECTOR_FEATURE_TYPE"] >> normFisherVectorTypeStr;
	NORM_FISHER_VECTOR_FEATURE_TYPE = normFisherVectorTypeStr == "NONE" ? FisherVectorWrapper::NONE :
									normFisherVectorTypeStr == "L2" ? FisherVectorWrapper::L2_NORM :
									normFisherVectorTypeStr == "LP" ? FisherVectorWrapper::LP_NORM :
									FisherVectorWrapper::UNKNOWN;
	CV_Assert(NORM_FISHER_VECTOR_FEATURE_TYPE>=0);

	node["K"] >> K;

	node["weights"] >> weights;
	node["means"] >> means;
	node["covs"] >> covs;

	normWeights = cv::Mat(1, weights.cols, CV_64F);
	logNormWeights = cv::Mat(1, weights.cols, CV_64F);
	double normW = 0.0;
	for (int i=0; i<weights.cols; i++) {
		normW += exp(weights.at<double>(i));
	}
	for (int i=0; i<weights.cols; i++) {
		normWeights.at<double>(i) = exp(weights.at<double>(i))/normW;
		logNormWeights.at<double>(i) = log(normWeights.at<double>(i));
	}

	int gmmDimension;
	node["gmm_dimension"] >> gmmDimension;
	
	int gmmNumClusters;
	node["gmm_numClusters"] >> gmmNumClusters;
	
	gmm = vl_gmm_new(VL_TYPE_FLOAT, gmmDimension, gmmNumClusters);
	vl_gmm_set_initialization(gmm,VlGMMKMeans);

	int gmmMaxNumIterations;
	node["gmm_maxNumIterations"] >> gmmMaxNumIterations;
	vl_gmm_set_max_num_iterations(gmm, gmmMaxNumIterations);

	int gmmNumRepetitions;
	node["gmm_numRepetitions"] >> gmmNumRepetitions;
	vl_gmm_set_num_repetitions(gmm, gmmNumRepetitions);

	int gmmVerbosity;
	node["gmm_verbosity"] >> gmmVerbosity;
	vl_gmm_set_verbosity(gmm, gmmVerbosity);

	float* gmmPriors = (float*)vl_calloc(gmmNumClusters, sizeof(float));
	for (int k=0; k<gmmNumClusters; k++) {
		gmmPriors[k] = weights.at<double>(k);
	}
	vl_gmm_set_priors(gmm, gmmPriors);

	float* gmmMeans = (float*)vl_calloc(gmmNumClusters * gmmDimension, sizeof(float));
	for (int k=0; k<gmmNumClusters; k++) {
		for (int d=0; d<gmmDimension; d++) {
			gmmMeans[k*gmmDimension + d] = means.at<double>(k, d);
		}
	}
	vl_gmm_set_means(gmm, gmmMeans);

	float* gmmCovariances = (float*)vl_calloc(gmmNumClusters * gmmDimension, sizeof(float));
	for (int k=0; k<gmmNumClusters; k++) {
		for (int d=0; d<gmmDimension; d++) {
			gmmCovariances[k*gmmDimension + d] = covs[k].at<double>(d, d);
		}
	}
	vl_gmm_set_covariances(gmm, gmmCovariances);
}

VlGMM* FisherVectorWrapper::calcVLFeatGMM(const cv::Mat& features, const int K, const int maxIter, const double eps, 
										cv::Mat& weights, cv::Mat& means, vector<cv::Mat>& covs)
{
	VlKMeans* kmeans = 0;

	double sigmaLowerBound = 0.000001;
	
	vl_size numClusters = K;
	vl_size numData = features.rows;
	vl_size dimension = features.cols;
	vl_size maxiter = maxIter;
	vl_size maxrep = 1;
	
	vl_size maxiterKM = GMM_INITIALIZE_KMEANS_ITERATION;
	vl_size ntrees = 3;
	vl_size maxComp = 20;
	
	float* vl_data = (float*)vl_malloc(sizeof(float)*numData*dimension);

	vl_set_num_threads(0) ; /* use the default number of threads */
	
	for (int i=0; i<numData; i++) {
		for (int d=0; d<dimension; d++) {
			if (features.type()==CV_64F) {
				vl_data[i*dimension+d] = features.at<double>(i, d);
			} else if (features.type()==CV_32F) {
				vl_data[i*dimension+d] = features.at<float>(i, d);
			} else {
				vl_data[i*dimension+d] = features.at<unsigned int>(i, d);
			}
		}
	}
	
	VlGMM* gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters) ;

	kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
	vl_kmeans_set_verbosity(kmeans, 2);
	vl_kmeans_set_max_num_iterations(kmeans, maxiterKM) ;
	vl_kmeans_set_max_num_comparisons(kmeans, maxComp) ;
	vl_kmeans_set_num_trees(kmeans, ntrees);
	vl_kmeans_set_algorithm(kmeans, VlKMeansANN);
	vl_kmeans_set_initialization(kmeans, VlKMeansRandomSelection);
	vl_gmm_set_initialization(gmm,VlGMMKMeans);
	vl_gmm_set_kmeans_init_object(gmm,kmeans);
	
	vl_gmm_set_max_num_iterations(gmm, maxiter) ;
	vl_gmm_set_num_repetitions(gmm, maxrep);
	vl_gmm_set_verbosity(gmm, 2);
	vl_gmm_set_covariance_lower_bound(gmm,sigmaLowerBound);
	
	vl_gmm_cluster(gmm, vl_data, numData);
	
	float const* vl_covs = (float const*)vl_gmm_get_covariances(gmm);
	float const* vl_means = (float const*)vl_gmm_get_means(gmm);
	float const* vl_weights = (float const*)vl_gmm_get_priors(gmm);

	weights = cv::Mat(1, numClusters, CV_64F);
	for (int k=0; k<numClusters; k++) {
		weights.at<double>(k) = vl_weights[k];
	}
	means = cv::Mat(numClusters, dimension, CV_64F);
	for (int k=0; k<numClusters; k++) {
		for (int d=0; d<dimension; d++) {
			means.at<double>(k, d) = vl_means[k*dimension + d];
		}
	}
	covs.resize(numClusters);
	for (int k=0; k<numClusters; k++) {
		covs[k] = cv::Mat::zeros(dimension, dimension, CV_64F);
		for (int d=0; d<dimension; d++) {
			covs[k].at<double>(d, d) = vl_covs[k*dimension + d];
		}
	}

	vl_free(vl_data);
    vl_kmeans_delete(kmeans);

	return gmm;
}

cv::Mat FisherVectorWrapper::calcFisherVector(const cv::Mat &descriptors)
{
	CV_Assert(K>0);

	int localFeatureNum = descriptors.rows;
	int localFeatureDim = descriptors.cols;

	float* vl_data = (float*)vl_malloc(sizeof(float)*localFeatureNum*localFeatureDim);
	for (int i=0; i<localFeatureNum; i++) {
		for (int j=0; j<localFeatureDim; j++) {
			vl_data[i*localFeatureDim+j] = descriptors.at<float>(i,j);
		}
	}

	int vl_flag = VL_FISHER_FLAG_FAST;
	switch (NORM_FISHER_VECTOR_FEATURE_TYPE) {
		case L2_NORM:{
			vl_flag = VL_FISHER_FLAG_NORMALIZED;
			break;
		}
		case LP_NORM:{
			vl_flag = VL_FISHER_FLAG_IMPROVED;
			break;
		}
		case NONE:{
			vl_flag = VL_FISHER_FLAG_FAST;
			break;
		}
		default:{
			vl_flag = VL_FISHER_FLAG_FAST;
			break;
		}
	}

	float* vl_enc = (float*)vl_malloc(sizeof(float)*2*localFeatureDim*K);
    vl_fisher_encode(vl_enc, VL_TYPE_FLOAT, vl_gmm_get_means(gmm), localFeatureDim, K, 
					vl_gmm_get_covariances(gmm), vl_gmm_get_priors(gmm), vl_data, localFeatureNum, vl_flag);

	cv::Mat fv(2*localFeatureDim*K, 1, CV_64F);
	for (int i=0; i<2*localFeatureDim*K; i++) {
		fv.at<double>(i) = vl_enc[i];
	}
	//cout << "fv" << endl;
	//cout << fv << endl;

	vl_free(vl_data);
	vl_free(vl_enc);

	return fv;
}