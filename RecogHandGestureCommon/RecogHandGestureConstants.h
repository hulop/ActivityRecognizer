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

#ifndef __RECOG_HAND_GESTURE_CONSTANTS__
#define __RECOG_HAND_GESTURE_CONSTANTS__

#include "FisherVectorWrapper.h"
#include "BowWrapper.h"

// parameters for Dense Trajectories
static const bool dtUseTraj = true;
static const bool dtUseHOG = true;
static const bool dtUseHOF = true;
static const bool dtUseMBH = true;
//static const int dtTimeIntervalDetectKeypoint = 1;
//static const int dtDenseKeypointStep = 10; // original paper setting 5
static const int dtTrackMedianFilterSize = 1;
static const int dtPyramidLevels = 2; // original paper setting : 8
static const int dtMaxTimeLength = 15;
static const float dtPyramidScale = 1.0/sqrt(2.0);
static const int dtDescGrid2D = 16;
static const int dtDescGridTime = 5;
static const int dtDescNeighborSize = 32;
/////

// parameters for Hand Detection
static const bool USE_HAND_MASK_KEYPOINT_REMOVE = false;
static const string HAND_DETECT_FEATURE_SET = "rvl";
static const int HAND_DETECT_NUM_MODELS_TO_AVERAGE = 10;
static const int HAND_DETECT_STEP_SIZE = 3;
static const float HAND_DETECT_PROB_THRES = 0.01;
static const int HAND_DETECT_AVERAGE_PROBABILITY_TIME_LENGTH = 3;
/////

// parameters for Video Stabilization
static const bool USE_VIDEO_STABILIZATION = false;
/////

// parameters for Dense Hand HOG
static const bool USE_HAND_HOG = true;
static const Size HAND_HOG_BLOCK_SIZE = Size(16, 16);
static const Size HAND_HOG_STRIDE_SIZE = Size(8, 8);
static const Size HAND_HOG_CELL_SIZE = Size(8, 8);
static const int HAND_HOG_RAD_BINS = 9;
static const int DENSE_HAND_HOG_KEYPOINT_STEP = 16;
static const int DENSE_HAND_HOG_GRID_2D = 16;
static const int DENSE_HAND_HOG_NEIGHBOR_SIZE = 96;
static const int DENSE_HAND_HOG_BIN_NUM = 9;
static const bool DENSE_HAND_HOG_USE_ZERO_MAGNITUDE_BIN = false;
static const bool DENSE_HAND_HOG_FULL_RADIAN = true;
static const bool DENSE_HAND_HOG_WEIGHT_PROBABILITY = false;
/////

// parameters for SIFT
static const bool USE_SIFT = false;
/////

// parameters for Fisher Vector
static const int PCA_TRAIN_FEATURE_NUM = 30000;
static const int PCA_TRAIN_FEATURE_NUM_PER_IMAGE = 100;
static const int FV_TRAIN_FEATURE_NUM = 300000;
static const int FV_TRAIN_FEATURE_NUM_PER_IMAGE = 100;

static const int FV_PCA_DIM = 32;
static const int FV_K = 256;
static const FisherVectorWrapper::NormFisherVectorFeatureType NORM_FISHER_VECTOR_FEATURE_TYPE = FisherVectorWrapper::LP_NORM;
/////

// parameters specific for Bag of Words (These are used by SIFT BoW baseline method)
static const int BOW_TRAIN_FEATURE_NUM = 300000;
static const int BOW_TRAIN_FEATURE_NUM_PER_IMAGE = 100;
static const int BOW_K = 1000;
static const BowWrapper::NormBowFeatureType NORM_BOW_FEATURE_TYPE = BowWrapper::L1_NORM_SQUARE_ROOT;
/////

// parameters for Linear SVM
static const bool USE_CROSS_VALIDATION_TRAIN_SVM = false;
static const float DEFAULT_TRAIN_SVM_C = 1.0;
static const float SCALE_FEATURE_BEFORE_SVM_MIN = -1.0;
static const float SCALE_FEATURE_BEFORE_SVM_MAX = 1.0;
/////

#endif __RECOG_HAND_GESTURE_CONSTANTS__