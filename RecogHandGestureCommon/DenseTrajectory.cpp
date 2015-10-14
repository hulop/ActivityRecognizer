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

#include "DenseTrajectory.h"
#include <iterator>
#include <omp.h>
#include "TimerUtils.h"
#include "RecogHandGestureConstants.h"
#ifdef USE_GPU_DENSE_FLOW
#include <opencv2/gpu/gpu.hpp>
#endif

// parameters for Gunnar Farneback dense optical flow
static const double gfPyrScale = sqrt(2.0)/2.0;
static const int gfLevels = 5;
static const int gfWinsize = 10;
static const int gfIterations = 2;
static const int gfPolyN = 7;
static const double gfPolySigma = 1.5;
static const int gfFlags = OPTFLOW_USE_INITIAL_FLOW | OPTFLOW_FARNEBACK_GAUSSIAN;
static const int gfFlagsGPU = OPTFLOW_FARNEBACK_GAUSSIAN; // OPTFLOW_USE_INITIAL_FLOW do not work for GPU
/////

// parameters for selecting tracking points by Shi-Tomashi method
// http://docs.opencv.org/doc/tutorials/features2d/trackingmotion/generic_corner_detector/generic_corner_detector.html
static const double shiTomasiQuality = 0.001;
static const int shiTomasiBlockSize = 3;
static const int shiTomasiApertureSize = 3;
/////

// parameters for HOG/HOF
static const int binNum = 8;
static const bool useZeroMagnitudeBinHog = false;
static const bool useZeroMagnitudeBinHof = true;
static const bool useZeroMagnitudeBinMbh = false;
static const bool fullRadianHogHofMbh = true;
/////

DenseTrajectory::DenseTrajectory(bool useTraj, bool useHOG, bool useHOF, bool useMBH, Size capSize, int timeIntervalDetectKeypoint, int denseKeypointStep, 
								int trackMedianFilterSize, int pyramidLevels, int maxTimeLength, float pyramidScale, int descGrid2D, int descGridTime, 
								int descNeighborSize)
{
	mDebug = 1;

	mUseTraj = useTraj;
	mUseHOG = useHOG;
	mUseHOF = useHOF;
	mUseMBH = useMBH;

	mCapSize = capSize;
	mTimeIntevalDetectKeypoint = timeIntervalDetectKeypoint;
	mTimeCountDetectKeypoint = 0;

	mDenseKeypointStep = denseKeypointStep;
	mTrackMedianFilterSize = trackMedianFilterSize;
	mPyramidLevels = pyramidLevels;
	mMaxTimeLength = maxTimeLength;
	mPyramidScale = pyramidScale;
	mDescGrid2D = descGrid2D;
	mDescGridTime = descGridTime;
	mDescNeighborSize = descNeighborSize;
	cout << "desc block size in 2D : " << mDescGrid2D << endl;
	cout << "desc block size in Time : " << mDescGridTime << endl;
	cout << "desc neighbor size around each track point : " << mDescNeighborSize << endl;

	mHog = new HogHofMbh(descGrid2D, descGrid2D, descGrid2D, binNum, useZeroMagnitudeBinHog, fullRadianHogHofMbh);
	mHof = new HogHofMbh(descGrid2D, descGrid2D, descGrid2D, binNum, useZeroMagnitudeBinHof, fullRadianHogHofMbh);
	mMbh = new HogHofMbh(descGrid2D, descGrid2D, descGrid2D, binNum, useZeroMagnitudeBinMbh, fullRadianHogHofMbh);
}

DenseTrajectory::~DenseTrajectory()
{
	delete mHog;
	delete mHof;
	delete mMbh;
}

void DenseTrajectory::detectKeypointsPyramid(const vector<Mat>& grayPyramid, const vector<Mat>& maskPyramid, vector<vector<cv::KeyPoint> >& keypointsPyramid)
{
	keypointsPyramid.resize(grayPyramid.size());

	#pragma omp parallel for
	for (int p=0; p<grayPyramid.size(); p++) {
		DenseFeatureDetector detector(1.f, 1, 0.1f, mDenseKeypointStep, (mDescNeighborSize+1)/2);

		vector<KeyPoint> denseKeypoints;
		detector.detect(grayPyramid[p], denseKeypoints);

		Mat shiTomasiDst = Mat::zeros(grayPyramid[p].size(), CV_32FC1);
		cornerMinEigenVal(grayPyramid[p], shiTomasiDst, shiTomasiBlockSize, shiTomasiApertureSize, BORDER_DEFAULT);
		double shiTomasiMinVal, shiTomasiMaxVal;
		minMaxLoc(shiTomasiDst, &shiTomasiMinVal, &shiTomasiMaxVal, 0, 0, Mat());
		double shiTomasiThres = shiTomasiMaxVal*shiTomasiQuality;

		vector<KeyPoint> keypoints;
		if (!maskPyramid[p].empty()) {
			for (int k=0; k<denseKeypoints.size(); k++) {
				if ((int)maskPyramid[p].at<uchar>(denseKeypoints[k].pt.y, denseKeypoints[k].pt.x) > 0 
					&& shiTomasiDst.at<float>(denseKeypoints[k].pt.y, denseKeypoints[k].pt.x) > shiTomasiThres) {
					keypoints.push_back(denseKeypoints[k]);
				}
			}
		} else {
			for (int k=0; k<denseKeypoints.size(); k++) {
				if (shiTomasiDst.at<float>(denseKeypoints[k].pt.y, denseKeypoints[k].pt.x) > shiTomasiThres) {
					keypoints.push_back(denseKeypoints[k]);
				}
			}
		}
		keypointsPyramid[p] = keypoints;
	}
}

void DenseTrajectory::trackKeypointsMedianFilterFlow(const cv::Mat& flow, int pyramidLevel, vector<cv::Point2f>& curPoints)
{
	curPoints.resize(mTrackPointsPyramid[pyramidLevel].size());

	#pragma omp parallel for
	for (int k=0; k<mTrackPointsPyramid[pyramidLevel].size(); k++) {
		int t = mTrackPointsPyramid[pyramidLevel][k].getTimeLength();
		CV_Assert(t>0);

		const Point2f& prevPoint = mTrackPointsPyramid[pyramidLevel][k].getPointHistory(t-1);

		vector<float> fx;
		vector<float> fy;
		for (int y=std::max<int>(prevPoint.y-mTrackMedianFilterSize, 0); y<=std::min<int>(prevPoint.y+mTrackMedianFilterSize, flow.rows-1); y++) {
			for (int x=std::max<int>(prevPoint.x-mTrackMedianFilterSize, 0); x<=std::min<int>(prevPoint.x+mTrackMedianFilterSize, flow.cols-1); x++) {
				const Point2f& fxy = flow.at<Point2f>(y, x);
				fx.push_back(fxy.x);
				fy.push_back(fxy.y);
			}
		}

		if (fx.size()>=4 && fy.size()>=4) {
			sort(fx.begin(), fx.end());
			sort(fy.begin(), fy.end());

			Point2f curPoint;
			curPoint.x = prevPoint.x + fx[fx.size()/2];
			curPoint.y = prevPoint.y + fy[fy.size()/2];
			curPoints[k] = curPoint;
		}
	}
}

bool isTrackPointMarkedDelete(const DenseTrajectoryTrackPoint& trackPoint) {
	return trackPoint.markedDelete;
}

Mat DenseTrajectory::update(Mat frame, Mat mask)
{
	CV_Assert(mask.empty() || frame.size()==mask.size());
	Mat gray;
	cvtColor(frame, gray, CV_BGR2GRAY);

	Mat grayMask;
	if (!mask.empty() && mask.type()!=CV_8UC1) {
		mask.convertTo(grayMask, CV_8UC1);
	} else {
		grayMask = mask;
	}

	// calculate pyramid images
	vector<cv::Mat> grayPyramid;
	vector<cv::Mat> maskPyramid;
	float scalePyramid = mPyramidScale;
	grayPyramid.push_back(gray);
	maskPyramid.push_back(grayMask);
	for (int p=1; p<mPyramidLevels; p++) {
		cv::Mat scaleGray;
		cv::resize(gray, scaleGray, cv::Size(), scalePyramid, scalePyramid);
		grayPyramid.push_back(scaleGray);

		if (!mask.empty()) {
			cv::Mat scaleMask;
			cv::resize(grayMask, scaleMask, cv::Size(), scalePyramid, scalePyramid);
			maskPyramid.push_back(grayMask);
		} else {
			maskPyramid.push_back(grayMask);
		}

		scalePyramid *= mPyramidScale;
	}

	// calculate Gunnar Farneback dense optical flow
	if (mPrevGrayPyramid.size()==0) { // initial frame : detect keypoints
		vector<vector<cv::KeyPoint> > keypointsPyramid;
		detectKeypointsPyramid(grayPyramid, maskPyramid, keypointsPyramid);

		mTrackPointsPyramid.resize(keypointsPyramid.size());

		#pragma omp parallel for
		for (int p=0; p<keypointsPyramid.size(); p++) {
			// create empty integral image for initial frame
			vector<Mat> emptyIImage;

			vector<DenseTrajectoryTrackPoint> trackPoints(keypointsPyramid[p].size());

			for (int k=0; k<keypointsPyramid[p].size(); k++) {
				DenseTrajectoryTrackPoint trackPoint(grayPyramid[p].size(), mMaxTimeLength, mDescNeighborSize, mDescGridTime);
				trackPoint.addPointToHistory(keypointsPyramid[p][k].pt);

				if (mUseHOG) {
					trackPoint.updateHogHistory(keypointsPyramid[p][k].pt, *mHog, emptyIImage);
				}
				if (mUseHOF) {
					trackPoint.updateHofHistory(keypointsPyramid[p][k].pt, *mHof, emptyIImage);
				}
				if (mUseMBH) {
					trackPoint.updateMbhHistory(keypointsPyramid[p][k].pt, *mMbh, emptyIImage, emptyIImage);
				}

				trackPoints[k] = trackPoint;
			}

			mTrackPointsPyramid[p] = trackPoints;
		}
	} else { // not inital frame : track keypoints
		vector<vector<Mat>> hogIImagePyramid(grayPyramid.size());
		vector<vector<Mat>> hofIImagePyramid(grayPyramid.size());
		vector<vector<Mat>> mbhxIImagePyramid(grayPyramid.size());
		vector<vector<Mat>> mbhyIImagePyramid(grayPyramid.size());

		TimerUtils::getInstance().start();
		
		#pragma omp parallel for
		for (int p=0; p<grayPyramid.size(); p++) {
			// calc dense optical flow
			double timeOptialFlow = double(getTickCount());
			cv::Mat flow;
#ifdef USE_GPU_DENSE_FLOW
			int cudaDevices = cv::gpu::getCudaEnabledDeviceCount();
			if (cudaDevices > 0) {
				gpu::GpuMat gpuGray(grayPyramid[p]), gpuPrevGray(mPrevGrayPyramid[p]), gpuFlowX, gpuFlowY;

				cv::gpu::FarnebackOpticalFlow gpuFarneback;
				gpuFarneback.numLevels = gfLevels;
				gpuFarneback.pyrScale = gfPyrScale;
				gpuFarneback.winSize = gfWinsize;
				gpuFarneback.numIters = gfIterations;
				gpuFarneback.polyN = gfPolyN;
				gpuFarneback.polySigma = gfPolySigma;
				gpuFarneback.flags = gfFlagsGPU;
				gpuFarneback(gpuGray, gpuPrevGray, gpuFlowX, gpuFlowY);

				Mat flowXY[2];
				gpuFlowX.download(flowXY[0]);
				gpuFlowY.download(flowXY[1]);
				merge(flowXY, 2, flow);
			} else {
#endif
				cv::calcOpticalFlowFarneback(mPrevGrayPyramid[p], grayPyramid[p], flow, gfPyrScale, gfLevels, gfWinsize, gfIterations, gfPolyN, gfPolySigma, gfFlags);
#ifdef USE_GPU_DENSE_FLOW
			}
#endif
			cout << "time to calculate optical flow [level=" << p << "]: " << (getTickCount() - timeOptialFlow) / getTickFrequency() << endl;

			// calc HOG/HOF/MBH integral images
			double timeIntegralImage = double(getTickCount());
			#pragma omp parallel for
			for (int iImageType=0; iImageType<3; iImageType++) {
				// calculate HOG for pyramid images
				if (iImageType==0 && mUseHOG) {
					vector<Mat> hogIImage;
					mHog->calcHogIImage(grayPyramid[p], hogIImage);
					hogIImagePyramid[p] = hogIImage;
				}
				// calc HOF for pyramid image
				if (iImageType==1 && mUseHOF) {
					vector<Mat> hofIImage;
					mHof->calcHofIImage(flow, hofIImage);
					hofIImagePyramid[p] = hofIImage;
				}
				// calc MBHx/MBHy for pyramid image
				if (iImageType==2 && mUseMBH) {
					vector<Mat> mbhxIImage;
					vector<Mat> mbhyIImage;
					mMbh->calcMbhIImage(flow, mbhxIImage, mbhyIImage);
					mbhxIImagePyramid[p] = mbhxIImage;
					mbhyIImagePyramid[p] = mbhyIImage;
				}
			}
			cout << "time to calculate integral image [level=" << p << "]: " << (getTickCount()-timeIntegralImage)/getTickFrequency() << endl;

			// track points
			vector<Point2f> curPoints;
			trackKeypointsMedianFilterFlow(flow, p, curPoints);

			// update points history
			double timeUpdatePoint = double(getTickCount());
			for (int k=0; k<mTrackPointsPyramid[p].size(); k++) {
				mTrackPointsPyramid[p][k].addPointToHistory(curPoints[k]);

				int r = (int)floor(.5+curPoints[k].y);
				int c = (int)floor(.5+curPoints[k].x);
				// If keypoints go outside of window, do not calculate feature. This point will be removed.
				if (c>=mDescGrid2D && c<flow.cols-mDescGrid2D && r>=mDescGrid2D && r<flow.rows-mDescGrid2D) {
					if (mUseHOG) {
						mTrackPointsPyramid[p][k].updateHogHistory(curPoints[k], *mHog, hogIImagePyramid[p]);
					}
					if (mUseHOF) {
						mTrackPointsPyramid[p][k].updateHofHistory(curPoints[k], *mHof, hofIImagePyramid[p]);
					}
					if (mUseMBH) {
						mTrackPointsPyramid[p][k].updateMbhHistory(curPoints[k], *mMbh, mbhxIImagePyramid[p], mbhyIImagePyramid[p]);
					}
				}
			}
			cout << "time to calculate update point [level=" << p << "]: " << (getTickCount()-timeUpdatePoint)/getTickFrequency() << endl;

			// remove track points which are old or go outside of window 
			double timeRemovePoint = double(getTickCount());
			// too slow to use erase of vector class
			/*
			for (vector<DenseTrajectoryTrackPoint>::iterator iter=mTrackPointsPyramid[p].begin(); iter!=mTrackPointsPyramid[p].end(); ) {
				int t = iter->getTimeLength();
				CV_Assert(t>0);

				Point2f curPoint = iter->getPointHistory(t-1);
				int r = (int)floor(.5+curPoint.y);
				int c = (int)floor(.5+curPoint.x);
				if (t>mMaxTimeLength+1 || c<mDescGrid2D || c>=flow.cols-mDescGrid2D || r<mDescGrid2D || r>=flow.rows-mDescGrid2D
					|| (!maskPyramid[p].empty() && (int)maskPyramid[p].at<uchar>(r, c)==0)) {
					//cout << "remove track points outside window" << endl;
					iter = mTrackPointsPyramid[p].erase(iter);
				} else {
					++iter;
				}
			}
			*/
			// much faster version to use remove_if of vector class
			for (int k=0; k<mTrackPointsPyramid[p].size(); k++) {
				int t = mTrackPointsPyramid[p][k].getTimeLength();
				CV_Assert(t>0);

				Point2f curPoint = mTrackPointsPyramid[p][k].getPointHistory(t-1);
				int r = (int)floor(.5+curPoint.y);
				int c = (int)floor(.5+curPoint.x);
				if (t>mMaxTimeLength+1 || c<mDescGrid2D || c>=flow.cols-mDescGrid2D || r<mDescGrid2D || r>=flow.rows-mDescGrid2D
					|| (!maskPyramid[p].empty() && (int)maskPyramid[p].at<uchar>(r, c)==0)) {
					//cout << "remove track points outside window" << endl;
					mTrackPointsPyramid[p][k].markedDelete = true;
				}
			}
			mTrackPointsPyramid[p].erase(std::remove_if(mTrackPointsPyramid[p].begin(), mTrackPointsPyramid[p].end(), isTrackPointMarkedDelete), 
										mTrackPointsPyramid[p].end());
			cout << "time to calculate remove point [level=" << p << "]: " << (getTickCount()-timeRemovePoint)/getTickFrequency() << endl;

			if (mDebug && p==0) {
				visualizeDenseOpticalFlow(flow);
			}
			if (mDebug && mUseHOG && p==0) {
				visualizeHogHofMbh(hogIImagePyramid[p], binNum, "HOG");
			}
			if (mDebug && mUseHOF && p==0) {
				visualizeHogHofMbh(hofIImagePyramid[p], binNum, "HOF");
			}
			if (mDebug && mUseMBH && p==0) {
				visualizeHogHofMbh(mbhxIImagePyramid[p], binNum, "MBHx");
				visualizeHogHofMbh(mbhyIImagePyramid[p], binNum, "MBHy");
			}
		}
		cout << "	time to update tracked keypoints : " << TimerUtils::getInstance().stop() << " secs." << endl;

		if (mTimeCountDetectKeypoint>=mTimeIntevalDetectKeypoint) {
			mTimeCountDetectKeypoint = 0;

			// add keypoints which do not have nearby tracked points 
			TimerUtils::getInstance().start();

			vector<vector<cv::KeyPoint> > keypointsPyramid;
			detectKeypointsPyramid(grayPyramid, maskPyramid, keypointsPyramid);
			cout << "	time to detect new keypoints : " << TimerUtils::getInstance().stop() << " secs." << endl;

			TimerUtils::getInstance().start();

			#pragma omp parallel for
			for (int p=0; p<keypointsPyramid.size(); p++) {
				cv::Mat numNearbyTrackPoints = cv::Mat::zeros(grayPyramid[p].size(), CV_32S);
				for (int k=0; k<mTrackPointsPyramid[p].size(); k++) {
					int t = mTrackPointsPyramid[p][k].getTimeLength();
					CV_Assert(t>0);

					Point2f point = mTrackPointsPyramid[p][k].getPointHistory(t-1);
					int y = point.y;
					int x = point.x;
					for (int m=std::max<int>(y-mDenseKeypointStep, 0); m<=std::min<int>(y+mDenseKeypointStep, numNearbyTrackPoints.rows-1); m++) {
						for (int n=std::max<int>(x-mDenseKeypointStep, 0); n<=std::min<int>(x+mDenseKeypointStep, numNearbyTrackPoints.cols-1); n++) {
							numNearbyTrackPoints.at<int>(m,n) += 1;
						}
					}
				}

				// create empty integral image for initial frame
				vector<Mat> emptyIImage;

				for (int k=0; k<keypointsPyramid[p].size(); k++) {
					if (numNearbyTrackPoints.at<int>(keypointsPyramid[p][k].pt.y, keypointsPyramid[p][k].pt.x)==0) {
						DenseTrajectoryTrackPoint trackPoint(grayPyramid[p].size(), mMaxTimeLength, mDescNeighborSize, mDescGridTime);
						trackPoint.addPointToHistory(keypointsPyramid[p][k].pt);

						if (mUseHOG) {
							trackPoint.updateHogHistory(keypointsPyramid[p][k].pt, *mHog, emptyIImage);
						}
						if (mUseHOF) {
							trackPoint.updateHofHistory(keypointsPyramid[p][k].pt, *mHof, emptyIImage);
						}
						if (mUseMBH) {
							trackPoint.updateMbhHistory(keypointsPyramid[p][k].pt, *mMbh, emptyIImage, emptyIImage);
						}

						mTrackPointsPyramid[p].push_back(trackPoint);
					}
				}
			}
			cout << "	time to add new keypoints : " << TimerUtils::getInstance().stop() << " secs." << endl;
		} else {
			mTimeCountDetectKeypoint++;
			cout << "	Skip to detect new keypoints." << endl;
		}
	}

	mPrevGrayPyramid.clear();
	copy(grayPyramid.begin(), grayPyramid.end(), back_inserter(mPrevGrayPyramid));

	Mat visualizePoint;
	if (mDebug) {
		visualizePoint = visualizeTrackPoints(frame);
	}
	return visualizePoint;
}

Mat DenseTrajectory::getDesc()
{
	Mat descriptors;
	for (int p=0; p<mTrackPointsPyramid.size(); p++) {
		for (int k=0; k<mTrackPointsPyramid[p].size(); k++) {
			if (mTrackPointsPyramid[p][k].getTimeLength()>mMaxTimeLength) {
				Mat desc;
				if (mUseTraj) {
					desc = mTrackPointsPyramid[p][k].getTrajDesc();
					CV_Assert(checkRange(desc));
				}
				if (mUseHOG) {
					Mat hog = mTrackPointsPyramid[p][k].getHogDesc();
					CV_Assert(checkRange(hog));
					if (desc.empty()) {
						desc = hog;
					} else {
						hconcat(desc, hog, desc);
					}
				}
				if (mUseHOF) {
					Mat hof = mTrackPointsPyramid[p][k].getHofDesc();
					CV_Assert(checkRange(hof));
					if (desc.empty()) {
						desc = hof;
					} else {
						hconcat(desc, hof, desc);
					}
				}
				if (mUseMBH) {
					Mat mbh = mTrackPointsPyramid[p][k].getMbhDesc();
					CV_Assert(checkRange(mbh));
					if (desc.empty()) {
						desc = mbh;
					} else {
						hconcat(desc, mbh, desc);
					}
				}
				descriptors.push_back(desc);
			}
		}
	}
	return descriptors;
}

Mat DenseTrajectory::visualizeTrackPoints(cv::Mat input)
{
	Mat flowLevel0;
	for (int p=0; p<mPrevGrayPyramid.size(); p++) {
		cv::Mat flowImage;
		cv::resize(input, flowImage, mPrevGrayPyramid[p].size());
		for (int k=0; k<mTrackPointsPyramid[p].size(); k++) {
			int t = mTrackPointsPyramid[p][k].getTimeLength();
			CV_Assert(t>0);

			if (t>1) {
				Point2f p1 = mTrackPointsPyramid[p][k].getPointHistory(t-1);
				Point2f p2 = mTrackPointsPyramid[p][k].getPointHistory(t-2);
				if (p1.x>0 && p1.x<flowImage.cols && p1.y>0 && p1.y<flowImage.rows 
					&& p2.x>0 && p2.x<flowImage.cols && p2.y>0 && p2.y<flowImage.rows) {
					line(flowImage, p2, p1, CV_RGB(0, 255, 0));
					circle(flowImage, p1, 2, CV_RGB(255, 0, 0), -1);
				}
			} else if (t==0) {
				Point2f p1 = mTrackPointsPyramid[p][k].getPointHistory(t-1);
				if (p1.x>0 && p1.x<flowImage.cols && p1.y>0 && p1.y<flowImage.rows) {
					circle(flowImage, p1, 2, CV_RGB(255, 0, 0), -1);
				}
			}
		}
		if (p==0) {
			flowLevel0 = flowImage;
		}

		stringstream ss;
		ss << "Farneback Dense Optical Flow Image, Level : " << (p+1);
		cv::imshow(ss.str(), flowImage);
	}
	return flowLevel0;
}

void DenseTrajectory::visualizeDenseOpticalFlow(Mat flow)
{
	Mat flowXY[2];
	split(flow, flowXY);

	Mat magnitude, angle;
	cartToPolar(flowXY[0], flowXY[1], magnitude, angle, true);

	Mat hsvPlanes[3];
	hsvPlanes[0] = angle;
	normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
	hsvPlanes[1] = magnitude;
	hsvPlanes[2] = Mat::ones(angle.size(), CV_32F);
	Mat hsv;
	merge(hsvPlanes, 3, hsv);
	cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
	imshow("Dense Optical Flow", hsv);
}

void DenseTrajectory::visualizeHogHofMbh(vector<Mat>& intImage, int binNum, string windowName)
{
	static const float VISUALIZE_MAX_MAGNITUDE = 100.0;

	float radianPerBin;
	if (fullRadianHogHofMbh) {
		radianPerBin = (float)2.0*CV_PI/binNum;
	} else {
		radianPerBin = (float)CV_PI/binNum;
	}

	Mat magnitude = Mat::zeros(intImage[0].size(), CV_32F);
	Mat angle = Mat::zeros(intImage[0].size(), CV_32F);
	for (int y=0; y<intImage[0].rows; y++) {
		for (int x=0; x<intImage[0].cols; x++) {
			int maxBinIndex = -1;
			float maxBinMagnitude = 0.0;
			for (int b=0; b<intImage.size(); b++) {
				float magnitude;
				if (y==0 && x==0) {
					magnitude = intImage[b].at<float>(y,x);
				} else if (y==0) {
					magnitude = intImage[b].at<float>(y,x) - intImage[b].at<float>(y,x-1);
				} else if (x==0) {
					magnitude = intImage[b].at<float>(y,x) - intImage[b].at<float>(y-1,x);
				} else {
					magnitude = intImage[b].at<float>(y,x) - intImage[b].at<float>(y,x-1) - intImage[b].at<float>(y-1,x) + intImage[b].at<float>(y-1,x-1);
				}
				if (magnitude>maxBinMagnitude) {
					maxBinIndex = b;
					maxBinMagnitude = magnitude;
				}
			}
			if (maxBinIndex>=0) {
				angle.at<float>(y,x) = (180.0/CV_PI)*(radianPerBin*maxBinIndex + radianPerBin*0.5);
				magnitude.at<float>(y,x) = min<float>(maxBinMagnitude, VISUALIZE_MAX_MAGNITUDE);
			}
		}
	}

	Mat hsv3[3];
	hsv3[0] = angle;
	normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
	hsv3[1] = magnitude;
	hsv3[2] = Mat::ones(angle.size(), CV_32F);
	Mat hsv;
	merge(hsv3, 3, hsv);
	cvtColor(hsv, hsv, cv::COLOR_HSV2BGR);
	imshow(windowName, hsv);
}
