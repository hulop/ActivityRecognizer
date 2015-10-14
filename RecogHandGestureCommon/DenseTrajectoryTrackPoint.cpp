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

#include "DenseTrajectoryTrackPoint.h"

DenseTrajectoryTrackPoint::DenseTrajectoryTrackPoint()
{
}

DenseTrajectoryTrackPoint::DenseTrajectoryTrackPoint(Size imageSize, int trajTimeLength, int cubeSize2D, int cubeSizeTime)
{
	CV_Assert(trajTimeLength>cubeSizeTime);

	this->markedDelete = false;
	this->trajTimeLength = trajTimeLength;
	this->cubeSize2D = cubeSize2D;
	this->cubeSizeTime = cubeSizeTime;
	this->shiftBlockSize = (cubeSize2D+1)/2;
	this->timeBlockNum = ceil((float)trajTimeLength/(float)cubeSizeTime);
	points.clear();
	hogDesc.clear();
	hofDesc.clear();
}

void DenseTrajectoryTrackPoint::addPointToHistory(Point2f point)
{
	points.push_back(point);
}

void DenseTrajectoryTrackPoint::updateDescHistory(Point2f point, HogHofMbh& hogHofMbh, vector<Mat>& intImage, vector<vector<float>>& descHistory)
{
	if (intImage.size()==0) {	// set empty descriptor for initial frame
		vector<float> desc;
		descHistory.push_back(desc);
	} else {
		int r = (int)floor(.5+point.y);
		int c = (int)floor(.5+point.x);

		int x1 = c-shiftBlockSize;
		int x2 = c+shiftBlockSize;
		int y1 = r-shiftBlockSize;
		int y2 = r+shiftBlockSize;

		CV_Assert(x1>=0 && y1>=0 && x2<intImage[0].size().width && y2<intImage[0].size().height);

		vector<float> desc;
		hogHofMbh.calcDescByIntegralImage(Rect(x1, y1, x2-x1, y2-y1), intImage, desc);
		descHistory.push_back(desc);
	}
}

void DenseTrajectoryTrackPoint::updateHogHistory(Point2f point, HogHofMbh& hog, vector<Mat>& hogIImage)
{
	updateDescHistory(point, hog, hogIImage, hogDesc);
}

void DenseTrajectoryTrackPoint::updateHofHistory(Point2f point, HogHofMbh& hof, vector<Mat>& hofIImage)
{
	updateDescHistory(point, hof, hofIImage, hofDesc);
}

void DenseTrajectoryTrackPoint::updateMbhHistory(Point2f point, HogHofMbh& mbh, vector<Mat>& mbhxIImage, vector<Mat>& mbhyIImage)
{
	updateDescHistory(point, mbh, mbhxIImage, mbhxDesc);
	updateDescHistory(point, mbh, mbhyIImage, mbhyDesc);
}

const Point2f& DenseTrajectoryTrackPoint::getPointHistory(int time)
{
	return points[time];
}

int DenseTrajectoryTrackPoint::getTimeLength()
{
	return points.size();
}

Mat DenseTrajectoryTrackPoint::getTrajDesc()
{
	CV_Assert(points.size()==trajTimeLength+1);

	Mat desc(1, 2*trajTimeLength, CV_32F);

	float norm = 0.0;
	for (int i=points.size()-trajTimeLength; i<points.size(); i++) {
		norm += cv::norm(points[i]-points[i-1]);
	}
	if (norm!=0.0) {
		int d = 0;
		for (int i=points.size()-trajTimeLength; i<points.size(); i++) {
			desc.at<float>(d) = (points[i].x - points[i-1].x)/norm;
			desc.at<float>(d+1) = (points[i].y - points[i-1].y)/norm;
			d += 2;
		}
	} else {
		for (int i=0; i<desc.cols; i++) {
			desc.at<float>(i) = 0.0;
		}
	}
	return desc;
}

Mat DenseTrajectoryTrackPoint::getHogDesc()
{
	CV_Assert(hogDesc.size()==trajTimeLength+1 && trajTimeLength>cubeSizeTime);

	int dim = hogDesc[hogDesc.size()-1].size();

	Mat desc = Mat::zeros(1, dim*timeBlockNum, CV_32F);

	for (int t=0; t<timeBlockNum; t++) {
		for (int i=t*cubeSizeTime+1; i<min<int>((t+1)*cubeSizeTime+1, trajTimeLength+1); i++) {
			CV_Assert(hogDesc[i].size()==dim);
			for (int j=0; j<dim; j++) {
				desc.at<float>(dim*t + j) += hogDesc[i][j];
			}
		}
	}

	return desc;
}

Mat DenseTrajectoryTrackPoint::getHofDesc()
{
	CV_Assert(hofDesc.size()==trajTimeLength+1 && trajTimeLength>cubeSizeTime);

	int dim = hofDesc[hofDesc.size()-1].size();

	Mat desc = Mat::zeros(1, dim*timeBlockNum, CV_32F);

	for (int t=0; t<timeBlockNum; t++) {
		for (int i=t*cubeSizeTime+1; i<min<int>((t+1)*cubeSizeTime+1, trajTimeLength+1); i++) {
			CV_Assert(hofDesc[i].size()==dim);
			for (int j=0; j<dim; j++) {
				desc.at<float>(dim*t + j) += hofDesc[i][j];
			}
		}
	}

	return desc;
}

Mat DenseTrajectoryTrackPoint::getMbhDesc()
{
	CV_Assert(mbhxDesc.size()==trajTimeLength+1 && mbhyDesc.size()==trajTimeLength+1 && trajTimeLength>cubeSizeTime);

	int dim = mbhxDesc[mbhxDesc.size()-1].size();

	Mat desc = Mat::zeros(1, dim*timeBlockNum*2, CV_32F);

	for (int t=0; t<timeBlockNum; t++) {
		for (int i=t*cubeSizeTime+1; i<min<int>((t+1)*cubeSizeTime+1, trajTimeLength+1); i++) {
			CV_Assert(mbhxDesc[i].size()==dim);
			CV_Assert(mbhyDesc[i].size()==dim);
			for (int j=0; j<dim; j++) {
				desc.at<float>(dim*t + j) += mbhxDesc[i][j];
				desc.at<float>(dim*timeBlockNum + dim*t + j) += mbhyDesc[i][j];
			}
		}
	}

	return desc;
}