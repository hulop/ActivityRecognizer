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

#ifndef __RECOG_HAND_GESTURE_COMMON_HOG__
#define __RECOG_HAND_GESTURE_COMMON_HOG__

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class HogHofMbh
{
private:
	int mBlockSize;
	int mBlockStride;
	int mCellSize;
	bool mUseZeroMagnitudeBin;
	int mRadBinNum;
	int mTotalBinNum;
	float mFullRadian;

	void calcDesc(const Mat& diffX, const Mat& diffY, vector<float>& hogDesc);
	void calcIntegralImage(const Mat& diffX, const Mat& diffY, vector<Mat>& intImage, const Mat& prob=cv::Mat());

public:
	HogHofMbh(int blockSize, int blockStride, int cellSize, int binNum, bool countZeroMagnitude, bool fullOrientation);

	void calcHogDesc(const Mat& image, vector<float>& hogDesc);
	void calcHofDesc(const Mat& flow, vector<float>& hofDesc);
	void calcMbhDesc(const Mat& flow, vector<float>& mbhxDesc, vector<float>& mbhyDesc);

	void calcHogIImage(const Mat& image, vector<Mat>& hogIImage);
	void calcHogIImage(const Mat& image, const Mat& prob, vector<Mat>& hogIImage);
	void calcHofIImage(const Mat& flow, vector<Mat>& hofIImage);
	void calcMbhIImage(const Mat& flow, vector<Mat>& mbhxIImage, vector<Mat>& mbhyIImage);
	int getDescByIntegralImageDim(const Size& size);
	void calcDescByIntegralImage(const Rect& rect, const vector<Mat>& intImage, vector<float>& desc);
};

#endif __RECOG_HAND_GESTURE_COMMON_HOG__