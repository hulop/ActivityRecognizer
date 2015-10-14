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

#include "HogHofMbh.h"

static const double thresholdZeroBin = 0.01;

HogHofMbh::HogHofMbh(int blockSize, int blockStride, int cellSize, int binNum, bool useZeroMagnitudeBin, bool fullOrientation)
{
	mBlockSize = blockSize;
	mBlockStride = blockStride;
	mCellSize = cellSize;
	mRadBinNum = binNum;
	mUseZeroMagnitudeBin = useZeroMagnitudeBin;
	if (useZeroMagnitudeBin) {
		mTotalBinNum = binNum + 1;
	} else {
		mTotalBinNum = binNum;
	}

	if (fullOrientation) {
		mFullRadian = 2.0*CV_PI;
	} else {
		mFullRadian = CV_PI;
	}
}

void HogHofMbh::calcDesc(const Mat& diffX, const Mat& diffY, vector<float>& desc)
{
	CV_Assert(diffX.type()==diffY.type() && diffX.size()==diffY.size());

	int numBlockCols = ceil((float)(diffX.cols-mBlockSize)/(float)mBlockStride)+1;
	int numBlockRows = ceil((float)(diffX.rows-mBlockSize)/(float)mBlockStride)+1;
	int numCellPerBlock = ceil((float)mBlockSize/(float)mCellSize);
	float binPerRadian = (float)mRadBinNum/mFullRadian;
	float radianPerBin = (float)mFullRadian/mRadBinNum;
	//cout << "numBlockCols : " << numBlockCols << endl;
	//cout << "numBlockRows : " << numBlockRows << endl;
	//cout << "binPerRadian : " << binPerRadian << endl;
	//cout << "radianPerBin : " << radianPerBin << endl;

	Mat mag(diffX.rows, diffX.cols, CV_32F);
	Mat theta(diffX.rows, diffX.cols, CV_32F);

	for(int y=0; y<diffX.rows; y++){
		for(int x=0; x<diffX.cols; x++){
			float dx = diffX.at<float>(y,x);
			float dy = diffY.at<float>(y,x);
			mag.at<float>(y, x) = sqrt(dx*dx + dy*dy);
			theta.at<float>(y, x) = 2.0*CV_PI * (cvFastArctan(dy, dx)/360.0);
			if (theta.at<float>(y, x)>mFullRadian) {
				theta.at<float>(y, x) -= mFullRadian;
			}
		}
	}

	vector<Mat> hist(mTotalBinNum);
	for (int i=0; i<mTotalBinNum ; i++) {
		hist[i] = cv::Mat::zeros(numBlockRows, numBlockCols, CV_32FC1);
	}
	for(int i=0; i<numBlockRows; i++){
		for(int j=0; j<numBlockCols; j++){
			for(int m=i*mBlockStride; m<min<int>(i*mBlockStride+mCellSize, diffX.rows); m++){
				for(int n=j*mBlockStride; n<min<int>(j*mBlockStride+mCellSize, diffX.cols); n++){
					if (mUseZeroMagnitudeBin && mag.at<float>(m, n)<thresholdZeroBin) {
						hist[mRadBinNum].at<float>(i, j) += 1.0;
					} else {
						// simple vote
						/*
						int h = floor(.5 + binPerRadian*(theta.at<float>(m, n)));
						if (h < 0) { h += mRadBinNum; }
						if (h >= mRadBinNum) { h -= mRadBinNum; }

						hist[h].at<float>(i, j) += mag.at<float>(m, n);
						*/
						// weight vote
						float h = binPerRadian*(theta.at<float>(m, n));
						int h1 = floor(h);
						if (h1 < 0) { h1 += mRadBinNum; }
						if (h1 >= mRadBinNum) { h1 -= mRadBinNum; }

						int h2 = h1-1;
						if (h2 < 0) { h2 += mRadBinNum; }
						if (h2 >= mRadBinNum) { h2 -= mRadBinNum; }

						int h3 = h+1;
						if (h3 < 0) { h3 += mRadBinNum; }
						if (h3 >= mRadBinNum) { h3 -= mRadBinNum; }

						float dist1 = abs(h1*radianPerBin+radianPerBin*0.5-theta.at<float>(m, n));
						float dist2 = abs(h2*radianPerBin+radianPerBin*0.5-theta.at<float>(m, n));
						float dist3 = abs(h3*radianPerBin+radianPerBin*0.5-theta.at<float>(m, n));

						float weight1 = max<float>(radianPerBin - dist1, 0)/radianPerBin;
						float weight2 = max<float>(radianPerBin - dist2, 0)/radianPerBin;
						float weight3 = max<float>(radianPerBin - dist3, 0)/radianPerBin;

						int minH1 = h1; 
						int minH2 = h2;
						float maxW1 = weight1; 
						float maxW2 = weight2;
						if (dist1>dist2 && dist1>dist3) {
							minH1 = h2;
							minH2 = h3;
							maxW1 = weight2; 
							maxW2 = weight3;
						} else if (dist2>dist1 && dist2>dist3){
							minH1 = h1;
							minH2 = h3;
							maxW1 = weight1; 
							maxW2 = weight3;
						}

						if (maxW1!=0 && maxW2!=0) {
							//cout << "theta : " << theta.at<float>(m, n)*(180/CV_PI) << endl;
							//cout << "h1 : " << (minH1*radianPerBin+radianPerBin*0.5)*(180/CV_PI) << ", weight1 : " << maxW1 << endl;
							//cout << "h2 : " << (minH2*radianPerBin+radianPerBin*0.5)*(180/CV_PI) << ", weight2 : " << maxW2 << endl;
							hist[minH1].at<float>(i, j) += maxW1 * mag.at<float>(m, n);
							hist[minH2].at<float>(i, j) += maxW2 * mag.at<float>(m, n);
						} else {
							int minH = minH1;
							if (maxW1==0) {
								minH = minH2;
							}
							float maxW = 1.0;
							//cout << "theta : " << theta.at<float>(m, n)*(180/CV_PI) << endl;
							//cout << "h : " << (minH*radianPerBin+radianPerBin*0.5)*(180/CV_PI) << ", weight : " << maxW << endl;
							hist[minH].at<float>(i, j) += maxW * mag.at<float>(m, n);
						}
					}
				}
			}
		}
	}

	desc.resize(numBlockCols*numBlockRows*numCellPerBlock*numCellPerBlock*mTotalBinNum);

	int countCell = 0;
	for (int blockX=0; blockX<numBlockCols; blockX++) {
		for (int blockY=0; blockY<numBlockRows; blockY++) {
			double norm = 0.0;
			for (int cellX=blockX; cellX<min<int>(blockX+numCellPerBlock, numBlockCols); cellX++) {
				for (int cellY=blockY; cellY<min<int>(blockY+numCellPerBlock, numBlockRows); cellY++) {
					for (int i=0; i<mTotalBinNum; i++) {
						norm += hist[i].at<float>(cellY, cellX) * hist[i].at<float>(cellY, cellX);
					}
				}
			}
			norm = sqrt(norm);

			for (int cellX=blockX; cellX<min<int>(blockX+numCellPerBlock, numBlockCols); cellX++) {
				for (int cellY=blockY; cellY<min<int>(blockY+numCellPerBlock, numBlockRows); cellY++) {
					for (int i=0; i<mTotalBinNum; i++) {
						desc[countCell*mTotalBinNum+i] = hist[i].at<float>(cellY, cellX) / norm;
					}
					countCell++;
				}
			}
		}
	}
}

//
// HOG : Output format is same to OpenCV
//
void HogHofMbh::calcHogDesc(const Mat& image, vector<float>& hogDesc)
{
	CV_Assert(image.type()==CV_8UC3 || image.type()==CV_8UC1);

	Mat gray;
	if (image.type()==CV_8UC3) {
		cvtColor(image, gray, CV_BGR2GRAY);	
	} else {
		gray = image;
	}

	Mat sobelX, sobelY;
	Sobel(gray, sobelX, CV_32FC1, 1, 0, 3);
	Sobel(gray, sobelY, CV_32FC1, 0, 1, 3);

	calcDesc(sobelX, sobelY, hogDesc);
}

void HogHofMbh::calcHofDesc(const Mat& flow, vector<float>& hofDesc)
{
	CV_Assert(flow.type()==CV_32FC2);

	Mat flowXY[2];
	split(flow, flowXY);

	calcDesc(flowXY[0], flowXY[1], hofDesc);
}

void HogHofMbh::calcMbhDesc(const Mat& flow, vector<float>& mbhxDesc, vector<float>& mbhyDesc)
{
	CV_Assert(flow.type()==CV_32FC2);

	Mat flowXY[2];
	split(flow, flowXY);

	Mat sobelXX, sobelXY, sobelYX, sobelYY;
	Sobel(flowXY[0], sobelXX, CV_32FC1, 1, 0, 3);
	Sobel(flowXY[0], sobelXY, CV_32FC1, 0, 1, 3);
	Sobel(flowXY[1], sobelYX, CV_32FC1, 1, 0, 3);
	Sobel(flowXY[1], sobelYY, CV_32FC1, 0, 1, 3);

	calcDesc(sobelXX, sobelXY, mbhxDesc);
	calcDesc(sobelYX, sobelYY, mbhyDesc);
}

void HogHofMbh::calcIntegralImage(const Mat& diffX, const Mat& diffY, vector<Mat>& intImage, const Mat& prob)
{
	CV_Assert(diffX.type()==diffY.type() && diffX.size()==diffY.size());

	int numCellPerBlock = ceil((float)mBlockSize/(float)mCellSize);
	float binPerRadian = (float)mRadBinNum/mFullRadian;
	float radianPerBin = (float)mFullRadian/mRadBinNum;
	//cout << "binPerRadian : " << binPerRadian << endl;
	//cout << "radianPerBin : " << radianPerBin << endl;

	intImage.clear();
	for (int d=0; d<mTotalBinNum; d++) {
		Mat iimage = Mat::zeros(diffX.rows, diffX.cols, CV_32F);
		intImage.push_back(iimage);
	}

	Mat mag(diffX.rows, diffX.cols, CV_32F);
	Mat theta(diffX.rows, diffX.cols, CV_32F);

	#pragma omp parallel for
	for(int y=0; y<diffX.rows; y++){
		for(int x=0; x<diffX.cols; x++){
			float dx = diffX.at<float>(y,x);
			if (cvIsNaN(dx) || cvIsInf(dx)) {
				dx = 0.0;
			}
			float dy = diffY.at<float>(y,x);
			if (cvIsNaN(dy) || cvIsInf(dy)) {
				dy = 0.0;
			}
			float fhypot = hypot(dx, dy);
			if (cvIsNaN(fhypot) || cvIsInf(fhypot)) {
				fhypot = 0.0;
			}
			mag.at<float>(y, x) = fhypot;
			if (!prob.empty()) {
				float weight = prob.at<float>(y,x);
				//weight = weight * weight; // squared weight;
				if (cvIsNaN(weight) || cvIsInf(weight)) {
					weight = 0.0;
				}
				mag.at<float>(y, x) *= weight;
			}
			float fatan = fastAtan2(dy, dx);
			if (cvIsNaN(fatan) || cvIsInf(fatan)) {
				fatan = 0.0;
			}
			theta.at<float>(y, x) = 2.0*CV_PI * (fatan/360.0);
			if (theta.at<float>(y, x)>mFullRadian) {
				theta.at<float>(y, x) -= mFullRadian;
			}
		}
	}

	vector<Mat> hist(mTotalBinNum);
	for(int b=0; b<mTotalBinNum; b++) {
		hist[b] = Mat::zeros(diffX.rows, diffX.cols, CV_32F);
	}
	for(int y=0; y<diffX.rows; y++){
		for(int x=0; x<diffX.cols; x++){
			if (mUseZeroMagnitudeBin && mag.at<float>(y, x)<thresholdZeroBin) {
				hist[mRadBinNum].at<float>(y,x) += 1.0;
			} else {
				// simple vote
				/*
				int h = floor(.5 + binPerRadian*(theta.at<float>(y, x)));
				if (h < 0) { h += mRadBinNum; }
				if (h >= mRadBinNum) { h -= mRadBinNum; }

				hist[h].at<float>(y,x) += mag.at<float>(y, x);
				*/
				// weight vote
				float h = binPerRadian*(theta.at<float>(y, x));
				int h1 = floor(h);
				if (h1 < 0) { h1 += mRadBinNum; }
				if (h1 >= mRadBinNum) { h1 -= mRadBinNum; }

				int h2 = h1-1;
				if (h2 < 0) { h2 += mRadBinNum; }
				if (h2 >= mRadBinNum) { h2 -= mRadBinNum; }

				int h3 = h+1;
				if (h3 < 0) { h3 += mRadBinNum; }
				if (h3 >= mRadBinNum) { h3 -= mRadBinNum; }

				float dist1 = abs(h1*radianPerBin+radianPerBin*0.5-theta.at<float>(y, x));
				float dist2 = abs(h2*radianPerBin+radianPerBin*0.5-theta.at<float>(y, x));
				float dist3 = abs(h3*radianPerBin+radianPerBin*0.5-theta.at<float>(y, x));

				float weight1 = max<float>(radianPerBin - dist1, 0)/radianPerBin;
				float weight2 = max<float>(radianPerBin - dist2, 0)/radianPerBin;
				float weight3 = max<float>(radianPerBin - dist3, 0)/radianPerBin;

				int minH1 = h1; 
				int minH2 = h2;
				float maxW1 = weight1; 
				float maxW2 = weight2;
				if (dist1>dist2 && dist1>dist3) {
					minH1 = h2;
					minH2 = h3;
					maxW1 = weight2; 
					maxW2 = weight3;
				} else if (dist2>dist1 && dist2>dist3){
					minH1 = h1;
					minH2 = h3;
					maxW1 = weight1; 
					maxW2 = weight3;
				}

				if (maxW1!=0 && maxW2!=0) {
					//cout << "theta : " << theta.at<float>(m, n)*(180/CV_PI) << endl;
					//cout << "h1 : " << (minH1*radianPerBin+radianPerBin*0.5)*(180/CV_PI) << ", weight1 : " << maxW1 << endl;
					//cout << "h2 : " << (minH2*radianPerBin+radianPerBin*0.5)*(180/CV_PI) << ", weight2 : " << maxW2 << endl;
					hist[minH1].at<float>(y,x) += maxW1 * mag.at<float>(y, x);
					hist[minH2].at<float>(y,x) += maxW2 * mag.at<float>(y, x);
				} else {
					int minH = minH1;
					if (maxW1==0) {
						minH = minH2;
					}
					float maxW = 1.0;
					//cout << "theta : " << theta.at<float>(m, n)*(180/CV_PI) << endl;
					//cout << "h : " << (minH*radianPerBin+radianPerBin*0.5)*(180/CV_PI) << ", weight : " << maxW << endl;
					hist[minH].at<float>(y,x) += maxW * mag.at<float>(y, x);
				}
			}

			for(int b=0; b<mTotalBinNum; b++) {
				intImage[b].at<float>(y,x) = hist[b].at<float>(y,x);
			}
		}
	}
	for(int y=1; y<diffX.rows; y++){
		for(int x=0; x<diffX.cols; x++){
			for(int b=0; b<mTotalBinNum; b++) {
				intImage[b].at<float>(y,x) += intImage[b].at<float>(y-1,x);
			}
		}
	}
	for(int y=0; y<diffX.rows; y++){
		for(int x=1; x<diffX.cols; x++){
			for(int b=0; b<mTotalBinNum; b++) {
				intImage[b].at<float>(y,x) += intImage[b].at<float>(y,x-1);
			}
		}
	}
}

void HogHofMbh::calcHogIImage(const Mat& image, vector<Mat>& hogIImage)
{
	CV_Assert(image.type()==CV_8UC3 || image.type()==CV_8UC1);

	Mat gray;
	if (image.type()==CV_8UC3) {
		cvtColor(image, gray, CV_BGR2GRAY);	
	} else {
		gray = image;
	}

	Mat sobelX, sobelY;
	Sobel(gray, sobelX, CV_32FC1, 1, 0, 3);
	Sobel(gray, sobelY, CV_32FC1, 0, 1, 3);

	calcIntegralImage(sobelX, sobelY, hogIImage);
}

void HogHofMbh::calcHogIImage(const Mat& image, const Mat& prob, vector<Mat>& hogIImage)
{
	CV_Assert(image.type()==CV_8UC3 || image.type()==CV_8UC1);

	Mat gray;
	if (image.type()==CV_8UC3) {
		cvtColor(image, gray, CV_BGR2GRAY);	
	} else {
		gray = image;
	}

	Mat sobelX, sobelY;
	Sobel(gray, sobelX, CV_32FC1, 1, 0, 3);
	Sobel(gray, sobelY, CV_32FC1, 0, 1, 3);

	calcIntegralImage(sobelX, sobelY, hogIImage, prob);
}

void HogHofMbh::calcHofIImage(const Mat& flow, vector<Mat>& hofIImage)
{
	CV_Assert(flow.type()==CV_32FC2);

	Mat flowXY[2];
	split(flow, flowXY);

	calcIntegralImage(flowXY[0], flowXY[1], hofIImage);
}

void HogHofMbh::calcMbhIImage(const Mat& flow, vector<Mat>& mbhxIImage, vector<Mat>& mbhyIImage)
{
	CV_Assert(flow.type()==CV_32FC2);

	Mat flowXY[2];
	split(flow, flowXY);

	Mat sobelXX, sobelXY, sobelYX, sobelYY;
	Sobel(flowXY[0], sobelXX, CV_32FC1, 1, 0, 3);
	Sobel(flowXY[0], sobelXY, CV_32FC1, 0, 1, 3);
	Sobel(flowXY[1], sobelYX, CV_32FC1, 1, 0, 3);
	Sobel(flowXY[1], sobelYY, CV_32FC1, 0, 1, 3);

	calcIntegralImage(sobelXX, sobelXY, mbhxIImage);
	calcIntegralImage(sobelYX, sobelYY, mbhyIImage);
}

int HogHofMbh::getDescByIntegralImageDim(const Size& size)
{
	int numBlockCols = floor((float)(size.width-mBlockSize)/(float)mBlockStride)+1;
	int numBlockRows = floor((float)(size.height-mBlockSize)/(float)mBlockStride)+1;
	int numCellPerBlock = ceil((float)mBlockSize/(float)mCellSize);
	float binPerRadian = (float)mRadBinNum/mFullRadian;
	float radianPerBin = (float)mFullRadian/mRadBinNum;

	return numBlockCols*numBlockRows*numCellPerBlock*numCellPerBlock*mTotalBinNum;
}

void HogHofMbh::calcDescByIntegralImage(const Rect& rect, const vector<Mat>& intImage, vector<float>& desc)
{
	CV_Assert(intImage.size()==mTotalBinNum && rect.width<=intImage[0].size().width && rect.height<=intImage[0].size().height
			&& rect.width>=mBlockSize && rect.height>=mBlockSize);

	int numBlockCols = floor((float)(rect.width-mBlockSize)/(float)mBlockStride)+1;
	int numBlockRows = floor((float)(rect.height-mBlockSize)/(float)mBlockStride)+1;
	int numCellPerBlock = ceil((float)mBlockSize/(float)mCellSize);
	float binPerRadian = (float)mRadBinNum/mFullRadian;
	float radianPerBin = (float)mFullRadian/mRadBinNum;
	//cout << "numBlockCols : " << numBlockCols << endl;
	//cout << "numBlockRows : " << numBlockRows << endl;
	//cout << "binPerRadian : " << binPerRadian << endl;
	//cout << "radianPerBin : " << radianPerBin << endl;

	//cout << "HogHofMbh Feature Dim : " << numBlockCols*numBlockRows*numCellPerBlock*numCellPerBlock*mTotalBinNum << endl;
	desc.resize(numBlockCols*numBlockRows*numCellPerBlock*numCellPerBlock*mTotalBinNum);
	std::fill(desc.begin(), desc.end(), 0);

	int countCell = 0;
	for (int blockX=0; blockX<numBlockCols; blockX++) {
		for (int blockY=0; blockY<numBlockRows; blockY++) {
			int blockX1 = rect.x + blockX*mBlockStride;
			int blockY1 = rect.y + blockY*mBlockStride;

			int blockStart = countCell*mTotalBinNum;
			for (int cellX=0; cellX<numCellPerBlock; cellX++) {
				for (int cellY=0; cellY<numCellPerBlock; cellY++) {
					int cellX1 = blockX1 + cellX*mCellSize;
					int cellY1 = blockY1 + cellY*mCellSize;
					int cellX2 = min<int>(blockX1+(cellX+1)*mCellSize-1, rect.x+rect.width-1);
					int cellY2 = min<int>(blockY1+(cellY+1)*mCellSize-1, rect.y+rect.height-1);

					for (int i=0; i<mTotalBinNum; i++) {
						desc[countCell*mTotalBinNum+i] = intImage[i].at<float>(cellY2, cellX2) - intImage[i].at<float>(cellY1, cellX2)
														- intImage[i].at<float>(cellY2, cellX1) + intImage[i].at<float>(cellY1, cellX1);
					}
					countCell++;
				}
			}
			int blockEnd = countCell*mTotalBinNum;
			double norm = 0.0;
			for (int i=blockStart; i<blockEnd; i++) {
				norm += pow(desc[i], 2);
			}
			norm = sqrt(norm);

			if (norm>0.0) {
				for (int i=blockStart; i<blockEnd; i++) {
					desc[i] /= norm;
				}
			}
			for (int i=blockStart; i<blockEnd; i++) {
				if (cvIsNaN(desc[i]) || cvIsInf(desc[i])) {
					desc[i] = 0.0;
				}
			}
		}
	}
}