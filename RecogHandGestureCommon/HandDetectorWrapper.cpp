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

#include "HandDetectorWrapper.h"
#include "RecogHandGestureConstants.h"
#include "HogHofMbh.h"

void HandDetectorWrapper::trainModels(string imgListFile, string mskListFile, string model_prefix, string modelListFile,
	string globfeat_prefix, string globFeatListFile, string feature_set, int max_models, int width)
{
	vector<string> imgFiles;
	{
		ifstream fs;
		fs.open(imgListFile);
		string file;
		while (fs >> file) {
			if (file.size()>0) {
				imgFiles.push_back(file);
			}
		}
	}
	vector<string> mskFiles;
	{
		ifstream fs;
		fs.open(mskListFile);
		string file;
		while (fs >> file) {
			if (file.size()>0) {
				mskFiles.push_back(file);
			}
		}
	}
	CV_Assert(imgFiles.size() == mskFiles.size());

	cout << "HandDetector::trainModels()" << endl;

	////////////// TODO Hand Detector 3
	handDetector._img_width = (float)width;

	LcFeatureExtractor	_extractor;
	LcRandomTreesR		_classifier;

	handDetector._feature_set = feature_set;
	_extractor.set_extractor(feature_set);
	////////////// TODO Hand Detector 3

	stringstream ss;
	int f = -1;
	int k = 0;
	ofstream modelOfs(modelListFile, ios_base::out);
	ofstream globalFeatOfs(globFeatListFile, ios_base::out);

	while (k<max_models) {
		f++;

		// load image and mask
		Mat mask_img = imread(mskFiles[f], 0);
		if (!mask_img.data || countNonZero(mask_img) == 0) {
			cout << "Skipping mask : " << mskFiles[f] << endl;
			continue;
		}
		else {
			cout << "Loading mask : " << mskFiles[f] << endl;
		}

		Mat color_img = imread(imgFiles[f], 1);
		if (!color_img.data) {
			cout << "Skipping masked image : " << imgFiles[f] << endl;
			continue;
		}
		else {
			cout << "Loading masked image : " << imgFiles[f] << endl;
		}

		////////////// TODO Hand Detector 4
		handDetector._img_height = color_img.rows * (handDetector._img_width / color_img.cols);
		handDetector._img_size = Size(handDetector._img_width, handDetector._img_height);

		resize(color_img, color_img, handDetector._img_size);
		resize(mask_img, mask_img, handDetector._img_size);
		////////////// TODO Hand Detector 4

		imshow("src", color_img);
		imshow("mask", mask_img);

		Mat dsp;
		cvtColor(mask_img, dsp, CV_GRAY2BGR);
		addWeighted(dsp, 0.5, color_img, 0.5, 0, dsp);
		imshow("blend", dsp);
		waitKey(1);

		// extract/save histgram
		Mat globfeat;
		computeColorHist_HSV(color_img, globfeat);

		ss.str("");
		ss << globfeat_prefix << "\\" << "hsv_histogram_" << k << ".xml";
		cout << "Writing global feature: " << ss.str() << endl;

		FileStorage fs;
		fs.open(ss.str(), FileStorage::WRITE);
		fs << "globfeat" << globfeat;
		fs.release();

		globalFeatOfs << ss.str() << endl;

		// train/save classfier
		Mat desc;
		Mat lab;
		vector<KeyPoint> kp;

		mask_img.convertTo(mask_img, CV_8UC1);
		////////////// TODO Hand Detector 5
		_extractor.work(color_img, desc, mask_img, lab, 1, &kp);
		_classifier.train(desc, lab);

		ss.str("");
		ss << model_prefix << "\\" << "model_" + feature_set + "_" << k << "_rdtr.xml";
		_classifier._random_tree.save(ss.str().c_str());
		////////////// TODO Hand Detector 5

		modelOfs << ss.str() << endl;

		k++;

		cout << "Finish training " << k << " th model " << endl;
	}
}

void HandDetectorWrapper::testInitialize(string modelListFile, string globFeatListFile, string feature_set, int knn, int width)
{
	stringstream ss;

	// set feature extractor
	cout << "Set feature extractor" << endl;
	////////////// TODO Hand Detector 6
	handDetector._img_width = (float)width;
	handDetector._feature_set = feature_set;
	handDetector._extractor.set_extractor(handDetector._feature_set);
	////////////// TODO Hand Detector 6

	// load classifiers
	{
		vector<string> filenames;
		ifstream fs;
		fs.open(modelListFile);
		string file;
		while (fs >> file) {
			if (file.size()>0) {
				filenames.push_back(file);
			}
		}

		int num_models = (int)filenames.size();

		cout << "Load classifiers" << endl;
		////////////// TODO Hand Detector 7
		handDetector._classifier = vector<LcRandomTreesR>(num_models);

		for (int i = 0; i<num_models; i++) {
			handDetector._classifier[i].load_full(filenames[i]);
		}
		////////////// TODO Hand Detector 7
	}

	// load histgram
	{
		vector<string> filenames;
		ifstream fs;
		fs.open(globFeatListFile);
		string file;
		while (fs >> file) {
			if (file.size()>0) {
				filenames.push_back(file);
			}
		}

		int num_models = (int)filenames.size();

		cout << "Load global features" << endl;
		for (int i = 0; i<num_models; i++) {
			Mat globalfeat;

			cout << filenames[i] << endl;
			FileStorage fs;
			fs.open(filenames[i], FileStorage::READ);
			fs["globfeat"] >> globalfeat;
			fs.release();

			////////////// TODO Hand Detector 8
			handDetector._hist_all.push_back(globalfeat);
			////////////// TODO Hand Detector 8
		}
	}

	// build KNN classifiers
	cout << "Building FLANN search structure...";
	////////////// TODO Hand Detector 9
	CV_Assert(handDetector._hist_all.rows == (int)handDetector._classifier.size()); // number of classifiers and number of global features should be same
	handDetector._indexParams = *new flann::KMeansIndexParams;
	handDetector._searchtree = *new flann::Index(handDetector._hist_all, handDetector._indexParams);
	handDetector._knn = knn;	//number of nearest neighbors 
	handDetector._indices = vector<int>(handDetector._knn);
	handDetector._dists = vector<float>(handDetector._knn);
	////////////// TODO Hand Detector 9

	cout << "Finish hand detector initialization." << endl;
}

void HandDetectorWrapper::test(Mat &img, int num_models, int step_size)
{
	////////////// TODO Hand Detector 10
	handDetector.test(img, num_models, step_size);
	////////////// TODO Hand Detector 10
}

Mat HandDetectorWrapper::getResponseImage()
{
	////////////// TODO Hand Detector 11
	return handDetector._response_img.clone();
	////////////// TODO Hand Detector 11
}

Mat HandDetectorWrapper::postprocess(Mat &img, float contourThres)
{
	vector<Point2f> pt;
	return postprocess(img, pt, contourThres);
}

Mat HandDetectorWrapper::postprocess(Mat &img, vector<Point2f> &pt, float contourThres)
{
	Mat tmp;
	GaussianBlur(img, tmp, cv::Size(11, 11), 0, 0, BORDER_REFLECT);
	////////////// TODO Hand Detector 12
	colormap(tmp, handDetector._blu, 1);
	////////////// TODO Hand Detector 12

	tmp = tmp > contourThres;

	vector<vector<cv::Point> > co;
	vector<Vec4i> hi;

	findContours(tmp, co, hi, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	tmp *= 0;

	Moments m;
	for (int i = 0; i<(int)co.size(); i++) {
		if (contourArea(Mat(co[i])) < (tmp.rows*tmp.cols*0.01)) {
			continue;
		}
		drawContours(tmp, co, i, CV_RGB(255, 255, 255), CV_FILLED, CV_AA);
		m = moments(Mat(co[i]));
		pt.push_back(Point2f(m.m10 / m.m00, m.m01 / m.m00));
	}

	return tmp;
}

int HandDetectorWrapper::getDenseHandHOGDim(Size block_size, Size strid_size, Size cell_size, int ori_bins)
{
	// create integral image for HOG
	HogHofMbh hog(DENSE_HAND_HOG_GRID_2D, DENSE_HAND_HOG_GRID_2D, DENSE_HAND_HOG_GRID_2D, DENSE_HAND_HOG_BIN_NUM,
		DENSE_HAND_HOG_USE_ZERO_MAGNITUDE_BIN, DENSE_HAND_HOG_FULL_RADIAN);

	int shiftBlockSize = (DENSE_HAND_HOG_NEIGHBOR_SIZE + 1) / 2;
	return hog.getDescByIntegralImageDim(Size(shiftBlockSize * 2 + 1, shiftBlockSize * 2 + 1));
}

Mat HandDetectorWrapper::computeDenseHandHOG(Mat &image, Mat &prob, Size block_size, Size strid_size, Size cell_size, int ori_bins, 
	Mat &mask, Mat& handHogDescs)
{
	// check input image type
	Mat gray;
	if (image.type() == CV_8UC3) {
		cvtColor(image, gray, CV_BGR2GRAY);
	}
	else {
		gray = image;
	}

	// calculate mask by bounding boxes
	/*
	Mat mask = Mat::zeros(gray.size(), CV_8UC1);
	for (int r=0; r<bbox.size(); r++) {
	rectangle(mask, bbox[r], Scalar::all(255), -1, CV_AA);
	}
	*/

	// check input probability range
	Mat handProb = normalizeProb(prob);

	// calculate pyramid images
	vector<cv::Mat> grayPyramid;
	vector<cv::Mat> maskPyramid;
	float scalePyramid = dtPyramidScale;
	grayPyramid.push_back(gray);
	maskPyramid.push_back(mask);
	for (int p = 1; p<dtPyramidLevels; p++) {
		cv::Mat scaleGray;
		cv::resize(gray, scaleGray, cv::Size(), scalePyramid, scalePyramid);
		grayPyramid.push_back(scaleGray);

		if (!mask.empty()) {
			cv::Mat scaleMask;
			cv::resize(mask, scaleMask, cv::Size(), scalePyramid, scalePyramid);
			maskPyramid.push_back(scaleMask);
		}
		else {
			maskPyramid.push_back(mask);
		}

		scalePyramid *= scalePyramid;
	}

	Mat vizDenseHogLevel0;
	vector<Mat> handHogDescsVec(grayPyramid.size());
#pragma omp parallel for
	for (int p = 0; p<grayPyramid.size(); p++) {
		// detect dense keypoints
		DenseFeatureDetector detector(1.f, 1, 0.1f, DENSE_HAND_HOG_KEYPOINT_STEP, (DENSE_HAND_HOG_NEIGHBOR_SIZE + 1) / 2);
		vector<KeyPoint> denseKeypoints;
		detector.detect(grayPyramid[p], denseKeypoints);

		// create integral image for HOG
		HogHofMbh hog(DENSE_HAND_HOG_GRID_2D, DENSE_HAND_HOG_GRID_2D, DENSE_HAND_HOG_GRID_2D, DENSE_HAND_HOG_BIN_NUM,
			DENSE_HAND_HOG_USE_ZERO_MAGNITUDE_BIN, DENSE_HAND_HOG_FULL_RADIAN);
		vector<Mat> hogIImage;
		if (DENSE_HAND_HOG_WEIGHT_PROBABILITY) {
			Mat probPyramid;
			cv::resize(handProb, probPyramid, grayPyramid[p].size());
			hog.calcHogIImage(grayPyramid[p], probPyramid, hogIImage);
		}
		else {
			hog.calcHogIImage(grayPyramid[p], hogIImage);
		}

		// compute dense hog from bounding boxes region
		for (int k = 0; k<denseKeypoints.size(); k++) {
			Point2f point = denseKeypoints[k].pt;
			if ((int)maskPyramid[p].at<uchar>(point.y, point.x) != 0) {
				int r = (int)floor(.5 + point.y);
				int c = (int)floor(.5 + point.x);

				int shiftBlockSize = (DENSE_HAND_HOG_NEIGHBOR_SIZE + 1) / 2;
				int x1 = c - shiftBlockSize;
				int x2 = c + shiftBlockSize;
				int y1 = r - shiftBlockSize;
				int y2 = r + shiftBlockSize;

				CV_Assert(x1 >= 0 && y1 >= 0 && x2<grayPyramid[p].size().width && y2<grayPyramid[p].size().height);

				vector<float> hogVec;
				hog.calcDescByIntegralImage(Rect(x1, y1, x2 - x1, y2 - y1), hogIImage, hogVec);

				Mat hogDesc = Mat(hogVec);
				hogDesc = hogDesc.t();
				hogDesc.convertTo(hogDesc, CV_32FC1);
				handHogDescsVec[p].push_back(hogDesc);
			}
		}

		// debug visualize
		stringstream ss;
		ss << "Hand Hog : Level " << (p + 1);
		Mat visDenseHog = visual_dense_hog(hogIImage, maskPyramid[p], 9, true, ss.str());
		//Mat visDenseHog = visual_dense_hog2(hogIImage, maskPyramid[p], 9, true, ss.str());

		if (p == 0) {
			vizDenseHogLevel0 = visDenseHog;
		}
	}
	for (int p = 0; p<grayPyramid.size(); p++) {
		handHogDescs.push_back(handHogDescsVec[p]);
	}

	return vizDenseHogLevel0;
}

void HandDetectorWrapper::rasterizeResVec(Mat &img, Mat&res, vector<KeyPoint> &keypts, cv::Size s, int bs)
{
	////////////// TODO Hand Detector 13
	handDetector.rasterizeResVec(img, res, keypts, s, bs);
	////////////// TODO Hand Detector 13
}

// normalize positive values to the range [0, 1]
Mat HandDetectorWrapper::normalizeProb(Mat &prob)
{
	CV_Assert(prob.type() == CV_32FC1);

	Mat normProb(prob.size(), CV_32FC1);
	for (int i = 0; i<prob.rows; i++) {
		for (int j = 0; j<prob.cols; j++) {
			normProb.at<float>(i, j) = 1.0 / (1.0 + exp(-prob.at<float>(i, j))) - 0.5;
			//cout << "prob : " << prob.at<float>(i,j) << ", norm prob : " << normProb.at<float>(i,j) << endl;
		}
	}
	return normProb;
}

void HandDetectorWrapper::colormap(Mat &src, Mat &dst, int do_norm)
{
	////////////// TODO Hand Detector 14
	handDetector.colormap(src, dst, do_norm);
	////////////// TODO Hand Detector 14
}

void HandDetectorWrapper::computeColorHist_HSV(Mat &src, Mat &hist)
{
	////////////// TODO Hand Detector 15
	handDetector.computeColorHist_HSV(src, hist);
	////////////// TODO Hand Detector 15
}

Mat HandDetectorWrapper::visual_dense_hog(vector<Mat>& intImage, Mat mask, int binNum, bool fullRadianHogHofMbh, string windowName)
{
	static const float VISUALIZE_MAX_MAGNITUDE = 100.0;

	float radianPerBin;
	if (fullRadianHogHofMbh) {
		radianPerBin = (float)2.0*CV_PI / binNum;
	}
	else {
		radianPerBin = (float)CV_PI / binNum;
	}

	Mat magnitude = Mat::zeros(intImage[0].size(), CV_32F);
	Mat angle = Mat::zeros(intImage[0].size(), CV_32F);
	for (int y = 0; y<intImage[0].rows; y++) {
		for (int x = 0; x<intImage[0].cols; x++) {
			if ((int)mask.at<uchar>(y, x) != 0) {
				int maxBinIndex = -1;
				float maxBinMagnitude = 0.0;
				for (int b = 0; b<intImage.size(); b++) {
					float magnitude;
					if (y == 0 && x == 0) {
						magnitude = intImage[b].at<float>(y, x);
					}
					else if (y == 0) {
						magnitude = intImage[b].at<float>(y, x) - intImage[b].at<float>(y, x - 1);
					}
					else if (x == 0) {
						magnitude = intImage[b].at<float>(y, x) - intImage[b].at<float>(y - 1, x);
					}
					else {
						magnitude = intImage[b].at<float>(y, x) - intImage[b].at<float>(y, x - 1) - intImage[b].at<float>(y - 1, x) + intImage[b].at<float>(y - 1, x - 1);
					}
					if (magnitude>maxBinMagnitude) {
						maxBinIndex = b;
						maxBinMagnitude = magnitude;
					}
				}
				if (maxBinIndex >= 0) {
					angle.at<float>(y, x) = (180.0 / CV_PI)*(radianPerBin*maxBinIndex + radianPerBin*0.5);
					magnitude.at<float>(y, x) = min<float>(maxBinMagnitude, VISUALIZE_MAX_MAGNITUDE);
				}
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

	return hsv;
}

// another visualization
Mat HandDetectorWrapper::visual_dense_hog2(vector<Mat>& intImage, Mat mask, int binNum, bool fullRadianHogHofMbh, string windowName)
{
	CV_Assert(intImage.size()>0);

	float radianPerBin;
	if (fullRadianHogHofMbh) {
		radianPerBin = (float)2.0*CV_PI / binNum;
	}
	else {
		radianPerBin = (float)CV_PI / binNum;
	}

	Size image_size = intImage[0].size();
	Size block_size = HAND_HOG_BLOCK_SIZE;
	Size strid_size = HAND_HOG_STRIDE_SIZE;
	Size cell_size = HAND_HOG_CELL_SIZE;

	int dsp_cell_size = 12;
	int ori_bins = binNum;

	int cell_num_cols = (image_size.width - block_size.width) / strid_size.width;
	int cell_num_rows = (image_size.height - block_size.height) / strid_size.height;

	Mat hdesc = Mat::zeros(cell_num_rows * dsp_cell_size, cell_num_cols * dsp_cell_size, CV_8UC3);

	for (int c = 0; c < cell_num_cols; c++) {
		for (int r = 0; r < cell_num_rows; r++) {
			int cy = r * dsp_cell_size + floor(dsp_cell_size * 0.5 + 0.5); // transpose here
			int cx = c * dsp_cell_size + floor(dsp_cell_size * 0.5 + 0.5);

			for (int a = 0; a < ori_bins; a++) {
				float m = 0.0;
				for (int x = c*strid_size.width; x < c*strid_size.width + block_size.width; x++) {
					for (int y = r*strid_size.height; y < r*strid_size.height + block_size.height; y++) {
						if ((int)mask.at<uchar>(y, x) != 0) {
							float magnitude;
							if (y == 0 && x == 0) {
								magnitude = intImage[a].at<float>(y, x);
							}
							else if (y == 0) {
								magnitude = intImage[a].at<float>(y, x) - intImage[a].at<float>(y, x - 1);
							}
							else if (x == 0) {
								magnitude = intImage[a].at<float>(y, x) - intImage[a].at<float>(y - 1, x);
							}
							else {
								magnitude = intImage[a].at<float>(y, x) - intImage[a].at<float>(y, x - 1) - intImage[a].at<float>(y - 1, x) + intImage[a].at<float>(y - 1, x - 1);
							}
							m += magnitude;
						}
					}
				}
				m /= (float)(block_size.width*block_size.height);

				float p = 0.48;
				float rad = radianPerBin * a;
				float h;
				if (rad == 0) {
					h = dsp_cell_size * p / cos(rad);
				}
				else {
					h = MIN(fabs(dsp_cell_size*p / cos(rad)), fabs(dsp_cell_size*p / sin(rad)));
				}
				int dx = floor(h * cos(CV_PI * 0.5 + rad) + 0.5);
				int dy = floor(h * sin(CV_PI * 0.5 + rad) + 0.5);

				if (fullRadianHogHofMbh) {
					line(hdesc, Point(cx, cy), Point(cx + dx, cy + dy), Scalar::all(m * 1000.), 1, CV_AA);
				}
				else {
					line(hdesc, Point(cx - dx, cy - dy), Point(cx + dx, cy + dy), Scalar::all(m * 1000.), 1, CV_AA);
				}
			}
		}
	}

	imshow(windowName, hdesc);

	return hdesc;
}