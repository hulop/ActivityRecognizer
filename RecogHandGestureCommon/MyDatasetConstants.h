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

#ifndef __RECOG_HAND_GESTURE_MY_DATASET_CONSTANTS__
#define __RECOG_HAND_GESTURE_MY_DATASET_CONSTANTS__

// resize captured video to fixed size
static const int MYDATASET_MOVIE_FRAME_WIDTH = 640;
static const int MYDATASET_MOVIE_FRAME_HEIGHT = 360;

// customized setting for dense trajectory
static const int MYDATASET_TIME_INTERVAL_DETECT_KEYPOINT = 1;
static const int MYDATASET_DENSE_KEYPOINT_STEP = 10; // original paper setting 5

// parameters for sliding time window
static const int MYDATASET_FPS = 25;
static const int MYDATASET_RECOG_GESTURE_FRAME_LENGTH = MYDATASET_FPS; // We assume each training movie has at least one gesture in this time length
static const int MYDATASET_RECOG_GESTURE_FRAME_STEP = MYDATASET_FPS/2; // Extract train/test gesturess by this time step

// parameters for Hand Detection
static const string MYDATASET_HAND_DETECT_MODEL_LIST_FILE = "..\\..\\data\\model\\HandDetect\\clip0027\\list.txt";
static const string MYDATASET_HAND_DETECT_GLOBAL_FEATURE_LIST_FILE = "..\\..\\data\\globfeat\\HandDetect\\clip0027\\list.txt";

#endif __RECOG_HAND_GESTURE_MY_DATASET_CONSTANTS__