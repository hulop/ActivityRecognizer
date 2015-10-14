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

#ifndef __RECOG_HAND_GESTURE_GTEA_CONSTANTS__
#define __RECOG_HAND_GESTURE_GTEA_CONSTANTS__

#include <vector>
#include <map>

using namespace std;

// scale ratio when parsing movie by annotation
static const float GTEA_PARSE_SCALE_MOVIE = 1.0;

// resize captured video to fixed size
static const int GTEA_MOVIE_FRAME_WIDTH = 720;
static const int GTEA_MOVIE_FRAME_HEIGHT = 404;

// customized setting for dense trajectory
static const int GTEA_TIME_INTERVAL_DETECT_KEYPOINT = 0;
static const int GTEA_DENSE_KEYPOINT_STEP = 10; // original paper setting 5

// parameters for leave one out by subject
const char* GTEA_SUBJECTS_ARRAY[] = {
	"S1",
	"S2",
	"S3",
	"S4"
};
vector<string> GTEA_SUBJECTS(begin(GTEA_SUBJECTS_ARRAY), end(GTEA_SUBJECTS_ARRAY));

static const string GTEA_TEST_SUBJECT = "S2";

const char* GTEA_TARGET_FOODS_ARRAY[] = {
	"Cheese",
	"Coffee",
	"CofHoney",
	"Hotdog",
	"Pealate",
	"Peanut",
	"Tea"
};
vector<string> GTEA_TARGET_FOODS(begin(GTEA_TARGET_FOODS_ARRAY), end(GTEA_TARGET_FOODS_ARRAY));

const char* GTEA_TARGET_ACTIONS_ARRAY[] = {
	"none",
	"open",
	"close",
	"fold",
	"pour",
	"put",
	"scoop",
	"shake",
	"spread",
	"stir",
	"take"
};
vector<string> GTEA_TARGET_ACTIONS(begin(GTEA_TARGET_ACTIONS_ARRAY), end(GTEA_TARGET_ACTIONS_ARRAY));

// parameters for sliding time window
static const int GTEA_FPS = 15;
static const int GTEA_RECOG_GESTURE_FRAME_LENGTH = GTEA_FPS; // We assume each training movie has at least one gesture in this time length
static const int GTEA_RECOG_GESTURE_FRAME_STEP = 1; // Extract train/test gesturess by this time step
static const int GTEA_MAX_FRAME_NONE_ACTION = GTEA_FPS*5; // For None action class, do not process all frames when creating training data
static const int GTEA_ACTION_SKIP_NUM_EVERY_NONE = 4; // For None action class, skip several files after processing every file when creating training data

// parameters for Hand Detection
pair<string, string> GTEA_HAND_DETECT_MODEL_LIST_FILES_TABLE[] = {
	pair<string, string>("S1", "..\\..\\data\\model\\GTEA\\S1\\list.txt"),
	pair<string, string>("S2", "..\\..\\data\\model\\GTEA\\S2\\list.txt"),
	pair<string, string>("S3", "..\\..\\data\\model\\GTEA\\S3\\list.txt"),
	pair<string, string>("S4", "..\\..\\data\\model\\GTEA\\S4\\list.txt")
};
map<string, string> GTEA_HAND_DETECT_MODEL_LIST_FILES(GTEA_HAND_DETECT_MODEL_LIST_FILES_TABLE, GTEA_HAND_DETECT_MODEL_LIST_FILES_TABLE+4);

pair<string, string> GTEA_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE[] = {
	pair<string, string>("S1", "..\\..\\data\\globfeat\\GTEA\\S1\\list.txt"),
	pair<string, string>("S2", "..\\..\\data\\globfeat\\GTEA\\S2\\list.txt"),
	pair<string, string>("S3", "..\\..\\data\\globfeat\\GTEA\\S3\\list.txt"),
	pair<string, string>("S4", "..\\..\\data\\globfeat\\GTEA\\S4\\list.txt")
};
map<string, string> GTEA_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES(GTEA_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE, GTEA_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE+4);

#endif __RECOG_HAND_GESTURE_GTEA_CONSTANTS__