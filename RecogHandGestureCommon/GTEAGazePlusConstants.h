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

#ifndef __RECOG_HAND_GESTURE_GTEA_GAZE_PLUS_CONSTANTS__
#define __RECOG_HAND_GESTURE_GTEA_GAZE_PLUS_CONSTANTS__

#include <vector>
#include <map>

using namespace std;

// scale ratio when parsing movie by annotation
static const float GTEAGAZE_PARSE_SCALE_MOVIE = 0.5;

// resize captured video to fixed size
static const int GTEAGAZE_MOVIE_FRAME_WIDTH = 480;
static const int GTEAGAZE_MOVIE_FRAME_HEIGHT = 360;

// customized setting for dense trajectory
static const int GTEAGAZE_TIME_INTERVAL_DETECT_KEYPOINT = 1;
static const int GTEAGAZE_DENSE_KEYPOINT_STEP = 10; // original paper setting 5

// parameters for leave one out by subject
const char* GTEAGAZE_SUBJECTS_ARRAY[] = {
	"Alireza",
	"Carlos",
	"Rahul",
	"Shaghayegh",
	"Yin"
};
vector<string> GTEAGAZE_SUBJECTS(begin(GTEAGAZE_SUBJECTS_ARRAY), end(GTEAGAZE_SUBJECTS_ARRAY));

const char* GTEAGAZE_TARGET_FOODS_ARRAY[] = {
	"American",
	"Pizza"
};
/*
const char* GTEAGAZE_TARGET_FOODS_ARRAY[] = {
	"American",
	"Pizza",
	"Burger",
	"Greek",
	"Pasta",
	"Turkey"
};
*/
vector<string> GTEAGAZE_TARGET_FOODS(begin(GTEAGAZE_TARGET_FOODS_ARRAY), end(GTEAGAZE_TARGET_FOODS_ARRAY));

// first version of verb array
/*
const char* GTEAGAZE_TARGET_VERBS_ARRAY[] = {
	"none",
	"crack",
	"cut",
	"distribute",
	"flip",
	"mix",
	"move around",
	"pour",
	"spread",
	"transfer"
};
*/
// remove difficult verb, crack, flip, transfer
const char* GTEAGAZE_TARGET_VERBS_ARRAY[] = {
	"none",
	"cut",
	"distribute",
	"mix",
	"move around",
	"pour",
	"spread"
};
/*
// remove difficult verb, crack, flip, transfer
// add verbs wash dry, squeeze, compress, peel
// use all types of foods for this verb set
const char* GTEAGAZE_TARGET_VERBS_ARRAY[] = {
	"none",
	"cut",
	"distribute",
	"mix",
	"move around",
	"pour",
	"spread",
	"wash dry",
	"squeeze",
	"compress",
	"peel"
};
*/
vector<string> GTEAGAZE_TARGET_VERBS(begin(GTEAGAZE_TARGET_VERBS_ARRAY), end(GTEAGAZE_TARGET_VERBS_ARRAY));

// parameters for sliding time window
static const int GTEAGAZE_FPS = 24;
static const int GTEAGAZE_ACTION_VERB_RECOG_GESTURE_FRAME_LENGTH = GTEAGAZE_FPS*2; // We assume each training movie has at least one gesture in this time length
static const int GTEAGAZE_ACTION_VERB_OBJECT_RECOG_GESTURE_FRAME_LENGTH = GTEAGAZE_FPS; // We assume each training movie has at least one gesture in this time length
static const int GTEAGAZE_RECOG_GESTURE_FRAME_STEP = GTEAGAZE_FPS/2; // Extract train/test gesturess by this time step
static const int GTEAGAZE_MAX_FRAME_NONE_ACTION = GTEAGAZE_FPS*5; // For None action class, do not process all frames when creating training data
static const int GTEAGAZE_ACTION_VERB_SKIP_NUM_EVERY_NONE = 4; // For None action verb class, skip several files after processing every file when creating training data
static const int GTEAGAZE_ACTION_VERB_OBJECT_SKIP_NUM_EVERY_NONE = 4; // For None action verb object class, skip several files after processing every file when creating training data

// parameters for Hand Detection
pair<string, string> GTEAGAZE_HAND_DETECT_MODEL_LIST_FILES_TABLE[] = {
	pair<string, string>("Alireza", "..\\..\\data\\model\\GTEAGazePlus\\Alireza_Snack\\list.txt"),
	pair<string, string>("Carlos", "..\\..\\data\\model\\GTEAGazePlus\\Carlos_Snack\\list.txt"),
	pair<string, string>("Rahul", "..\\..\\data\\model\\GTEAGazePlus\\Rahul_Snack\\list.txt"),
	pair<string, string>("Shaghayegh", "..\\..\\data\\model\\GTEAGazePlus\\Shaghayegh_Snack\\list.txt"),
	pair<string, string>("Yin", "..\\..\\data\\model\\GTEAGazePlus\\Yin_Snack\\list.txt")
};
map<string, string> GTEAGAZE_HAND_DETECT_MODEL_LIST_FILES(GTEAGAZE_HAND_DETECT_MODEL_LIST_FILES_TABLE, GTEAGAZE_HAND_DETECT_MODEL_LIST_FILES_TABLE+5);

pair<string, string> GTEAGAZE_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE[] = {
	pair<string, string>("Alireza", "..\\..\\data\\globfeat\\GTEAGazePlus\\Alireza_Snack\\list.txt"),
	pair<string, string>("Carlos", "..\\..\\data\\globfeat\\GTEAGazePlus\\Carlos_Snack\\list.txt"),
	pair<string, string>("Rahul", "..\\..\\data\\globfeat\\GTEAGazePlus\\Rahul_Snack\\list.txt"),
	pair<string, string>("Shaghayegh", "..\\..\\data\\globfeat\\GTEAGazePlus\\Shaghayegh_Snack\\list.txt"),
	pair<string, string>("Yin", "..\\..\\data\\globfeat\\GTEAGazePlus\\Yin_Snack\\list.txt")
};
map<string, string> GTEAGAZE_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES(GTEAGAZE_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE, GTEAGAZE_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE+5);

#endif __RECOG_HAND_GESTURE_GTEA_GAZE_PLUS_CONSTANTS__