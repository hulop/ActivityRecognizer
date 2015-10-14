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

#ifndef __RECOG_HAND_GESTURE_CMUKITCHEN_CONSTANTS__
#define __RECOG_HAND_GESTURE_CMUKITCHEN_CONSTANTS__

#include <vector>
#include <map>

using namespace std;

// scale ratio when parsing movie by annotation
static const float CMUKITCHEN_PARSE_SCALE_MOVIE = 0.5;

// resize captured video to fixed size
static const int CMUKITCHEN_MOVIE_FRAME_WIDTH = 400;
static const int CMUKITCHEN_MOVIE_FRAME_HEIGHT = 300;

// customized setting for dense trajectory
static const int CMUKITCHEN_TIME_INTERVAL_DETECT_KEYPOINT = 1;
static const int CMUKITCHEN_DENSE_KEYPOINT_STEP = 10; // original paper setting 5

// parameters for leave one out by subject
const char* CMUKITCHEN_SUBJECTS_ARRAY[] = {
	"S07",
	"S08",
	"S09",
	"S12",
	"S13",
	"S14",
	"S16",
	"S17",
	"S18",
	"S19",
	"S20",
	"S22"
};
vector<string> CMUKITCHEN_SUBJECTS(begin(CMUKITCHEN_SUBJECTS_ARRAY), end(CMUKITCHEN_SUBJECTS_ARRAY));

const char* CMUKITCHEN_TARGET_FOODS_ARRAY[] = {
	"Brownie"
};
vector<string> CMUKITCHEN_TARGET_FOODS(begin(CMUKITCHEN_TARGET_FOODS_ARRAY), end(CMUKITCHEN_TARGET_FOODS_ARRAY));

const char* CMUKITCHEN_TARGET_ACTION_ARRAY[] = {
	"none---",
	"stir-big_bowl--",
	"crack-egg--",
	"pour-brownie_bag-into-big_bowl",
	"open-brownie_bag--",
	"pour-oil-into-big_bowl",
	"pour-water-into-big_bowl",
	"pour-big_bowl-into-baking_pan"
};
vector<string> CMUKITCHEN_TARGET_ACTIONS(begin(CMUKITCHEN_TARGET_ACTION_ARRAY), end(CMUKITCHEN_TARGET_ACTION_ARRAY));

pair<string, int> CMUKITCHEN_BROWNIE_MOVIE_START_FRAME_TABLE[] = {
	pair<string, int>("S07", 508),
	pair<string, int>("S08", 300),
	pair<string, int>("S09", 226),
	pair<string, int>("S12", 400),
	pair<string, int>("S13", 290),
	pair<string, int>("S14", 386),
	pair<string, int>("S16", 168),
	pair<string, int>("S17", 236),
	pair<string, int>("S18", 316),
	pair<string, int>("S19", 354),
	pair<string, int>("S20", 212),
	pair<string, int>("S22", 262)
};
map<string, int> CMUKITCHEN_BROWNIE_MOVIE_START_FRAME(CMUKITCHEN_BROWNIE_MOVIE_START_FRAME_TABLE, CMUKITCHEN_BROWNIE_MOVIE_START_FRAME_TABLE+12);

pair<string, int> CMUKITCHEN_BROWNIE_MOVIE_END_FRAME_TABLE[] = {
	pair<string, int>("S07", 10309),
	pair<string, int>("S08", 9000),
	pair<string, int>("S09", 13334),
	pair<string, int>("S12", 15233),
	pair<string, int>("S13", 20151),
	pair<string, int>("S14", 11705),
	pair<string, int>("S16", 12338),
	pair<string, int>("S17", 11518),
	pair<string, int>("S18", 12088),
	pair<string, int>("S19", 14970),
	pair<string, int>("S20", 10576),
	pair<string, int>("S22", 17315)
};
map<string, int> CMUKITCHEN_BROWNIE_MOVIE_END_FRAME(CMUKITCHEN_BROWNIE_MOVIE_END_FRAME_TABLE, CMUKITCHEN_BROWNIE_MOVIE_END_FRAME_TABLE+12);

static const int CMUKITCHEN_FPS = 30;
static const int CMUKITCHEN_RECOG_GESTURE_FRAME_LENGTH = CMUKITCHEN_FPS*2; // We assume each training movie has at least one gesture in this time length
static const int CMUKITCHEN_RECOG_GESTURE_FRAME_STEP = CMUKITCHEN_FPS/2; // Extract train/test gesturess by this time step
static const int CMUKITCHEN_MAX_FRAME_NONE_ACTION = CMUKITCHEN_FPS*5; // For None action class, do not process all frames when creating training data
static const int CMUKITCHEN_ACTION_SKIP_NUM_EVERY_NONE = 4; // For None action class, skip several files after processing every file when creating training data

// parameters for Hand Detection
pair<string, string> CMUKITCHEN_HAND_DETECT_MODEL_LIST_FILES_TABLE[] = {
	pair<string, string>("S07", "..\\..\\data\\model\\CMUKitchen\\S07_Salad_Video\\list.txt"),
	pair<string, string>("S08", "..\\..\\data\\model\\CMUKitchen\\S08_Salad_Video\\list.txt"),
	pair<string, string>("S09", "..\\..\\data\\model\\CMUKitchen\\S09_Salad_Video\\list.txt"),
	pair<string, string>("S12", "..\\..\\data\\model\\CMUKitchen\\S12_Salad_Video\\list.txt"),
	pair<string, string>("S13", "..\\..\\data\\model\\CMUKitchen\\S13_Salad_Video\\list.txt"),
	pair<string, string>("S14", "..\\..\\data\\model\\CMUKitchen\\S14_Salad_Video\\list.txt"),
	pair<string, string>("S16", "..\\..\\data\\model\\CMUKitchen\\S16_Salad_Video\\list.txt"),
	pair<string, string>("S17", "..\\..\\data\\model\\CMUKitchen\\S17_Salad_Video\\list.txt"),
	pair<string, string>("S18", "..\\..\\data\\model\\CMUKitchen\\S18_Salad_Video\\list.txt"),
	pair<string, string>("S19", "..\\..\\data\\model\\CMUKitchen\\S19_Salad_Video\\list.txt"),
	pair<string, string>("S20", "..\\..\\data\\model\\CMUKitchen\\S20_Salad_Video\\list.txt"),
	pair<string, string>("S22", "..\\..\\data\\model\\CMUKitchen\\S22_Salad_Video\\list.txt")
};
map<string, string> CMUKITCHEN_HAND_DETECT_MODEL_LIST_FILES(CMUKITCHEN_HAND_DETECT_MODEL_LIST_FILES_TABLE, CMUKITCHEN_HAND_DETECT_MODEL_LIST_FILES_TABLE+12);

pair<string, string> CMUKITCHEN_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE[] = {
	pair<string, string>("S07", "..\\..\\data\\globfeat\\CMUKitchen\\S07_Salad_Video\\list.txt"),
	pair<string, string>("S08", "..\\..\\data\\globfeat\\CMUKitchen\\S08_Salad_Video\\list.txt"),
	pair<string, string>("S09", "..\\..\\data\\globfeat\\CMUKitchen\\S09_Salad_Video\\list.txt"),
	pair<string, string>("S12", "..\\..\\data\\globfeat\\CMUKitchen\\S12_Salad_Video\\list.txt"),
	pair<string, string>("S13", "..\\..\\data\\globfeat\\CMUKitchen\\S13_Salad_Video\\list.txt"),
	pair<string, string>("S14", "..\\..\\data\\globfeat\\CMUKitchen\\S14_Salad_Video\\list.txt"),
	pair<string, string>("S16", "..\\..\\data\\globfeat\\CMUKitchen\\S16_Salad_Video\\list.txt"),
	pair<string, string>("S17", "..\\..\\data\\globfeat\\CMUKitchen\\S17_Salad_Video\\list.txt"),
	pair<string, string>("S18", "..\\..\\data\\globfeat\\CMUKitchen\\S18_Salad_Video\\list.txt"),
	pair<string, string>("S19", "..\\..\\data\\globfeat\\CMUKitchen\\S19_Salad_Video\\list.txt"),
	pair<string, string>("S20", "..\\..\\data\\globfeat\\CMUKitchen\\S20_Salad_Video\\list.txt"),
	pair<string, string>("S22", "..\\..\\data\\globfeat\\CMUKitchen\\S22_Salad_Video\\list.txt")
};
map<string, string> CMUKITCHEN_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES(CMUKITCHEN_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE, CMUKITCHEN_HAND_DETECT_GLOBAL_FEATURE_LIST_FILES_TABLE+12);

#endif __RECOG_HAND_GESTURE_CMUKITCHEN_CONSTANTS__