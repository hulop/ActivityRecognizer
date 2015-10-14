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

#include "FileUtils.h"

#include <fstream>

string FileUtils::getFileName(const string &path)
{
	size_t pos1;

	pos1 = path.rfind('\\');
	if(pos1 != string::npos){
		return path.substr(pos1+1, path.size()-pos1-1);
	}

	pos1 = path.rfind('/');
	if(pos1 != string::npos){
		return path.substr(pos1+1, path.size()-pos1-1);
	}

    return path;
}

void FileUtils::readTSV(const string& tsvFile, vector<vector<string> >& tokens) {
    ifstream file(tsvFile.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

	for (int i=0; i<tokens.size(); i++) {
		tokens[i].clear();
	}
	tokens.clear();

	string line;
	while (getline(file,line)) {
		vector<string> lineTokens;

		string token;
		istringstream stream(line);
		while (getline(stream, token, '\t')) {
			if (token.size()>0) {
				lineTokens.push_back(token);
			}
		}
		//CV_Assert(tokens.size()==0 || tokens[tokens.size()-1].size()==lineTokens.size());

		tokens.push_back(lineTokens);
	}
}

void FileUtils::readSSV(const string& tsvFile, vector<vector<string> >& tokens) {
    ifstream file(tsvFile.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

	for (int i=0; i<tokens.size(); i++) {
		tokens[i].clear();
	}
	tokens.clear();

	string line;
	while (getline(file,line)) {
		vector<string> lineTokens;

		string token;
		istringstream stream(line);
		while (getline(stream, token, ' ')) {
			if (token.size()>0) {
				lineTokens.push_back(token);
			}
		}
		//CV_Assert(tokens.size()==0 || tokens[tokens.size()-1].size()==lineTokens.size());

		tokens.push_back(lineTokens);
	}
}

void FileUtils::saveMat(string filename, Mat& mat) 
{
	cv::FileStorage storage(filename, cv::FileStorage::WRITE);
	storage << "mat" << mat;
	storage.release();
}

void FileUtils::readMat(string filename, Mat &mat) 
{
	cv::FileStorage storage(filename, cv::FileStorage::READ);
	storage["mat"] >> mat;
	storage.release();
}

void FileUtils::saveMatBin(string filename, Mat& mat)
{
	std::ofstream ofs(filename, std::ios::binary);
	CV_Assert(ofs.is_open());

	if(mat.empty()){
		int s = 0;
		ofs.write((const char*)(&s), sizeof(int));
		return;
	}

	int type = mat.type();
	ofs.write((const char*)(&mat.rows), sizeof(int));
	ofs.write((const char*)(&mat.cols), sizeof(int));
	ofs.write((const char*)(&type), sizeof(int));
	ofs.write((const char*)(mat.data), mat.elemSize() * mat.total());
}

void FileUtils::readMatBin(string filename, Mat& mat)
{
	std::ifstream ifs(filename, std::ios::binary);
	CV_Assert(ifs.is_open());
	
	int rows, cols, type;
	ifs.read((char*)(&rows), sizeof(int));
	if(rows==0){
		return;
	}
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int));

	mat.release();
	mat.create(rows, cols, type);
	ifs.read((char*)(mat.data), mat.elemSize() * mat.total());
}