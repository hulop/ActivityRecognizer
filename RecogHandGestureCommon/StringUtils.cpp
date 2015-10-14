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

#include "StringUtils.h"

vector<string> StringUtils::splitString(const string &str, const string &delim){
	vector<string> res;
	size_t current = 0, found, delimlen = delim.size();
	while((found = str.find(delim, current)) != string::npos){
		res.push_back(string(str, current, found - current));
		current = found + delimlen;
	}
	res.push_back(string(str, current, str.size() - current));
	return res;
}

string StringUtils::trim(string const& str)
{
	if(str.empty()) {
		return str;
	}

	size_t first = str.find_first_not_of(' ');
	std::size_t last  = str.find_last_not_of(' ');
	return str.substr(first, last-first+1);
}