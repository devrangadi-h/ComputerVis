//
//  functions.hpp
//  Project4
//
//  Created by Hardik Devrangadi and Thejaswini Goriparthi on 3/14/23.
//

#ifndef functions_hpp
#define functions_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

bool extractChessboardCorners(cv::Mat& frame, cv::Size& patternSize, std::vector<cv::Point2f>& corners);
std::vector<cv::Vec3f> generateChessboard3DPoints(cv::Size& patternSize);
void calibrateFrame(cv::Mat & frame, std::vector<std::vector<cv::Vec3f>> & objectPoints, std::vector<std::vector<cv::Point2f>> & imagePoints, Mat & cameraMatrix, Mat & distCoeffs);
int fourCornerPlot(Mat& frame, std::vector<Point3f>& fourPoints, std::vector<cv::Vec3f>& objectPoints, std::vector<cv::Point2f>& imagePoints, Mat& cameraMatrix, Mat& distCoeffs);
int octahedronPlot(Mat& frame, std::vector<Point3f>& octaPoints, std::vector<cv::Vec3f>& objectPoints, std::vector<cv::Point2f>& imagePoints, Mat& cameraMatrix, Mat& distCoeffs);
#endif /* functions_hpp */
#pragma once
