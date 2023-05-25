//
//  functions.cpp
//  Project4
//
//  Created by Hardik Devrangadi and Thejaswini Goriparthi on 3/14/23.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "functions.h"

//function to extract the chessboard corners
bool extractChessboardCorners(cv::Mat& frame, cv::Size& patternSize, std::vector<cv::Point2f>& corners) {
    bool foundCorners = findChessboardCorners(frame, patternSize, corners); //finding the corners of a chessboard
    if (foundCorners) {
        cv::Mat grayscale;
        cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
        // Refine the corner locations to subpixel accuracy
        cv::TermCriteria termCrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 1, 0.1);
        cornerSubPix(grayscale, corners, cv::Size(10, 10), cv::Size(-1, -1), termCrit);

    }
    // Return a boolean indicating whether the corners were successfully found
    return foundCorners;
}

std::vector<cv::Vec3f> generateChessboard3DPoints(cv::Size& patternSize)
{
    // Generate a set of 3D points corresponding to the chessboard corners
    std::vector<cv::Vec3f> corners;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            corners.push_back(cv::Vec3f(j, i, 0));
        }
    }
    // Return the 3D points
    return corners;
}

void calibrateFrame(cv::Mat& frame, std::vector<std::vector<cv::Vec3f>>& objectPoints, std::vector<std::vector<cv::Point2f>>& imagePoints, Mat& cameraMatrix, Mat& distCoeffs) {
    // Define the size of the calibration images
    cv::Size imageSize(frame.cols, frame.rows);

    // Initialize the camera matrix
    cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1);
    cameraMatrix.at<double>(0, 2) = imageSize.width / 2;
    cameraMatrix.at<double>(1, 2) = imageSize.height / 2;

    // Initialize the distortion coefficients
    distCoeffs = cv::Mat::zeros(5, 1, CV_64FC1);

    // Calibrate the camera
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    //returning the camera matrix obtained from Calibrate camera function
    std::cout << "Camera matrix:" << std::endl;
    std::cout << cameraMatrix << std::endl;
    //returning the distortion coefficients obtained from Calibrate camera function
    std::cout << "Distortion coefficients:" << std::endl;
    std::cout << distCoeffs << std::endl;
    //returning the reprojection error of the camera used
    std::cout << "Reprojection error: " << rms << std::endl;

}


int fourCornerPlot(Mat& frame, std::vector<Point3f>& fourPoints, std::vector<cv::Vec3f>& objectPoints, std::vector<cv::Point2f>& imagePoints, Mat& cameraMatrix, Mat& distCoeffs) {

    // Declare variables for rotation and translation vectors
    cv::Mat rvec, tvec;
    vector<Point2f> projectedPoints;

    // Use solvePnP to compute rotation and translation vectors from the 3D object points and their corresponding 2D image points
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);


    // Print out rotation and translation data
    std::cout << "Rotation vector: " << rvec.t() << std::endl;
    std::cout << "Translation vector: " << tvec.t() << std::endl;

    // Iterate through the four corner points and project each one onto the image plane
    for (int i = 0; i < 4; i++) {
        cv::projectPoints(fourPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
        // Draw a circle at each projected point
        cv::circle(frame, projectedPoints[i], 3, Scalar(255, 255, 255), 3);
    }

    //    line(frame, projectedPoints[0], projectedPoints[1], Scalar(147, 20, 255), 2);
    //    line(frame, projectedPoints[1], projectedPoints[2], Scalar(147, 20, 255), 2);
    //    line(frame, projectedPoints[2], projectedPoints[3], Scalar(147, 20, 255), 2);
    //    line(frame, projectedPoints[3], projectedPoints[0], Scalar(147, 20, 255), 2);
    //    line(frame, projectedPoints[0], projectedPoints[2], Scalar(147, 20, 255), 2);
    //    line(frame, projectedPoints[1], projectedPoints[3], Scalar(147, 20, 255), 2);


    return 0;
}

int octahedronPlot(Mat& frame, std::vector<Point3f>& octaPoints, std::vector<cv::Vec3f>& objectPoints, std::vector<cv::Point2f>& imagePoints, Mat& cameraMatrix, Mat& distCoeffs) {
    cv::Mat rvec, tvec;
    vector<Point2f> projectedPoints;
    // Use solvePnP to compute rotation and translation vectors from the 3D object points and their corresponding 2D image points
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    // Iterate through the six octahedron points and project each one onto the image plane
    for (int i = 0; i < 6; i++) {
        cv::projectPoints(octaPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
        cv::circle(frame, projectedPoints[i], 3, Scalar(255, 255, 255), 2);
    }
    line(frame, projectedPoints[0], projectedPoints[1], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[0], projectedPoints[2], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[0], projectedPoints[3], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[0], projectedPoints[4], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[0], projectedPoints[5], Scalar(147, 20, 255), 1);

    line(frame, projectedPoints[1], projectedPoints[2], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[1], projectedPoints[3], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[1], projectedPoints[4], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[1], projectedPoints[5], Scalar(147, 20, 255), 1);

    line(frame, projectedPoints[2], projectedPoints[3], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[2], projectedPoints[4], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[2], projectedPoints[5], Scalar(147, 20, 255), 1);

    line(frame, projectedPoints[3], projectedPoints[4], Scalar(147, 20, 255), 1);
    line(frame, projectedPoints[3], projectedPoints[5], Scalar(147, 20, 255), 1);

    line(frame, projectedPoints[4], projectedPoints[5], Scalar(147, 20, 255), 1);
    return 0;
}
