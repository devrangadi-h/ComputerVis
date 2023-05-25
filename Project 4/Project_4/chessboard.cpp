//
//  main.cpp
//  Project4
//
//  Created by Hardik Devrangadi and Thejaswini Goriparthi on 3/14/23.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/calib3d.hpp>
#include "functions.hpp"
#include <string>

using namespace cv;
using namespace std;

int main3()
{

    VideoCapture* capdev;
    // open the video device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    //size of checkerboard
    Size patternSize(8, 6);
    // generate the 3D points for the chessboard corners
    std::vector<cv::Vec3f> points3D = generateChessboard3DPoints(patternSize);
    // declare a vector of vectors to store the 3D object points
    std::vector<std::vector<cv::Vec3f>> objectPoints;
    // declare a vector of vectors to store the 2D image points
    std::vector<std::vector<cv::Point2f>> imagePoints;
    // declare a vector to store the calibration images
    std::vector<cv::Mat> calibration_images;
    // declare a vector to store the 3D points for the corners of a rectangle
    vector<Point3f> fourPoints;
    //define the four corner points
    fourPoints.push_back(Point3f(0, 0, 0));
    fourPoints.push_back(Point3f(7, 0, 0));
    fourPoints.push_back(Point3f(0, 5, 0));
    fourPoints.push_back(Point3f(7, 5, 0));
    // declare a vector to store the 3D points for the corners of an octahedron
    vector <Point3f> octaPoints;
    //define six corners of the octahedron
    octaPoints.push_back(Point3f(2.79289, 2.5, -1.));
    octaPoints.push_back(Point3f(3.5, 3.20711, -1.));
    octaPoints.push_back(Point3f(3.5, 2.5, -1.70711));
    octaPoints.push_back(Point3f(3.5, 2.5, -0.29289));
    octaPoints.push_back(Point3f(3.5, 1.79289, -1.));
    octaPoints.push_back(Point3f(4.20711, 2.5, -1.));


    Mat frame, cameraMatrix, distCoeffs;

    cv::namedWindow("Video", 1);
    bool found = false;

    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        //resizing the original frame
        resize(frame, frame, Size(), 0.5, 0.5);

        Mat frame2 = frame.clone();

        // Find the corners of the target pattern in the input image
        vector<Point2f> corners;

        found = extractChessboardCorners(frame, patternSize, corners); //extract chessboard corners from current frame
        if (found) { // display the chessboard corners
            drawChessboardCorners(frame2, patternSize, corners, found); //draw chessboard corners on the resized frame
        }

        //To obtain the maximum and minimum x and y values from all the items of the corner vector
        float minx = 10000, miny = 10000;
        float maxx = 0, maxy = 0;
        for (int i = 0; i < corners.size(); i++) {

            if (minx > corners[i].x)
                minx = corners[i].x; //minimum of x
            if (miny > corners[i].y)
                miny = corners[i].y; //minimum of y
            if (maxx < corners[i].x)
                maxx = corners[i].x; //maximum of x
            if (maxy < corners[i].y)
                maxy = corners[i].y; //maximum of y
        }

        //Vector to store the destination points for perspective transformation
        vector<Point2f> dstpoints;

        //Adding the four corner points to the vector of destination image
        dstpoints.push_back(Point2f(minx, miny));
        dstpoints.push_back(Point2f(maxx, miny));
        dstpoints.push_back(Point2f(minx, maxy));
        dstpoints.push_back(Point2f(maxx, maxy));

        //vector to store the source points
        vector<Point2f> srcpoints;
        //Adding the four corner points to the vector of source(checkerboard) image for each of the destination point
        srcpoints.push_back(Point2f(200, 200));
        srcpoints.push_back(Point2f(1719, 200));
        srcpoints.push_back(Point2f(200, 1077));
        srcpoints.push_back(Point2f(1719, 1077));

        //homography matrix using source and destination points
        Mat h = findHomography(srcpoints, dstpoints);
        Mat dstpic = frame2.clone();
        // Apply the perspective transformation to the 'dst' image and store the result in 'dstpic'
        warpPerspective(dst, dstpic, h, dstpic.size());

        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        // If 't' is pressed, save frames to files
        else if (key == 't') {
            string filename1 = "/Users/hardikdevrangadi/Desktop/Project4/osave6.jpg";
            string filename2 = "/Users/hardikdevrangadi/Desktop/Project4/save6.jpg";
            imwrite(filename1, frame);
            imwrite(filename2, frame2);
        }
        // If 's' is pressed, save calibration points if corners are found
        else if (key == 's') {
            if (found) {  // Save the corner and point sets
                imagePoints.push_back(corners);
                objectPoints.push_back(points3D);
                calibration_images.push_back(frame2);
                printf("Saved %d corners\n", (int)corners.size());
                printf("Saved %d points\n", (int)points3D.size());
            }

            else {
                printf("Chessboard not found\n");
            }
        }
        // If there are enough saved images, calibrate the camera and plot four corners
        if (imagePoints.size() > 5) {
            calibrateFrame(frame2, objectPoints, imagePoints, cameraMatrix, distCoeffs);
            fourCornerPlot(frame2, fourPoints, points3D, corners, cameraMatrix, distCoeffs);
            octahedronPlot(frame2,octaPoints,points3D,corners,cameraMatrix,distCoeffs);
        }
        // If 'p' is pressed, toggle the flag (extension #)
        else if (key == 'p') {
            flag = !flag;
        }
        // If the flag is true, replace pixels in the frame with pixels from a destination image
        if (flag == true) {

            for (int i = 0; i < dstpic.rows; i++) {
                for (int j = 0; j < dstpic.cols; j++) {
                    if (dstpic.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                        frame2.at<Vec3b>(i, j) = dstpic.at<Vec3b>(i, j);
                    }
                }
            }

        }


        imshow("VideoMain", frame2);
    }
    delete capdev;
    return 0;
}
