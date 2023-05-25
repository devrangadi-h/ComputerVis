#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;

int main(int argc, char** argv) {

    // Read the input image
    //Mat image = imread("input_image.jpg", IMREAD_GRAYSCALE);

    //// Check if the image was successfully read
    //if (image.empty()) {
    //    cout << "Could not open or find the image" << endl;
    //    return -1;
    //}

    VideoCapture* capdev;
    // open the video device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }


    // Define the parameters for Harris corner detection
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.01;

    // Create a matrix to store the output of Harris corner detection
    Mat corners;

    Mat frame, cameraMatrix, distCoeffs;

    cv::namedWindow("Video", 1);
    bool found = false;

    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        resize(frame, frame, Size(), 0.5, 0.5);

        Mat frame2 = frame.clone();

        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Create a matrix to store the output of Harris corner detection
        Mat corners;

        if (waitKey(1) == 'h') {
            // Apply Harris corner detection
            cornerHarris(grayFrame, corners, blockSize, apertureSize, k);

            // Apply Harris corner detection
            //cornerHarris(frame2, corners, blockSize, apertureSize, k);

            // Normalize the output to values between 0 and 255
            Mat normalizedCorners;
            normalize(corners, normalizedCorners, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

            // Convert the output to a grayscale image
            Mat cornerImage;
            convertScaleAbs(normalizedCorners, cornerImage);

            // Draw circles around the detected corners
            for (int i = 0; i < normalizedCorners.rows; i++) {
                for (int j = 0; j < normalizedCorners.cols; j++) {
                    if (int(normalizedCorners.at<float>(i, j)) > 100) {
                        circle(cornerImage, Point(j, i), 3, Scalar(255, 0, 0), 1, 8, 0);
                    }
                }
            }

            // Display the input image and the output of Harris corner detection
            namedWindow("Input Image", WINDOW_NORMAL);
            imshow("Input Image", frame2);
            namedWindow("Harris Corners", WINDOW_NORMAL);
            imshow("Harris Corners", cornerImage);

        }

        else if (waitKey(1) == 's') {
            //Detect the keypoints using SURF Detector
            int minHessian = 400;
            Ptr<SURF> detector = SURF::create(minHessian);
            std::vector<KeyPoint> keypoints;
            detector->detect(frame, keypoints);
            //Draw keypoints
            Mat img_keypoints;
            drawKeypoints(frame, keypoints, img_keypoints);
            //Show detected (drawn) keypoints
            imshow("SURF Keypoints", img_keypoints);
        }

        else if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera
    delete capdev;

    return 0;
}

