
#include <math.h>

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "pipeline.hpp"

using namespace Eigen;
using namespace cv;
using namespace std;

void stereo_calib(const cv::Mat& img_L, const cv::Mat& img_R, Mat& R, Mat& T,
                  Mat& K_L, Mat& K_R, Mat& D1, Mat& D2) {
  Mat gray_L, gray_R;
  Mat E, F;

  vector<Mat> tvecsMat;
  vector<Mat> rvecsMat;
  vector<cv::Point3f> objectpoint;
  vector<vector<cv::Point3f>> objpoint;
  vector<vector<Point2f>> imagePoints1, imagePoints2;
  vector<Point2f> corner_L, corner_R;

  bool foundL = 0, foundR = 0;  //判断成功与否的工具
  int board_Row = 7;
  int board_Col = 11;
  int squaresize = 30;
  Size boardsize = Size(board_Col, board_Row);

  cvtColor(img_L, gray_L, CV_BGR2GRAY);
  cvtColor(img_R, gray_R, CV_BGR2GRAY);
  foundL = findChessboardCorners(img_L, boardsize, corner_L);  //
  foundR = findChessboardCorners(img_R, boardsize, corner_R);
  if (foundL == true && foundR == true) {
    cornerSubPix(
        gray_L, corner_L, cv::Size(5, 5), cv::Size(-1, -1),
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
    cornerSubPix(
        gray_R, corner_R, cv::Size(5, 5), cv::Size(-1, -1),
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
    imagePoints1.push_back(corner_L);
    imagePoints2.push_back(corner_R);
  } else {
    std::cout << "Points Not founded,findChessboardCorners failed!"
              << std::endl;
  }

  for (int i = 0; i < board_Row; i++) {
    for (int j = 0; j < board_Col; j++) {
      objectpoint.push_back(cv::Point3f(i * squaresize, j * squaresize,
                                        0.0f));  // squarsize:30.3mm
    }
  }
  objpoint.push_back(objectpoint);  //根据方块大小算世界坐标，

  calibrateCamera(objpoint, imagePoints1, img_L.size(), K_L, D1, rvecsMat,
                  tvecsMat, 0);
  calibrateCamera(objpoint, imagePoints2, img_R.size(), K_R, D2, rvecsMat,
                  tvecsMat, 0);
  stereoCalibrate(
      objpoint, imagePoints1, imagePoints2, K_L, D1, K_R, D2, img_L.size(), R,
      T, E, F, CALIB_USE_INTRINSIC_GUESS,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                       1e-6));

  cout << "R=" << R << endl;
  cout << "T=" << T << endl;
  cout << "KL=" << K_L << endl;
  cout << "KR=" << K_R << endl;
  cout << "D1=" << D1 << endl;
  cout << "D2=" << D2 << endl;
}