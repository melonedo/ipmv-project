#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "pipeline.hpp"

using namespace cv;
using namespace std;


// #define SHOW_DISPARITY

void compute_cost(const Mat& image_L, const Mat& image_R, Mat& cost_L,
                  Mat& cost_R) {
  const size_t Row = image_L.size[0];
  const size_t Col = image_L.size[1];
  const size_t MaxDistance = cost_L.size[0];

  Mat sum_L(Row, Col, CV_32SC1);
  Mat sum_R(Row, Col, CV_32SC1);
  Mat sqr_sum_L(Row, Col, CV_32SC1);
  Mat sqr_sum_R(Row, Col, CV_32SC1);


  Mat gray_L, gray_R;
  cvtColor(image_L, gray_L, CV_BGR2GRAY);
  cvtColor(image_R, gray_R, CV_BGR2GRAY);

  //左图灰度前缀和
  sum_L.at<int32_t>(0, 0) = gray_L.at<uint8_t>(0, 0);
  for (int y = 1; y < Col; y++)
    sum_L.at<int32_t>(0, y) =
        sum_L.at<int32_t>(0, y - 1) + gray_L.at<uint8_t>(0, y);
  for (int x = 1; x < Row; x++) {
    sum_L.at<int32_t>(x, 0) =
        sum_L.at<int32_t>(x - 1, 0) + gray_L.at<uint8_t>(x, 0);
    for (int y = 1; y < Col; y++)
      sum_L.at<int32_t>(x, y) =
          sum_L.at<int32_t>(x, y - 1) + sum_L.at<int32_t>(x - 1, y) -
          sum_L.at<int32_t>(x - 1, y - 1) + gray_L.at<uint8_t>(x, y);
  }

  //右图灰度前缀和
  sum_R.at<int32_t>(0, 0) = gray_R.at<uint8_t>(0, 0);
  for (int y = 1; y < Col; y++)
    sum_R.at<int32_t>(0, y) =
        sum_R.at<int32_t>(0, y - 1) + gray_R.at<uint8_t>(0, y);
  for (int x = 1; x < Row; x++) {
    sum_R.at<int32_t>(x, 0) =
        sum_R.at<int32_t>(x - 1, 0) + gray_R.at<uint8_t>(x, 0);
    for (int y = 1; y < Col; y++)
      sum_R.at<int32_t>(x, y) =
          sum_R.at<int32_t>(x, y - 1) + sum_R.at<int32_t>(x - 1, y) -
          sum_R.at<int32_t>(x - 1, y - 1) + gray_R.at<uint8_t>(x, y);
  }

  //左图灰度平方前缀和
  sqr_sum_L.at<int32_t>(0, 0) =
      gray_L.at<uint8_t>(0, 0) * gray_L.at<uint8_t>(0, 0);
  for (int y = 1; y < Col; y++)
    sqr_sum_L.at<int32_t>(0, y) =
        sqr_sum_L.at<int32_t>(0, y - 1) +
        gray_L.at<uint8_t>(0, y) * gray_L.at<uint8_t>(0, y);
  for (int x = 1; x < Row; x++) {
    sqr_sum_L.at<int32_t>(x, 0) =
        sqr_sum_L.at<int32_t>(x - 1, 0) +
        gray_L.at<uint8_t>(x, 0) * gray_L.at<uint8_t>(x, 0);
    for (int y = 1; y < Col; y++)
      sqr_sum_L.at<int32_t>(x, y) =
          sqr_sum_L.at<int32_t>(x, y - 1) + sqr_sum_L.at<int32_t>(x - 1, y) -
          sqr_sum_L.at<int32_t>(x - 1, y - 1) +
          gray_L.at<uint8_t>(x, y) * gray_L.at<uint8_t>(x, y);
  }

  //右图灰度平方前缀和
  sqr_sum_R.at<int32_t>(0, 0) =
      gray_R.at<uint8_t>(0, 0) * gray_R.at<uint8_t>(0, 0);
  for (int y = 1; y < Col; y++)
    sqr_sum_R.at<int32_t>(0, y) =
        sqr_sum_R.at<int32_t>(0, y - 1) +
        gray_R.at<uint8_t>(0, y) * gray_R.at<uint8_t>(0, y);
  for (int x = 1; x < Row; x++) {
    sqr_sum_R.at<int32_t>(x, 0) =
        sqr_sum_R.at<int32_t>(x - 1, 0) +
        gray_R.at<uint8_t>(x, 0) * gray_R.at<uint8_t>(x, 0);
    for (int y = 1; y < Col; y++)
      sqr_sum_R.at<int32_t>(x, y) =
          sqr_sum_R.at<int32_t>(x, y - 1) + sqr_sum_R.at<int32_t>(x - 1, y) -
          sqr_sum_R.at<int32_t>(x - 1, y - 1) +
          gray_R.at<uint8_t>(x, y) * gray_R.at<uint8_t>(x, y);
  }

  cost_L.setTo(Scalar::all(0));
  cost_R.setTo(Scalar::all(0));

#ifndef SHOW_DISPARITY
#pragma omp parallel for
#endif
  for (int d = 0; d < MaxDistance; d++) {
#ifdef SHOW_DISPARITY
    Mat temp(Row, Col, CV_32FC1);
#endif
    Mat sum(Row, Col, CV_32SC1);
    //分子前缀和，以左图坐标为准
    sum.at<int32_t>(0, d) = gray_L.at<uint8_t>(0, d) * gray_R.at<uint8_t>(0, 0);
    for (int y = d + 1; y < Col; y++)
      sum.at<int32_t>(0, y) =
          sum.at<int32_t>(0, y - 1) +
          gray_L.at<uint8_t>(0, y) * gray_R.at<uint8_t>(0, y - d);
    for (int x = 1; x < Row; x++) {
      sum.at<int32_t>(x, d) =
          sum.at<int32_t>(x - 1, d) +
          gray_L.at<uint8_t>(x, d) * gray_R.at<uint8_t>(x, 0);
      for (int y = d + 1; y < Col; y++)
        sum.at<int32_t>(x, y) =
            sum.at<int32_t>(x - 1, y) + sum.at<int32_t>(x, y - 1) -
            sum.at<int32_t>(x - 1, y - 1) +
            gray_L.at<uint8_t>(x, y) * gray_R.at<uint8_t>(x, y - d);
    }

    for (int x = 1 + RAD; x < Row - 1 - RAD; x++)
      for (int y = d + 1 + RAD; y < Col - 1 - RAD; y++) {
        //左图区块
        int xMax_L = x + RAD;
        int xMin_L = x - RAD;
        int yMax_L = y + RAD;
        int yMin_L = y - RAD;
        //右图区块
        int xMax_R = x + RAD;
        int xMin_R = x - RAD;
        int yMax_R = y - d + RAD;
        int yMin_R = y - d - RAD;
        //像素个数
        int pixels = (2 * RAD + 1) * (2 * RAD + 1);
        //左图像素强度和标准差
        float I_L = sum_L.at<int32_t>(xMax_L, yMax_L) -
              sum_L.at<int32_t>(xMin_L - 1, yMax_L) -
              sum_L.at<int32_t>(xMax_L, yMin_L - 1) +
              sum_L.at<int32_t>(xMin_L - 1, yMin_L - 1);
        float var_L = sqr_sum_L.at<int32_t>(xMax_L, yMax_L) -
                sqr_sum_L.at<int32_t>(xMin_L - 1, yMax_L) -
                sqr_sum_L.at<int32_t>(xMax_L, yMin_L - 1) +
                sqr_sum_L.at<int32_t>(xMin_L - 1, yMin_L - 1);
        var_L = sqrt(var_L - I_L * I_L / (float)pixels);
        I_L /= (float)pixels;
        //右图像素强度和标准差
        float I_R = sum_R.at<int32_t>(xMax_R, yMax_R) -
              sum_R.at<int32_t>(xMin_R - 1, yMax_R) -
              sum_R.at<int32_t>(xMax_R, yMin_R - 1) +
              sum_R.at<int32_t>(xMin_R - 1, yMin_R - 1);
        float var_R = sqr_sum_R.at<int32_t>(xMax_R, yMax_R) -
                sqr_sum_R.at<int32_t>(xMin_R - 1, yMax_R) -
                sqr_sum_R.at<int32_t>(xMax_R, yMin_R - 1) +
                sqr_sum_R.at<int32_t>(xMin_R - 1, yMin_R - 1);
        var_R = sqrt(var_R - I_R * I_R / (float)pixels);
        I_R /= (float)pixels;
        //左右图对比
        float sum_LR = sum.at<int32_t>(xMax_L, yMax_L) -
                 sum.at<int32_t>(xMin_L - 1, yMax_L) -
                 sum.at<int32_t>(xMax_L, yMin_L - 1) +
                 sum.at<int32_t>(xMin_L - 1, yMin_L - 1);
        if (var_L * var_R != 0) {
          cost_L.at<float>(d, x, y) =
              (sum_LR - (float)pixels * I_L * I_R) / (var_L * var_R);
          cost_R.at<float>(d, x, y - d) =
              (sum_LR - (float)pixels * I_L * I_R) / (var_L * var_R);
#ifdef SHOW_DISPARITY
          temp.at<float>(x, y) = cost_L.at<float>(d, x, y);
#endif
        }
      }
#ifdef SHOW_DISPARITY
      namedWindow("test", WINDOW_NORMAL);
      putText(temp, "d="s + std::to_string(d), {0, 150}, FONT_HERSHEY_SIMPLEX, 3, Scalar{1}, 5, 8, false);
      imshow("test", temp);
      waitKey(1);
#endif
  }
}
