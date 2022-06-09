#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "pipeline.hpp"

using namespace cv;
using namespace std;

#define rad 1

void compute_cost(const Mat& image_L, const Mat& image_R, Mat& cost_L,
                  Mat& cost_R) {
  size_t Row = image_L.size[0];
  size_t Col = image_L.size[1];
  size_t MaxDistance = cost_L.size[2];

  Mat sum(Row, Col, CV_32SC1);
  Mat sum_L(Row, Col, CV_32SC1);
  Mat sum_R(Row, Col, CV_32SC1);
  Mat sqr_sum_L(Row, Col, CV_32SC1);
  Mat sqr_sum_R(Row, Col, CV_32SC1);
  Mat temp(Row, Col, CV_32SC1);

  int pixels;
  float sum_LR;
  float I_L, I_R;
  float var_L, var_R;
  int xMax_L, xMin_L, yMax_L, yMin_L;
  int xMax_R, xMin_R, yMax_R, yMin_R;
  int x, y, d;

  Mat gray_L, gray_R;
  cvtColor(image_L, gray_L, CV_BGR2GRAY);
  cvtColor(image_R, gray_R, CV_BGR2GRAY);

  //左图灰度前缀和
  sum_L.at<int32_t>(0, 0) = gray_L.at<uint8_t>(0, 0);
  for (y = 1; y < Col; y++)
    sum_L.at<int32_t>(0, y) =
        sum_L.at<int32_t>(0, y - 1) + gray_L.at<uint8_t>(0, y);
  for (x = 1; x < Row; x++) {
    sum_L.at<int32_t>(x, 0) =
        sum_L.at<int32_t>(x - 1, 0) + gray_L.at<uint8_t>(x, 0);
    for (y = 1; y < Col; y++)
      sum_L.at<int32_t>(x, y) =
          sum_L.at<int32_t>(x, y - 1) + sum_L.at<int32_t>(x - 1, y) -
          sum_L.at<int32_t>(x - 1, y - 1) + gray_L.at<uint8_t>(x, y);
  }

  //右图灰度前缀和
  sum_R.at<int32_t>(0, 0) = gray_R.at<uint8_t>(0, 0);
  for (y = 1; y < Col; y++)
    sum_R.at<int32_t>(0, y) =
        sum_R.at<int32_t>(0, y - 1) + gray_R.at<uint8_t>(0, y);
  for (x = 1; x < Row; x++) {
    sum_R.at<int32_t>(x, 0) =
        sum_R.at<int32_t>(x - 1, 0) + gray_R.at<uint8_t>(x, 0);
    for (y = 1; y < Col; y++)
      sum_R.at<int32_t>(x, y) =
          sum_R.at<int32_t>(x, y - 1) + sum_R.at<int32_t>(x - 1, y) -
          sum_R.at<int32_t>(x - 1, y - 1) + gray_R.at<uint8_t>(x, y);
  }

  //左图灰度平方前缀和
  sqr_sum_L.at<int32_t>(0, 0) =
      gray_L.at<uint8_t>(0, 0) * gray_L.at<uint8_t>(0, 0);
  for (y = 1; y < Col; y++)
    sqr_sum_L.at<int32_t>(0, y) =
        sqr_sum_L.at<int32_t>(0, y - 1) +
        gray_L.at<uint8_t>(0, y) * gray_L.at<uint8_t>(0, y);
  for (x = 1; x < Row; x++) {
    sqr_sum_L.at<int32_t>(x, 0) =
        sqr_sum_L.at<int32_t>(x - 1, 0) +
        gray_L.at<uint8_t>(x, 0) * gray_L.at<uint8_t>(x, 0);
    for (y = 1; y < Col; y++)
      sqr_sum_L.at<int32_t>(x, y) =
          sqr_sum_L.at<int32_t>(x, y - 1) + sqr_sum_L.at<int32_t>(x - 1, y) -
          sqr_sum_L.at<int32_t>(x - 1, y - 1) +
          gray_L.at<uint8_t>(x, y) * gray_L.at<uint8_t>(x, y);
  }

  //右图灰度平方前缀和
  sqr_sum_R.at<int32_t>(0, 0) =
      gray_R.at<uint8_t>(0, 0) * gray_R.at<uint8_t>(0, 0);
  for (y = 1; y < Col; y++)
    sqr_sum_R.at<int32_t>(0, y) =
        sqr_sum_R.at<int32_t>(0, y - 1) +
        gray_R.at<uint8_t>(0, y) * gray_R.at<uint8_t>(0, y);
  for (x = 1; x < Row; x++) {
    sqr_sum_R.at<int32_t>(x, 0) =
        sqr_sum_R.at<int32_t>(x - 1, 0) +
        gray_R.at<uint8_t>(x, 0) * gray_R.at<uint8_t>(x, 0);
    for (y = 1; y < Col; y++)
      sqr_sum_R.at<int32_t>(x, y) =
          sqr_sum_R.at<int32_t>(x, y - 1) + sqr_sum_R.at<int32_t>(x - 1, y) -
          sqr_sum_R.at<int32_t>(x - 1, y - 1) +
          gray_R.at<uint8_t>(x, y) * gray_R.at<uint8_t>(x, y);
  }

  for (d = 0; d <= MaxDistance; d++) {
    //分子前缀和，以左图坐标为准
    sum.at<int32_t>(0, d) = gray_L.at<uint8_t>(0, d) * gray_R.at<uint8_t>(0, 0);
    for (y = d + 1; y < Col; y++)
      sum.at<int32_t>(0, y) =
          sum.at<int32_t>(0, y - 1) +
          gray_L.at<uint8_t>(0, y) * gray_R.at<uint8_t>(0, y - d);
    for (x = 1; x < Row; x++) {
      sum.at<int32_t>(x, d) =
          sum.at<int32_t>(x - 1, d) +
          gray_L.at<uint8_t>(x, d) * gray_R.at<uint8_t>(x, 0);
      for (y = d + 1; y < Col; y++)
        sum.at<int32_t>(x, y) =
            sum.at<int32_t>(x - 1, y) + sum.at<int32_t>(x, y - 1) -
            sum.at<int32_t>(x - 1, y - 1) +
            gray_L.at<uint8_t>(x, y) * gray_R.at<uint8_t>(x, y - d);
    }
    temp.setTo(Scalar::all(0));
    for (x = 1 + rad; x < Row - 1 - rad; x++)
      for (y = d + 1 + rad; y < Col - 1 - rad; y++) {
        //左图区块
        xMax_L = x + rad;
        xMin_L = x - rad;
        yMax_L = y + rad;
        yMin_L = y - rad;
        //右图区块
        xMax_R = x + rad;
        xMin_R = x - rad;
        yMax_R = y - d + rad;
        yMin_R = y - d - rad;
        //像素个数
        pixels = (2 * rad + 1) * (2 * rad + 1);
        //左图像素强度和标准差
        I_L = sum_L.at<int32_t>(xMax_L, yMax_L) -
              sum_L.at<int32_t>(xMin_L - 1, yMax_L) -
              sum_L.at<int32_t>(xMax_L, yMin_L - 1) +
              sum_L.at<int32_t>(xMin_L - 1, yMin_L - 1);
        var_L = sqr_sum_L.at<int32_t>(xMax_L, yMax_L) -
                sqr_sum_L.at<int32_t>(xMin_L - 1, yMax_L) -
                sqr_sum_L.at<int32_t>(xMax_L, yMin_L - 1) +
                sqr_sum_L.at<int32_t>(xMin_L - 1, yMin_L - 1);
        var_L = sqrt(var_L - I_L * I_L / (float)pixels);
        I_L /= (float)pixels;
        //右图像素强度和标准差
        I_R = sum_R.at<int32_t>(xMax_R, yMax_R) -
              sum_R.at<int32_t>(xMin_R - 1, yMax_R) -
              sum_R.at<int32_t>(xMax_R, yMin_R - 1) +
              sum_R.at<int32_t>(xMin_R - 1, yMin_R - 1);
        var_R = sqr_sum_R.at<int32_t>(xMax_R, yMax_R) -
                sqr_sum_R.at<int32_t>(xMin_R - 1, yMax_R) -
                sqr_sum_R.at<int32_t>(xMax_R, yMin_R - 1) +
                sqr_sum_R.at<int32_t>(xMin_R - 1, yMin_R - 1);
        var_R = sqrt(var_R - I_R * I_R / (float)pixels);
        I_R /= (float)pixels;
        //左右图对比
        sum_LR = sum.at<int32_t>(xMax_L, yMax_L) -
                 sum.at<int32_t>(xMin_L - 1, yMax_L) -
                 sum.at<int32_t>(xMax_L, yMin_L - 1) +
                 sum.at<int32_t>(xMin_L - 1, yMin_L - 1);
        if (var_L * var_R != 0) {
          temp.at<float>(x, y) = cost_L.at<float>(d, x, y) =
              (sum_LR - (float)pixels * I_L * I_R) / (var_L * var_R);
          cost_R.at<float>(d, x, y - d) =
              (sum_LR - (float)pixels * I_L * I_R) / (var_L * var_R);
        }
      }
    /*if (d == 142) {
    namedWindow("test", WINDOW_NORMAL);
    imshow("test", temp);
    waitKey(0);
    }*/
  }
}