#include <math.h>

#include "pipeline.hpp"

using namespace cv;

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

  int pixels;
  float sum_LR;
  float I_L, I_R;
  float var_L, var_R;
  int xMax_L, xMin_L, yMax_L, yMin_L;
  int xMax_R, xMin_R, yMax_R, yMin_R;
  int i, j, d;

  //左图灰度前缀和
  sum_L.at<int32_t>(0, 0) = image_L.at<uint8_t>(0, 0);
  for (j = 1; j < Col; j++)
    sum_L.at<int32_t>(0, j) =
        sum_L.at<int32_t>(0, j - 1) + image_L.at<uint8_t>(0, j);
  for (i = 1; i < Row; i++) {
    sum_L.at<int32_t>(i, 0) =
        sum_L.at<int32_t>(i - 1, 0) + image_L.at<uint8_t>(i, 0);
    for (j = 1; j < Col; j++)
      sum_L.at<int32_t>(i, j) =
          sum_L.at<int32_t>(i, j - 1) + sum_L.at<int32_t>(i - 1, j) -
          sum_L.at<int32_t>(i - 1, j - 1) + image_L.at<uint8_t>(i, j);
  }

  //右图灰度前缀和
  sum_R.at<int32_t>(0, 0) = image_R.at<uint8_t>(0, 0);
  for (j = 1; j < Col; j++)
    sum_R.at<int32_t>(0, j) =
        sum_R.at<int32_t>(0, j - 1) + image_R.at<uint8_t>(0, j);
  for (i = 1; i < Row; i++) {
    sum_R.at<int32_t>(i, 0) =
        sum_R.at<int32_t>(i - 1, 0) + image_R.at<uint8_t>(i, 0);
    for (j = 1; j < Col; j++)
      sum_R.at<int32_t>(i, j) =
          sum_R.at<int32_t>(i - 1, j) + sum_R.at<int32_t>(i, j - 1) -
          sum_R.at<int32_t>(i - 1, j - 1) + image_R.at<uint8_t>(i, j);
  }

  //左图灰度平方前缀和
  sqr_sum_L.at<int32_t>(0, 0) =
      image_L.at<uint8_t>(0, 0) * image_L.at<uint8_t>(0, 0);
  for (j = 1; j < Col; j++)
    sqr_sum_L.at<int32_t>(0, j) =
        sqr_sum_L.at<int32_t>(0, j - 1) +
        image_L.at<uint8_t>(0, j) * image_L.at<uint8_t>(0, j);
  for (i = 1; i < Row; i++) {
    sqr_sum_L.at<int32_t>(i, 0) =
        sqr_sum_L.at<int32_t>(i - 1, 0) +
        image_L.at<uint8_t>(i, 0) * image_L.at<uint8_t>(i, 0);
    for (j = 1; j < Col; j++)
      sqr_sum_L.at<int32_t>(i, j) =
          sqr_sum_L.at<int32_t>(i, j - 1) + sqr_sum_L.at<int32_t>(i - 1, j) -
          sqr_sum_L.at<int32_t>(i - 1, j - 1) +
          image_L.at<uint8_t>(i, j) * image_L.at<uint8_t>(i, j);
  }

  //右图灰度平方前缀和
  sqr_sum_R.at<int32_t>(0, 0) =
      image_R.at<uint8_t>(0, 0) * image_R.at<uint8_t>(0, 0);
  for (j = 1; j < Col; j++)
    sqr_sum_R.at<int32_t>(0, j) =
        sqr_sum_R.at<int32_t>(0, j - 1) +
        image_R.at<uint8_t>(0, j) * image_R.at<uint8_t>(0, j);
  for (i = 1; i < Row; i++) {
    sqr_sum_R.at<int32_t>(i, 0) =
        sqr_sum_R.at<int32_t>(i - 1, 0) +
        image_R.at<uint8_t>(i, 0) * image_R.at<uint8_t>(i, 0);
    for (j = 1; j < Col; j++)
      sqr_sum_R.at<int32_t>(i, j) =
          sqr_sum_R.at<int32_t>(i, j - 1) + sqr_sum_R.at<int32_t>(i - 1, j) -
          sqr_sum_R.at<int32_t>(i - 1, j - 1) +
          image_R.at<uint8_t>(i, j) * image_R.at<uint8_t>(i, j);
  }

  for (d = 0; d <= MaxDistance; d++) {
    //分子前缀和
    sum.at<int32_t>(d, 0) =
        image_L.at<uint8_t>(d, 0) * image_R.at<uint8_t>(0, 0);
    for (j = 1; j < Col; j++)
      sum.at<int32_t>(d, j) =
          sum.at<int32_t>(d, j - 1) +
          image_L.at<uint8_t>(d, j) * image_R.at<uint8_t>(0, j);
    for (i = d + 1; i < Row; i++) {
      sum.at<int32_t>(i, 0) =
          sum.at<int32_t>(i - 1, 0) +
          image_L.at<uint8_t>(i, 0) * image_R.at<uint8_t>(i - d, 0);
      for (j = 1; j < Col; j++)
        sum.at<int32_t>(i, j) =
            sum.at<int32_t>(i - 1, j) + sum.at<int32_t>(i, j - 1) -
            sum.at<int32_t>(i - 1, j - 1) +
            image_L.at<uint8_t>(i, j) * image_R.at<uint8_t>(i - d, j);
    }

    for (i = d; i < Row; i++)
      for (j = 0; j < Col; j++) {
        //左图区块
        xMax_L = (i + rad) < Row ? (i + rad) : Row;
        xMin_L = (i - rad) > 0 ? (i - rad) : 0;
        yMax_L = (j + rad) < Col ? (j + rad) : Col;
        yMin_L = (j - rad) > 0 ? (j - rad) : 0;
        //右图区块
        xMax_R = (i - d + rad) < Row ? (i - d + rad) : Row;
        xMin_R = (i - d - rad) > 0 ? (i - d - rad) : 0;
        yMax_R = (j + rad) < Col ? (j + rad) : Col;
        yMin_R = (j - rad) > 0 ? (j - rad) : 0;
        //像素个数
        pixels = (2 * rad + 1) * (2 * rad + 1);
        //左图像素强度和标准差
        I_L = sum_L.at<int32_t>(xMax_L, yMax_L);
        var_L = sqr_sum_L.at<int32_t>(xMax_L, yMax_L);
        if (xMin_L != 0) {
          I_L -= sum_L.at<int32_t>(xMin_L - 1, yMax_L);
          var_L -= sqr_sum_L.at<int32_t>(xMin_L - 1, yMax_L);
        }
        if (yMin_L != 0) {
          I_L -= sum_L.at<int32_t>(xMax_L, yMin_L - 1);
          var_L -= sqr_sum_L.at<int32_t>(xMax_L, yMin_L - 1);
        }
        if (xMin_L != 0 && yMin_L != 0) {
          I_L += sum_L.at<int32_t>(xMin_L - 1, yMin_L - 1);
          var_L += sqr_sum_L.at<int32_t>(xMin_L - 1, yMin_L - 1);
        }
        var_L = sqrt(var_L - I_L * I_L / (float)pixels);
        I_L /= (float)pixels;
        //右图像素强度和标准差
        I_R = sum_R.at<int32_t>(xMax_R, yMax_R);
        var_R = sqr_sum_R.at<int32_t>(xMax_R, yMax_R);
        if (xMin_R != 0) {
          I_R -= sum_R.at<int32_t>(xMin_R - 1, yMax_R);
          var_R -= sqr_sum_R.at<int32_t>(xMin_R - 1, yMax_R);
        }
        if (yMin_R != 0) {
          I_R -= sum_R.at<int32_t>(xMax_R, yMin_R - 1);
          var_R -= sqr_sum_R.at<int32_t>(xMax_R, yMin_R - 1);
        }
        if (xMin_R != 0 && yMin_R != 0) {
          I_R += sum_R.at<int32_t>(xMin_R - 1, yMin_R - 1);
          var_R += sqr_sum_R.at<int32_t>(xMin_R - 1, yMin_R - 1);
        }
        var_R = sqrt(var_R - I_R * I_R / (float)pixels);
        I_R /= (float)pixels;
        //左右图对比
        sum_LR = sum.at<int32_t>(xMax_L, yMax_L);
        if (xMin_L != 0) sum_LR -= sum.at<int32_t>(xMin_L - 1, yMax_L);
        if (yMin_L != 0) sum_LR -= sum.at<int32_t>(xMax_L, yMin_L - 1);
        if (xMin_L != 0 && yMin_L != 0)
          sum_LR += sum.at<int32_t>(xMin_L - 1, yMin_L - 1);

        cost_L.at<float>(i, j, d) =
            (sum_LR - (float)pixels * I_L * I_R) / (var_L * var_R);
        cost_R.at<float>(i - d, j, d) =
            (sum_LR - (float)pixels * I_L * I_R) / (var_L * var_R);
      }
  }
}