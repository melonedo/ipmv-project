#include "pipeline.hpp"
#include <math.h>
#define Row 2000
#define Col 1100

using namespace cv;

void compute_cost(const Mat& image_L, const Mat& image_R, Mat& cost_L, Mat& cost_R) {
  cost.setTo(Scalar::all(0));

  int sum[Row][Col];
  int sum_L[Row][Col], sum_R[Row][Col];
  int sqr_sum_L[Row][Col], sqr_sum_R[Row][Col];
  int pixels;
  float SumLeftRight;
  float PixelIntensityLeft, PixelIntensityRight;
  float StandardDeviationLeft, StandardDeviationRight;
  int xMax_L, xMin_L, yMax_L, yMin_L;
  int xMax_R, xMin_R, yMax_R, yMin_R;
  int i, j, d;

  //左图灰度前缀和
  sum_L[0][0] = image_L.ptr<uchar>(0)[0];
  for (j = 1; j < image_L.col; j++)
    sum_L[0][j] = sum_L[0][j - 1] + image_L.ptr<uchar>(0)[j];
  for (i = 1; i < image_L.row; i++) {
    sum_L[i][0] = sum_L[i - 1][0] + image_L.ptr<uchar>(i)[0];
    for (j = 1; j < image_L.col; j++)
      sum_L[i][j] = sum_L[i][j - 1] + sum_L[i - 1][j] - sum_L[i - 1][j - 1] +
                    image_L.ptr<uchar>(i)[j];
  }

  //右图灰度前缀和
  sum_R[0][0] = image_R.ptr<uchar>(0)[0];
  for (j = 1; j < image_R.col; j++)
    sum_R[0][j] = sum_R[0][j - 1] + image_R.ptr<uchar>(0)[j];
  for (i = 1; i < image_R.row; i++) {
    sum_R[i][0] = sum_R[i - 1][0] + image_R.ptr<uchar>(i)[0];
    for (j = 1; j < image_R.col; j++)
      sum_R[i][j] = sum_R[i - 1][j] + sum_R[i][j - 1] - sum_R[i - 1][j - 1] +
                    image_R.ptr<uchar>(i)[j];
  }

  //左图灰度平方前缀和
  sqr_sum_L[0][0] = image_L.ptr<uchar>(0)[0] * image_L.ptr<uchar>(0)[0];
  for (j = 1; j < image_L.col; j++)
    sqr_sum_L[0][j] = sqr_sum_L[0][j - 1] +
                      image_L.ptr<uchar>(0)[j] * image_L.ptr<uchar>(0)[j];
  for (i = 1; i < image_L.row; i++) {
    sqr_sum_L[i][0] = sqr_sum_L[i - 1][0] +
                      image_L.ptr<uchar>(i)[0] * image_L.ptr<uchar>(i)[0];
    for (j = 1; j < image_L.col; j++)
      sqr_sum_L[i][j] = sqr_sum_L[i][j - 1] + sqr_sum_L[i - 1][j] -
                        sqr_sum_L[i - 1][j - 1] +
                        image_L.ptr<uchar>(i)[j] * image_L.ptr<uchar>(i)[j];
  }

  //右图灰度平方前缀和
  sqr_sum_R[0][0] = image_R.ptr<uchar>(0)[0] * image_R.ptr<uchar>(0)[0];
  for (j = 1; j < image_R.col; j++)
    sqr_sum_R[0][j] = sqr_sum_R[0][j - 1] +
                      image_R.ptr<uchar>(0)[j] * image_R.ptr<uchar>(0)[j];
  for (i = 1; i < image_R.row; i++) {
    sqr_sum_R[i][0] = sqr_sum_R[i - 1][0] +
                      image_R.ptr<uchar>(i)[0] * image_R.ptr<uchar>(i)[0];
    for (j = 1; j < image_R.col; j++)
      sqr_sum_R[i][j] = sqr_sum_R[i][j - 1] + sqr_sum_R[i - 1][j] -
                        sqr_sum_R[i - 1][j - 1] +
                        image_R.ptr<uchar>(i)[j] * image_R.ptr<uchar>(i)[j];
  }

  for (d = 0; d <= MaxDistance; d++) {
    //分子前缀和
    sum[d][0] = image_L.ptr<uchar>(d)[0] * image_R.ptr<uchar>(0)[0];
    for (j = 1; j < image_L.col; j++)
      sum[d][j] =
          sum[d][j - 1] + image_L.ptr<uchar>(d)[j] * image_R.ptr<uchar>(0)[j];
    for (i = d + 1; i < image_L.row; i++) {
      sum[i][0] = sum[i - 1][0] +
                  image_L.ptr<uchar>(i)[0] * image_R.ptr<uchar>(i - d)[0];
      for (j = 1; j < image_L.col; j++)
        sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] +
                    image_L.ptr<uchar>(i)[j] * image_R.ptr<uchar>(i - d)[j];
    }

    for (i = d; i < image_L.row; i++)
      for (j = 0; j < image_L.col; j++) {
		//左图区块
        xMax_L = (i + rad) < image_L.row ? (i + rad) : image_L.row;
        xMin_L = (i - rad) > 0 ? (i - rad) : 0;
        yMax_L = (j + rad) < image_L.col ? (j + rad) : image_L.col;
        yMin_L = (j - rad) > 0 ? (j - rad) : 0;
		//右图区块
        xMax_R = (i - d + rad) < image_R.row ? (i - d + rad) : image_R.row;
        xMin_R = (i - d - rad) > 0 ? (i - d - rad) : 0;
        yMax_R = (j + rad) < image_R.col ? (j + rad) : image_R.col;
        yMin_R = (j - rad) > 0 ? (j - rad) : 0;
		//像素个数
        pixels = (2 * rad + 1) * (2 * rad + 1);
		//左图像素强度和标准差
        PixelIntensityLeft = sum_L[xMax_L][yMax_L];
        StandardDeviationLeft = sqr_sum_L[xMax_L][yMax_L];
        if (xMin_L != 0) {
          PixelIntensityLeft -= sum_L[xMin_L - 1][yMax_L];
          StandardDeviationLeft -= sqr_sum_L[xMin_L - 1][yMax_L];
        }
        if (yMin_L != 0) {
          PixelIntensityLeft -= sum_L[xMax_L][yMin_L - 1];
          StandardDeviationLeft -= sqr_sum_L[xMax_L][yMin_L - 1];
        }
        if (xMin_L != 0 && yMin_L != 0) {
          PixelIntensityLeft += sum_L[xMin_L - 1][yMin_L - 1];
          StandardDeviationLeft += sqr_sum_L[xMin_L - 1][yMin_L - 1];
        }
        StandardDeviationLeft =
            sqrt(StandardDeviationLeft -
                 PixelIntensityLeft * PixelIntensityLeft / (float) pixels);
        PixelIntensityLeft /= (float)pixels;
        //右图像素强度和标准差
        PixelIntensityRight = sum_R[xMax_R][yMax_R];
        StandardDeviationRight = sqr_sum_R[xMax_R][yMax_R];
        if (xMin_R != 0) {
          PixelIntensityRight -= sum_R[xMin_R - 1][yMax_R];
          StandardDeviationRight -= sqr_sum_R[xMin_R - 1][yMax_R];
        }
        if (yMin_R != 0) {
          PixelIntensityRight -= sum_R[xMax_R][yMin_R - 1];
          StandardDeviationRight -= sqr_sum_R[xMax_R][yMin_R - 1];
        }
        if (xMin_R != 0 && yMin_R != 0) {
          PixelIntensityRight += sum_R[xMin_R - 1][yMin_R - 1];
          StandardDeviationRight += sqr_sum_R[xMin_R - 1][yMin_R - 1];
        }
        StandardDeviationRight =
            sqrt(StandardDeviationRight -
                 PixelIntensityRight * PixelIntensityRight / (float)pixels);
        PixelIntensityRight /= (float)pixels;
		//左右图对比
		SumLeftRight = sum[xMax_L][yMax_L];
        if (xMin_L != 0) SumLeftRight -= sum[xMin_L - 1][yMax_L];
        if (yMin_L != 0) SumLeftRight -= sum[xMax_L][yMin_L - 1];
        if (xMin_L != 0 && yMin_L != 0) SumLeftRight += sum[xMin_L - 1][yMin_L - 1];
		
        cost_L.at<float>(i, j, d) = (SumLeftRight -
             (float)pixels * PixelIntensityLeft * PixelIntensityRight) /
            (StandardDeviationLeft * StandardDeviationRight);
        cost_R.at<float>(i - d, j, d) = (SumLeftRight -
			 (float)pixels * PixelIntensityLeft * PixelIntensityRight) /
            (StandardDeviationLeft * StandardDeviationRight);
      }
  }
}