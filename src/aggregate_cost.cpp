#include <math.h>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"

using namespace cv;
using namespace std;

#define rad 1
#define aggsize 7
#define sigma1 1.75
#define sigma2 3.5

using namespace cv;

void aggregate_cost(const Mat& image, const Mat& cost_in, Mat& cost_out) {
  size_t Row = cost_in.size[0];
  size_t Col = cost_in.size[1];
  size_t MaxDistance = cost_in.size[2];
  cost_out = cost_in;

  Mat gray;
  cvtColor(image, gray, CV_BGR2GRAY);

  float coefficient1[2 * aggsize + 1][2 * aggsize + 1], coefficient2[256];

  int x, y, d;
  int p, q;

  int Dmax;
  float sum1, sum2;
  float cost_center;
  int intensity_center;
  float cost_neighbour;
  float weight;

  for (int x = -aggsize; x <= aggsize; x++)
    for (int y = -aggsize; y <= aggsize; y++) {
      coefficient1[x + aggsize][y + aggsize] =
          exp(-sqrt((float)x * (float)x + (float)y * (float)y) /
              (2 * sigma1 * sigma1));
    }
  for (int x = 0; x < 256; x++)
    coefficient2[x] = exp(-abs((float)x) / (2 * sigma2 * sigma2));

#pragma omp parallel for
  for (x = 2 + rad; x < Row - 2 - rad; x++)
    for (y = 2 + rad; y < Col - 2 - rad; y++) {
      Dmax = ((x - (1 + rad + 1)) < MaxDistance) ? (x - (1 + rad + 1))
                                                 : MaxDistance;
      for (d = 0; d <= Dmax; d++) {
        sum1 = sum2 = 0;
        cost_center = cost_in.at<float>(d, x, y);
        intensity_center = gray.at<uint8_t>(x, y);
        for (p = -aggsize; p <= aggsize; p++)
          for (q = -aggsize; q <= aggsize; q++) {
            cost_neighbour = cost_in.at<float>(d, x + p, y + q);
            weight = coefficient2[(int)abs(intensity_center -
                                           gray.at<uint8_t>(d, x + p, y + q))] *
                     coefficient1[p + aggsize][q + aggsize];
            sum1 += weight;
            sum2 += weight * cost_neighbour;
          }
        cost_out.at<float>(d, x, y) = sum2 / sum1;
      }
    }
}