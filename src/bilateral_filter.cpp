#include <math.h>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"

using namespace cv;
using namespace std;

#define rad 1
#define aggsize 10
#define sigma1 1.75
#define sigma2 3.5

using namespace cv;

// #define SHOW_AGGREGATION

void bilateral_filter(const Mat& image, const Mat& cost_in, Mat& cost_out_l,
                      Mat& cost_out_r) {
  size_t Row = cost_in.size[1];
  size_t Col = cost_in.size[2];
  size_t MaxDistance = cost_in.size[0];
  // std::cout << Row << ' ' << Col << ' ' << MaxDistance << endl;
  // cost_out = cost_in;

  Mat gray;
  cvtColor(image, gray, CV_BGR2GRAY);

  float coefficient1[2 * aggsize + 1][2 * aggsize + 1], coefficient2[256];

  for (int x = -aggsize; x <= aggsize; x++)
    for (int y = -aggsize; y <= aggsize; y++)
      coefficient1[x + aggsize][y + aggsize] =
          exp(-sqrt((float)x * (float)x + (float)y * (float)y) /
              (2 * sigma1 * sigma1));
  for (int x = 0; x < 256; x++)
    coefficient2[x] = exp(-abs((float)x) / (2 * sigma2 * sigma2));

#ifndef SHOW_AGGREGATION
#pragma omp parallel for
#endif
  for (int d = 0; d < MaxDistance; d++) {
#ifdef SHOW_AGGREGATION
    Mat temp(Row, Col, CV_32FC1, {0.f});
#endif
    for (int x = aggsize; x < Row - aggsize - 1; x++) {
      for (int y = std::max(aggsize, d); y < Col - aggsize - 1; y++) {
        float sum1, sum2;
        sum1 = sum2 = 0;
        int intensity_center = gray.at<uint8_t>(x, y);
        for (int p = -aggsize; p <= aggsize; p++) {
          for (int q = -aggsize; q <= aggsize; q++) {
            float cost_neighbour = cost_in.at<float>(d, x + p, y + q);
            float weight = coefficient2[abs(intensity_center -
                                            gray.at<uint8_t>(x + p, y + q))] *
                           coefficient1[q + aggsize][p + aggsize];
            sum1 += weight;
            sum2 += weight * cost_neighbour;
          }
        }
        cost_out_l.at<float>(d, x, y) = sum2 / sum1;
        cost_out_r.at<float>(d, x, y - d) = sum2 / sum1;
#ifdef SHOW_AGGREGATION
        temp.at<float>(x, y) = cost_out_l.at<float>(d, x, y);
#endif
      }
    }
#ifdef SHOW_AGGREGATION
    putText(temp, "d="s + std::to_string(d), {0, 150}, FONT_HERSHEY_SIMPLEX, 3,
            Scalar{1}, 5, 8, false);
    imshow("test", temp);
    waitKey(1);
#endif
  }
}