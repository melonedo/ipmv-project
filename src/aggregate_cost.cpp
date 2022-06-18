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

void aggregate_cost(const Mat& image, const Mat& cost_in, Mat& cost_out_l, Mat& cost_out_r) {
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


#pragma omp parallel for
  for (int d = 0; d < MaxDistance; d++) {
    // Mat temp(Row, Col, CV_32FC1, {0.f});
    for (int x = aggsize; x < Row - aggsize - 1; x++) {
      for (int y = std::max(aggsize, d); y < Col - aggsize - 1; y++) {
        float sum1, sum2;
        sum1 = sum2 = 0;
        int intensity_center = gray.at<uint8_t>(x, y);
        for (int p = -aggsize; p <= aggsize; p++) {
          for (int q = -aggsize; q <= aggsize; q++) {
            float cost_neighbour = cost_in.at<float>(d, x + p, y + q);
            float weight =
                coefficient2[abs(intensity_center -
                                      gray.at<uint8_t>(x + p, y + q))] *
                coefficient1[q + aggsize][p + aggsize];
            sum1 += weight;
            sum2 += weight * cost_neighbour;
          }
        }
        cost_out_l.at<float>(d, x, y) = sum2 / sum1;
        cost_out_r.at<float>(d, x, y - d) = sum2 / sum1;
        // temp.at<float>(x, y) = cost_out.at<float>(d, x, y);
      }
    }
    // putText(temp, "d="s + std::to_string(d), {0, 150}, FONT_HERSHEY_SIMPLEX, 3,
    //         Scalar{1}, 5, 8, false);
    // imshow("test", temp);
    // waitKey(1);
  }
}