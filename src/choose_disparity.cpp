#include <math.h>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"
using namespace cv;

NOALIAS void choose_disparity(const Mat& cost, Mat& disp) {
  const size_t MaxDistance = cost.size[0];
  const size_t Row = cost.size[1];
  const size_t Col = cost.size[2];

#pragma omp parallel for
  for (int x = 1 + RAD; x < Row - 1 - RAD; x++) {
    for (int y = 1 + RAD; y < Col - 1 - RAD; y++) {
      float max_value = 0;
      uint16_t disp_value = 0;
      for (int k = 0; k < MaxDistance; k++) {
        float cost_value = cost.at<float>(k, x, y);
        if ((cost_value > max_value)) {
          max_value = cost_value;
          disp_value = k;
        }      
      }
      disp.at<uint16_t>(x, y) = disp_value;
    }
  }
  /*imshow("disp0", disp);
  waitKey(1);*/
}
