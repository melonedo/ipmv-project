#include "pipeline.hpp"
#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#define rad 1

void choose_disparity(const Mat& cost_L, const Mat& cost_R, Mat& disp_L,Mat& disp_R) {

  size_t Row = cost_L.size[1];
  size_t Col = cost_L.size[2];

  float cost_value_l = 0;
  float cost_value_r = 0;
  float max_value_l = 0;
  float max_value_r = 0;
  unsigned int disp_value_l = 0;
  unsigned int disp_value_r = 0;
  int dmin = 0;
  int dmax = 142;
  int x, y, d;
  
  for (int x = 1 + rad; x < Row - 1 - rad; x++) {
    for (int y = 1 + rad; y < Col - 1 - rad; y++) {
      for (int k = dmin; k < dmax; k++) {
        cost_value_l = cost_L.at<float>(k, x, y);
        cost_value_r = cost_R.at<float>(k, x, y);
        if ((cost_value_l > max_value_l)) {
          max_value_l = cost_value_l;
          disp_value_l = k;
        }
        if ((cost_value_r > max_value_r)) {
          max_value_r = cost_value_r;
          disp_value_r = k;
        }
      }
      disp_L.at<float>(x, y) = disp_value_l;
      disp_R.at<float>(x, y) = disp_value_r;
    }
  }


  /*disp.setTo(Scalar::all(0));*/
}

/*1.dmax,dmin?
   2.  1 + rad  meaning?
   3.   º”ÀŸ£ø
*/