#include "pipeline.hpp"
#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#define rad 1

void choose_disparity(const Mat& cost, Mat& disp) {

  size_t Row = cost.size[1];
  size_t Col = cost.size[2];

  float cost_value = 0;
  float max_value = 0;
  unsigned int disp_value = 0;
  int dmin = 0;
  int dmax = 142;
  int x, y, d;
  
  for (int x = 1 + rad; x < Row - 1 - rad; x++) {
    for (int y = 1 + rad; y < Col - 1 - rad; y++) {
      for (int k = dmin; k < dmax; k++) {
        cost_value = cost.at<float>(k, x, y);
        if ((cost_value > max_value)) {
          max_value = cost_value;
          disp_value = k;
        }
      }
      disp.at<float>(x, y) = disp_value;
    }
  }


  /*disp.setTo(Scalar::all(0));*/
}

/*1.dmax,dmin?
   2.  1 + rad  meaning?
   3.   º”ÀŸ£ø
*/