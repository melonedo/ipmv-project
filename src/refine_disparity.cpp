#include "pipeline.hpp"
#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define rad 1
#define threshold 1
#define tau 2
using namespace cv;

void refine_disparity(const Mat& disp_l, const Mat& disp_r, const Mat& cost,
                      Mat& disp_out) {
  disp_out.setTo(Scalar::all(0));
  size_t Row = disp_l.size[1];
  size_t Col = disp_l.size[2];
  for (int x = 1 + rad; x < Row - 1 - rad; x++) {
    for (int y = 1 + rad; y < Col - 1 - rad; y++) {
      unsigned int dl = disp_l.at<uint8_t>(x, y);
      unsigned int dr = disp_l.at<uint8_t>(x - dl, y);
      if (abs(dl - dr) > threshold) {  
        disp_out.at<uint8_t>(x, y) = 0;
      } 
      else {
        float cl = cost.at<float>((dl - 1 + tau), x, y);
        float cr = cost.at<float>((dl + 1 + tau), x, y);
        float cm = cost.at<float>((dl + tau), x, y);
        if ((cm>cl)&&(cm>cr)) {
          float d = (cl - cr) / (2 * cl + 2 * cr - 4 * cm) + dl;
          disp_out.at<uint8_t>(x, y) = d;
        }
      }
    }
  }
}

// 1.int dl = disp_l.at<uint8_t>(x-dl, y);减去dl不会溢出吗？