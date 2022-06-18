#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"

#define threshold 10

using namespace cv;

void refine_disparity(const Mat& disp_l,
                      const Mat& disp_r, const Mat& cost,
                      Mat& disp_out) {
  const size_t MaxDistance = cost.size[0];  // Maxdsitance=170
  const size_t Row = disp_l.size[0];
  const size_t Col = disp_l.size[1];
  /*imshow("disp_out", disp_out);*/
  /*int n = 0;*/
  /*disp_out.setTo(Scalar::all(170));*/
#pragma omp parallel for
  for (int x = 0; x < Row; x++) {
    for (int y = 0; y < Col; y++) {
      uint16_t dl = disp_l.at<uint16_t>(x, y);
      if (y < dl) {
        disp_out.at<float>(x, y) = 0;
        continue;
      }
      uint16_t dr = disp_r.at<uint16_t>(x, y - dl);
      if (abs(dl - dr) > threshold) {
        disp_out.at<float>(x, y) = 0;
      } else if (dl > 1 && dl < MaxDistance - 1){
        float cl = cost.at<float>((dl - 1), x, y);
        float cr = cost.at<float>((dl + 1), x, y);
        float cm = cost.at<float>((dl), x, y);
        if ((cm > cl) && (cm > cr)) {
          float d = (cl - cr) / (2 * cl + 2 * cr - 4 * cm) + dl;
          disp_out.at<float>(x, y) = d;
        } else {
          disp_out.at<float>(x, y) = cl;
        }
      }
    }
  }
  // imshow("disp1", disp_out);
  // waitKey(1);
}

// 1.int dl = disp_l.at<uint8_t>(x-dl, y);减去dl不会溢出吗？