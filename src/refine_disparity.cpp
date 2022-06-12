#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"

#define threshold 3000
#define tau 2
using namespace cv;

void refine_disparity(const Mat& disp_l, const Mat& disp_r, const Mat& cost,
                      Mat& disp_out) {
  const size_t MaxDistance = cost.size[0];
  const size_t Row = disp_l.size[0];
  const size_t Col = disp_l.size[1];
#pragma omp parallel for
  for (int x = 1 + RAD; x < Row - 1 - RAD; x++) {
    for (int y = 1 + RAD; y < Col - 1 - RAD; y++) {
      unsigned int dl = disp_l.at<uint8_t>(x, y);
      unsigned int dr = disp_r.at<uint8_t>(x, y - dl);
      if (abs(dl - dr) > threshold) {
        disp_out.at<float>(x, y) = 0;
      } else if (dl >= 1 + tau && dl + 1 + tau < MaxDistance) {
        float cl = cost.at<float>((dl - 1 + tau), x, y);
        float cr = cost.at<float>((dl + 1 + tau), x, y);
        float cm = cost.at<float>((dl + tau), x, y);
        if ((cm > cl) && (cm > cr)) {
          float d = (cl - cr) / (2 * cl + 2 * cr - 4 * cm) + dl;
          disp_out.at<float>(x, y) = d;
        }
      }
    }
  }
  // imshow("disp1", disp_out);
  // waitKey(1);
}

// 1.int dl = disp_l.at<uint8_t>(x-dl, y);减去dl不会溢出吗？