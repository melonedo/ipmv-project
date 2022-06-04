#include "pipeline.hpp"

using namespace cv;

void refine_disparity(const Mat& disp_l, const Mat& disp_r, const Mat& cost,
                      Mat& disp_out) {
  disp_out.setTo(Scalar::all(0));
}
