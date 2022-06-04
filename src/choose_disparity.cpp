#include "pipeline.hpp"

using namespace cv;

void choose_disparity(const Mat& cost, Mat& disp) {
  disp.setTo(Scalar::all(0));
}
