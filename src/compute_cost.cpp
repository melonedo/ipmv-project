#include "pipeline.hpp"

using namespace cv;

void compute_cost(const Mat& image, Mat& cost) {
  cost.setTo(Scalar::all(0));
  cost.at<float>(0, 0, 0) = 1;
}