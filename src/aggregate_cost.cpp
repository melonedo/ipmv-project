#include "pipeline.hpp"

using namespace cv;

void aggregate_cost(const Mat& cost_in, Mat& cost_out) {
  cost_out = cost_in;
}