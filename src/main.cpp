#include <omp.h>

#include <Eigen/dense>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#include "pipeline.hpp"
#include "util.hpp"

using namespace cv;

int main(int argc, const char *argv[]) {
  std::string testset = argc >= 2 ? argv[1] : "data/artroom1";

  Calib calib = read_calib(testset + "/calib.txt");
  // calib.ndisp = 8;  // 方便调试
  Mat image_l = imread(testset + "/im0.png");
  Mat image_r = imread(testset + "/im1.png");
  // cv::resize(image_l, image_l, {960, 540});
  // cv::resize(image_r, image_r, {960, 540});
  // calib.height /= 2;
  // calib.width /= 2;

  std::vector<int> shape2{calib.height, calib.width};
  std::vector<int> shape3{calib.ndisp, calib.height, calib.width};

  Mat cost_l{shape3, CV_32FC1, {0.f}};
  Mat cost_r{shape3, CV_32FC1, {0.f}};

  Mat cost_out_l{shape3, CV_32FC1};
  Mat cost_out_r{shape3, CV_32FC1};

  Mat disp_l{shape2, CV_8UC1};
  Mat disp_r{shape2, CV_8UC1};

  Mat disp_out{shape2, CV_32FC1};

  using namespace std::chrono;
  high_resolution_clock::time_point t1, t2;

  t1 = high_resolution_clock::now();

  compute_cost(image_l, image_r, cost_l, cost_r);

  segment_tree(image_l, image_r, cost_l, cost_r, cost_out_l, cost_out_r);
  // bilateral_filter(image_l, cost_l, cost_out_l, cost_out_r);

  choose_disparity(cost_out_l, disp_l);
  choose_disparity(cost_out_r, disp_r);

  refine_disparity(disp_l, disp_r, cost_out_l, disp_out);

  t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "Time: " << time_span.count() * 1000 << "ms" << std::endl;

  imshow("result", disp_out / calib.vmax);

  PFM truth = read_pfm(testset + "/disp0.pfm");
  imshow("truth", truth.data / calib.vmax);

  waitKey();
  return 0;
}
