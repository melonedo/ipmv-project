#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#include "pipeline.hpp"
#include "util.hpp"

using namespace cv;

TestResult run_testset(const std::string& testset, int method, bool refine,
                       bool rectify_image, bool calibrate) {
  Calib calib = read_calib(testset + "/calib.txt");

  Mat image_l, image_r;

  if (rectify_image) {
    // 相机拍的就是jpg
    image_l = imread(testset + "/im0.jpg");
    image_r = imread(testset + "/im1.jpg");
  } else {
    image_l = imread(testset + "/im0.png");
    image_r = imread(testset + "/im1.png");
  }

  assert(image_l.rows == image_r.rows);
  assert(image_l.cols == image_r.cols);

  // calib.ndisp = 8;  // 方便调试
  //   cv::resize(image_l, image_l, {image_l.rows / 2, image_r.cols / 2});
  //   cv::resize(image_r, image_r, {image_l.rows / 2, image_r.cols / 2});

  calib.height = image_l.rows;
  calib.width = image_l.cols;

  std::vector<int> shape2{calib.height, calib.width};
  std::vector<int> shape3{calib.ndisp, calib.height, calib.width};

  Mat cost_l{shape3, CV_32FC1, {0.f}};
  Mat cost_r{shape3, CV_32FC1, {0.f}};

  Mat cost_out_l{shape3, CV_32FC1};
  Mat cost_out_r{shape3, CV_32FC1};

  Mat disp_l{shape2, CV_16UC1};
  Mat disp_r{shape2, CV_16UC1};

  Mat disp_out{shape2, CV_32FC1, {0.f}};

  Mat R, KL, KR, DL, DR;
  Vec3d T;

  using namespace std::chrono;
  high_resolution_clock::time_point t1, t2;

  t1 = high_resolution_clock::now();

  if (calibrate)
    stereo_calib(image_l, image_r, R, T, KL, KR, DL, DR);
  else
     preset_steroparams(R, T, KL, KR, DL, DR);

  if (rectify_image) {
    Mat image_l_rected{shape2, CV_8UC3};
    Mat image_r_rected{shape2, CV_8UC3};
    stereo_rectification(image_l, image_r, image_l_rected, image_r_rected);
    image_l = image_l_rected;
    image_r = image_r_rected;
    imshow("image_l_rected", image_l);
    imshow("image_r_rected", image_r);
    waitKey();
  }
  compute_cost(image_l, image_r, cost_l, cost_r);

  if (method == USE_BILATERAL_FILTER) {
    bilateral_filter(image_l, cost_l, cost_out_l, cost_out_r);
  } else if (method == USE_SEGMENT_TREE) {
    segment_tree(image_l, image_r, cost_l, cost_r, cost_out_l, cost_out_r,
                 !refine);
  }

  choose_disparity(cost_out_l, disp_l);
  if (refine) {
    choose_disparity(cost_out_r, disp_r);
    refine_disparity(disp_l, disp_r, cost_out_l, disp_out);
  } else {
    disp_l.convertTo(disp_out, CV_32FC1);
  }

  t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  imshow("result", disp_out / calib.vmax);

  waitKey();
  return {0.f, 0.f};
  PFM truth = read_pfm(testset + "/disp0.pfm");
  //   imshow("truth", truth.data / calib.vmax);

  Mat diff, invisible, visible;
  inRange(truth.data, {0}, {1000}, visible);
  bitwise_not(visible, invisible);
  //   imshow("invisible", invisible);
  absdiff(disp_out, truth.data, diff);
  //   imshow("difference", diff / 5);
  inRange(diff, {0}, {5}, diff);
  bitwise_or(invisible, diff, diff);
  //   imshow("inrange", diff);
  float error_percentage =
      100 - countNonZero(diff) * 100.f / calib.height / calib.width;

  std::cout << testset << "\ttime: " << time_span.count() << "s"
            << "\terror pixels: " << error_percentage << "%" << std::endl;

  imwrite(testset + "/result.tiff", disp_out / calib.vmax);
  imwrite(testset + "/truth.tiff", truth.data / calib.vmax);
  //   waitKey(0);
  return {time_span.count(), error_percentage};
}
