#pragma once

#include <opencv2/opencv.hpp>

// #define NOALIAS __declspec(noalias)
#ifndef NOALIAS
#define NOALIAS
#endif

#define RAD 1

#define GRAPH_ROOT 86510

enum { USE_SEGMENT_TREE, USE_BILATERAL_FILTER };

struct TestResult {
  double time;
  double percent;
};

TestResult run_testset(const std::string& testset, int method,
                       bool refine = true, bool rectify_image = false,
                       bool calibrate = false);

void compute_cost(const cv::Mat& image_L, const cv::Mat& image_R,
                  cv::Mat& cost_L, cv::Mat& cost_R);

void choose_disparity(const cv::Mat& cost, cv::Mat& disp);

void refine_disparity(const cv::Mat& disp_l, const cv::Mat& disp_r,
                      const cv::Mat& cost, cv::Mat& disp_out);

void stereo_rectification(const cv::Mat& img_L, const cv::Mat& img_R,
                          const cv::Mat& R, const cv::Mat& T,
                          const cv::Mat& K_L, const cv::Mat& K_R,
                          const cv::Mat& D1, const cv::Mat& D2,
                          cv::Mat& image_l_rected, cv::Mat& image_r_rected);

void preset_steroparams(cv::Mat& R, cv::Mat& T, cv::Mat& K_L, cv::Mat& K_R,
                        cv::Mat& D1, cv::Mat& D2);

void stereo_calib(const cv::Mat& img_L, const cv::Mat& img_R, cv::Mat& R,
                  cv::Mat& T, cv::Mat& K_L, cv::Mat& K_R, cv::Mat& D1,
                  cv::Mat& D2);

void segment_tree(const cv::Mat& image_l, const cv::Mat& image_r,
                  const cv::Mat& cost_in_l, const cv::Mat& cost_in_r,
                  cv::Mat& cost_out_l, cv::Mat& cost_out_r,
                  bool left_only = false);

void construct_tree(const cv::Mat& image, cv::Mat& graph);

void bilateral_filter(const cv::Mat& image, const cv::Mat& cost_in,
                      cv::Mat& cost_out_l, cv::Mat& cost_out_r);
