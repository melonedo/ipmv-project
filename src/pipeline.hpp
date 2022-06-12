#pragma once

#include <opencv2/opencv.hpp>

#define RAD 2

#define GRAPH_ROOT 1153000

void compute_cost(const cv::Mat& image_L, const cv::Mat& image_R,
                  cv::Mat& cost_L, cv::Mat& cost_R);

void aggregate_cost(const cv::Mat& cost_in, const cv::Mat& image, const cv::Mat& ref, cv::Mat& cost_out);

void choose_disparity(const cv::Mat& cost, cv::Mat& disp);

void refine_disparity(const cv::Mat& disp_l, const cv::Mat& disp_r,
                      const cv::Mat& cost, cv::Mat& disp_out);
//void stereo_rectification(const cv::Mat& img_L, const cv::Mat& img_R,
//                          const cv::Mat& KL, const cv::Mat& KR,
//                          const cv::Mat& DL, const cv::Mat& DR,
//                          const cv::Mat& R, const cv::Mat& T, cv::Mat & image_l_rected, cv::Mat& image_r_rected);

void construct_tree(const cv::Mat& in, cv::Mat& out);
