#pragma once

#include <opencv2/opencv.hpp>

#define RAD 1

void compute_cost(const cv::Mat& image_L, const cv::Mat& image_R,
                  cv::Mat& cost_L, cv::Mat& cost_R);

void aggregate_cost(const cv::Mat& cost_in, cv::Mat& cost_out);

void choose_disparity(const cv::Mat& cost, cv::Mat& disp);

void refine_disparity(const cv::Mat& disp_l, const cv::Mat& disp_r,
                      const cv::Mat& cost, cv::Mat& disp_out);
void stereo_rectification(const cv::Mat& img_L, const cv::Mat& img_R,
                          const cv::Mat& KL, const cv::Mat& KR,
                          const cv::Mat& DL, const cv::Mat& DR,
                          const cv::Mat& R, const cv::Mat& T,
                          Mat& image_l_rected, Mat& image_r_rected);
