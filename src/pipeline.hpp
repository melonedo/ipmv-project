#pragma once

#include <opencv2/opencv.hpp>

// #define NOALIAS __declspec(noalias)
#ifndef NOALIAS
#define NOALIAS
#endif

#define RAD 2

#define GRAPH_ROOT 865100

void compute_cost(const cv::Mat& image_L, const cv::Mat& image_R,
                  cv::Mat& cost_L, cv::Mat& cost_R);

void choose_disparity(const cv::Mat& cost, cv::Mat& disp);

void refine_disparity(const cv::Mat& disp_l, const cv::Mat& disp_r,
                      const cv::Mat& cost, cv::Mat& disp_out);

void segment_tree(const cv::Mat& image_l, const cv::Mat& image_r,
                  const cv::Mat& cost_in_l, const cv::Mat& cost_in_r,
                  cv::Mat& cost_out_l, cv::Mat& cost_out_r);

void construct_tree(const cv::Mat& image, cv::Mat& graph);

void bilateral_filter(const cv::Mat& image, const cv::Mat& cost_in,
                      cv::Mat& cost_out_l, cv::Mat& cost_out_r);
