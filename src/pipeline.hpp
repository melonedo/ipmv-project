#pragma once

#include <opencv2/opencv.hpp>

extern int MaxDistance;
extern int rad;

void compute_cost(const cv::Mat& image, cv::Mat& cost);

void aggregate_cost(const cv::Mat& cost_in, cv::Mat& cost_out);

void choose_disparity(const cv::Mat& cost, cv::Mat& disp);

void refine_disparity(const cv::Mat& disp_l, const cv::Mat& disp_r,
                      const cv::Mat& cost, cv::Mat& disp_out);
