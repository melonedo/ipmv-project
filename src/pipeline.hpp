#pragma once

#include <opencv2/opencv.hpp>

void compute_cost(const cv::Mat& image_L, const cv::Mat& image_R,
                  cv::Mat& cost_L, cv::Mat& cost_R);

void aggregate_cost(const cv::Mat& image, const cv::Mat& cost_in,
                    cv::Mat& cost_out_l, cv::Mat& cost_out_r);

void choose_disparity(const cv::Mat& cost, cv::Mat& disp);

void refine_disparity(const cv::Mat& disp_l, const cv::Mat& disp_r,
                      const cv::Mat& cost, cv::Mat& disp_out);

void bilateral_filter(const cv::Mat& image, const cv::Mat& cost_in,
                      cv::Mat& cost_out_l, cv::Mat& cost_out_r);
