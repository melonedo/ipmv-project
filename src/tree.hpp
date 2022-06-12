#pragma once

#include <opencv2/opencv.hpp>

// 为0代表向右的边，为1代表向下的边
#define DIRECTION_MASK 1 << 23

template <typename F>
void dfs(const cv::Mat tree, F &&f) {
    
}