#pragma once

#include <string>
#include <opencv/cv.hpp>

struct Calib {
  float cam0[9];
  float cam1[9];
  float doffs;
  float baseline;
  int width;
  int height;
  int ndisp;
  int vmin;
  int vmax;
};

Calib read_calib(const std::string &path);

struct PFM {
  bool color;
  int width;
  int height;
  float scale;
  cv::Mat data;
};

PFM read_pfm(const std::string &path);
