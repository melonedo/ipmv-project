#include "util.hpp"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <string>

Calib read_calib(const std::string& path) {
  Calib ret;
  FILE* f = fopen(path.c_str(), "r");
  assert(f);
  int n;
  n = fscanf(f, "cam0=[%f %f %f; %f %f %f; %f %f %f] ", &ret.cam0[0],
             &ret.cam0[1], &ret.cam0[2], &ret.cam0[3], &ret.cam0[4],
             &ret.cam0[5], &ret.cam0[6], &ret.cam0[7], &ret.cam0[8]);
  assert(n == 9);
  n = fscanf(f, "cam1=[%f %f %f; %f %f %f; %f %f %f] ", &ret.cam1[0],
             &ret.cam1[1], &ret.cam1[2], &ret.cam1[3], &ret.cam1[4],
             &ret.cam1[5], &ret.cam1[6], &ret.cam1[7], &ret.cam1[8]);
  assert(n == 9);
  n = fscanf(f, "doffs=%f baseline=%f ", &ret.doffs, &ret.baseline);
  assert(n == 2);
  n = fscanf(f, "width=%d height=%d ndisp=%d vmin=%d vmax=%d ", &ret.width,
             &ret.height, &ret.ndisp, &ret.vmin, &ret.vmax);
  assert(n == 5);
  fclose(f);
  return ret;
}

// http://netpbm.sourceforge.net/doc/pfm.html
PFM read_pfm(const std::string& path) {
  PFM ret;
  FILE* f = fopen(path.c_str(), "rb");
  assert(f);
  int n;
  n = fscanf(f, "Pf %d %d %f", &ret.width, &ret.height, &ret.scale);
  assert(n == 3);
  // only handle monochrome case
  ret.color = false;
  assert(ret.scale < 0);
  ret.scale = fabs(ret.scale);
  ret.data = cv::Mat(ret.height, ret.width, CV_32FC1);
  // discard next char
  fgetc(f);
  int count = fread(ret.data.data, sizeof(float), ret.width * ret.height, f);
  assert(count == ret.width * ret.height);
  // assert(feof(f));
  assert(!ferror(f));
  cv::flip(ret.data, ret.data, 0);
  fclose(f);
  return ret;
}
