#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

#include "pipeline.hpp"

using namespace cv;

int main(int argc, const char* argv[]) {
  std::vector<TestResult> results;

  // 只跑代表性的数据
  run_testset("data/artroom1", USE_SEGMENT_TREE, true, false);
  waitKey(0);
  run_testset("data/ukulele2", USE_SEGMENT_TREE, false, true);
  waitKey(0);
  return 0;
  
  for (const auto& entry : std::filesystem::directory_iterator("data")) {
    if (!entry.is_directory()) continue;
    results.push_back(
        run_testset(entry.path().string(), USE_SEGMENT_TREE, false));
  }

  double total_time = 0;
  double total_error = 0;

  for (auto r : results) {
    total_time += r.time;
    total_error += r.percent;
  }

  std::cout << "Average time: " << total_time / results.size()
            << " Average error percentage: " << total_error / results.size()
            << "%" << std::endl;

  return 0;
}
