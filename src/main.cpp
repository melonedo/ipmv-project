#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

#include "pipeline.hpp"

using namespace cv;

int main(int argc, const char* argv[]) {
  std::string testset = argc >= 2 ? argv[1] : "data/artroom2";
  std::vector<TestResult> results;

  run_testset("build/ukulele2", USE_SEGMENT_TREE, false);
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
