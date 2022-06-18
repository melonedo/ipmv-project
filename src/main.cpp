#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

#include "pipeline.hpp"

using namespace cv;

int main(int argc, const char* argv[]) {
  std::string testset = argc >= 2 ? argv[1] : "data/artroom2";

  for (const auto& entry : std::filesystem::directory_iterator("data")) {
    if (!entry.is_directory()) continue;
    run_testset(entry.path().string(), USE_SEGMENT_TREE);
  }

  return 0;
}
