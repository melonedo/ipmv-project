#include <opencv2/opencv.hpp>
#include <stack>

#include "pipeline.hpp"

using namespace cv;

#define sigma (.5f * 255)

// #define SHOW_AGGREGATION

uint8_t calculate_weight(Vec3b l, Vec3b r);

// cost_rec = current_cost + weighted sum of cost_rec of its direct chidren
void compute_cost_rec(const float* cost_ptr, const Mat_<Vec3b>& image_,
                      const Mat_<uint8_t>& graph_, const float coef[256],
                      Mat_<float>& cost_rec_) {
  const size_t Col = image_.size[1];
  struct State {
    float cost_rec;
    uint32_t num : 3;
    uint32_t pos : 29;
    Vec3b i0;
    uint8_t node;
  };
  std::stack<State> stack;
  stack.push({cost_ptr[GRAPH_ROOT], 0, GRAPH_ROOT, image_(GRAPH_ROOT),
              graph_(GRAPH_ROOT)});
  float ret = NAN;
  while (!stack.empty()) {
    State& top = stack.top();
    switch (top.num) {
      case 0:
        if (top.node & (1 << 0) &&
            ((top.node >> 4) != 0 || top.pos == GRAPH_ROOT)) {
          uint32_t i = top.pos + 1;
          stack.push({cost_ptr[i], 0, i, image_(i), graph_(i)});
          top.num = 1;
          break;
        }
        goto L1;
      case 1:
        top.cost_rec +=
            ret * coef[calculate_weight(top.i0, image_(top.pos + 1))];
      L1:
        if (top.node & (1 << 1) && (top.node >> 4) != 1) {
          uint32_t i = top.pos + Col;
          stack.push({cost_ptr[i], 0, i, image_(i), graph_(i)});
          top.num = 2;
          break;
        }
        goto L2;
      case 2:
        top.cost_rec +=
            ret * coef[calculate_weight(top.i0, image_(top.pos + Col))];
      L2:
        if (top.node & (1 << 2) && (top.node >> 4) != 2) {
          uint32_t i = top.pos - 1;
          stack.push({cost_ptr[i], 0, i, image_(i), graph_(i)});
          top.num = 3;
          break;
        }
        goto L3;
      case 3:
        top.cost_rec +=
            ret * coef[calculate_weight(top.i0, image_(top.pos - 1))];
      L3:
        if (top.node & (1 << 3) && (top.node >> 4) != 3) {
          uint32_t i = top.pos - Col;
          stack.push({cost_ptr[i], 0, i, image_(i), graph_(i)});
          top.num = 4;
          break;
        }
        goto L4;
      case 4:
        top.cost_rec +=
            ret * coef[calculate_weight(top.i0, image_(top.pos - Col))];
      L4:
        cost_rec_(top.pos) = ret = top.cost_rec;
        stack.pop();
        break;

      default:
        assert(0 && "Unreachable");
    }
  }
  /* version with native stack
  Vec3b i0 = image_(root);
  uint8_t node = graph_(root);
  float cost_rec = cost_ptr[root];
  if (node & (1 << 0)) {
    uint32_t i = root + 1;
    cost_rec += compute_cost_rec(cost_ptr, image_, graph_, coef, i) *
                coef[calculate_weight(i0, image_(i))];
  }
  if (node & (1 << 1)) {...}
  if (node & (1 << 2)) {...}
  if (node & (1 << 3)) {...}
  return cost_rec;
  */
}

// cost_d(p) = S * cost_d(parent) + (1 - S^2) * cost_rec(p)
// S = exp(-edge_weight / sigma)
void compute_cost_d(const float* cost_in, const Mat_<Vec3b>& image_,
                    const Mat_<uint8_t>& graph_, const Mat_<float>& cost_rec_,
                    const float coef[256], float* cost_out) {
  const size_t Col = image_.size[1];
  std::stack<uint32_t> stack;
  stack.push(GRAPH_ROOT);
  while (!stack.empty()) {
    uint32_t top = stack.top();
    uint8_t node = graph_(top);
    stack.pop();
    if (node & (1 << 0) && ((node >> 4) != 0 || top == GRAPH_ROOT)) {
      uint32_t i = top + 1;
      stack.push(i);
    }
    if (node & (1 << 1) && (node >> 4) != 1) {
      uint32_t i = top + Col;
      stack.push(i);
    }
    if (node & (1 << 2) && (node >> 4) != 2) {
      uint32_t i = top - 1;
      stack.push(i);
    }
    if (node & (1 << 3) && (node >> 4) != 3) {
      uint32_t i = top - Col;
      stack.push(i);
    }

    if (top == GRAPH_ROOT) {
      cost_out[top] = cost_rec_(top);
    } else {
      uint32_t parent;
      uint8_t dir = node >> 4;
      switch (node >> 4) {
        case 0:
          parent = top + 1;
          break;
        case 1:
          parent = top + Col;
          break;
        case 2:
          parent = top - 1;
          break;
        case 3:
          parent = top - Col;
          break;
      }
      float s = coef[calculate_weight(image_(top), image_(parent))];
      cost_out[top] = s * cost_out[parent] + (1 - s * s) * cost_rec_(top);
    }
  }
}

void aggregate_cost(const Mat& cost_in, const Mat& image, const Mat& graph,
                    Mat& cost_out) {
  const size_t MaxDistance = cost_in.size[0];
  const size_t Row = cost_in.size[1];
  const size_t Col = cost_in.size[2];

  assert(image.type() == CV_8UC3);
  const Mat_<Vec3b>& image_ = reinterpret_cast<const Mat_<Vec3b>&>(image);
  assert(graph.type() == CV_8UC1);
  const Mat_<uint8_t>& graph_ = reinterpret_cast<const Mat_<uint8_t>&>(graph);

  float coef[256];
  for (int i = 0; i < 256; i++) {
    coef[i] = std::exp(-i / sigma);
  }

#ifndef SHOW_AGGREGATION
#pragma omp parallel for
#endif
  for (int d = 0; d < MaxDistance; d++) {
    const float* cost_in_ptr = cost_in.ptr<float>(d);
    float* cost_out_ptr = cost_out.ptr<float>(d);
    Mat_<float> cost_rec_(Row, Col);
    compute_cost_rec(cost_in_ptr, image_, graph_, coef, cost_rec_);
    compute_cost_d(cost_in_ptr, image_, graph_, cost_rec_, coef, cost_out_ptr);

#ifdef SHOW_AGGREGATION
    namedWindow("test", WINDOW_NORMAL);
    Mat temp = Mat(Row, Col, CV_32FC1, cost_out_ptr) / 1000;
    using namespace std::string_literals;
    putText(temp, "d="s + std::to_string(d), {0, 150}, FONT_HERSHEY_SIMPLEX, 3,
            Scalar{1}, 5, 8, false);
    imshow("test", temp);
    waitKey(1);
#endif
  }
}

void segment_tree(const cv::Mat& image_l, const cv::Mat& image_r,
                  const cv::Mat& cost_in_l, const cv::Mat& cost_in_r,
                  cv::Mat& cost_out_l, cv::Mat& cost_out_r) {
  Mat reference_l(image_l.rows, image_l.cols, CV_8UC1, Scalar{0});
  Mat reference_r(image_l.rows, image_l.cols, CV_8UC1, Scalar{0});
  construct_tree(image_l, reference_l);
  construct_tree(image_r, reference_r);
  aggregate_cost(cost_in_l, image_l, reference_l, cost_out_l);
  aggregate_cost(cost_in_r, image_r, reference_r, cost_out_r);
}
