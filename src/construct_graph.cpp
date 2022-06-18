#include <algorithm>
#include <execution>
#include <opencv2/opencv.hpp>
#include <random>
#include <stack>
#include <vector>

#include "pipeline.hpp"

using namespace cv;

// Segment-Tree based Cost Aggregation for Stereo Matching
#define K 1200

struct Edge {
  uint32_t weight : 8;
  uint32_t direction : 1;  // true时偏移量为stride
  uint32_t base : 23;
};

void get_points(Edge edge, uint32_t stride, uint32_t& p1, uint32_t& p2) {
  p1 = edge.base;
  if (edge.direction) {
    p2 = p1 + stride;
  } else {
    p2 = p1 + 1;
  }
}

uint8_t get_edge_weight(Edge edge) { return edge.weight; }

class DisjointSet {
 public:
  DisjointSet(uint32_t size) : parent(size), max_weight(size) {
    for (uint32_t i = 0; i < size; i++) {
      parent[i] = i;
      max_weight[i] = 0;
    }
  }

  uint32_t find(uint32_t i) const {
    if (parent[i] == i) {
      return i;
    } else {
      return parent[i] = find(parent[i]);
    }
  }

  void join(uint32_t i, uint32_t j, uint8_t weight) {
    uint32_t parent_i = find(i);
    uint32_t parent_j = find(j);
    // j所在树成为i所在树的子树
    parent[parent_j] = parent_i;
    max_weight[parent_i] = std::max(max_weight[parent_i], max_weight[parent_j]);
  }

  uint8_t get_max_weight(uint32_t i) const { return max_weight[find(i)]; }

  bool disjoint(uint32_t i, uint32_t j) const { return find(i) != find(j); }

 private:
  mutable std::vector<uint32_t> parent;
  std::vector<uint8_t> max_weight;
};

bool merge_criterion(const DisjointSet& set, uint32_t p1, uint32_t p2,
                     uint8_t weight) {
  float w1 = set.get_max_weight(p1);
  float w2 = set.get_max_weight(p2);
  static_assert(std::numeric_limits<double>::is_iec559, "Expect K/0 == +Inf");
  return weight <= w1 + K / w1 && weight <= w2 + K / w2;
}

uint8_t calculate_weight(Vec3b l, Vec3b r) {
  return std::max(
      {std::abs(l[0] - r[0]), std::abs(l[1] - r[1]), std::abs(l[2] - r[2])});
}

#define sigma (.3f * 255)

void show_tree(const Mat_<Vec3b>& image_, const Mat_<uint8_t>& graph_) {
  float coef[256];
  for (int i = 0; i < 256; i++) {
    coef[i] = std::exp(-i / sigma);
  }
  Mat_<float> temp(image_.size[0], image_.size[1]);
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
      temp(top) = 1;
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
      temp(top) = s * temp(parent);
    }
  }
  namedWindow("test", WINDOW_NORMAL);
  Mat gray, res;
  cvtColor(image_, gray, CV_BGR2GRAY);
  multiply(gray, temp, res, 5, CV_8UC1);
  imshow("test", res);
  waitKey(0);
}

void construct_tree(const cv::Mat& image, cv::Mat& graph) {
  const int Row = image.size[0];
  const int Col = image.size[1];

  // 边
  std::vector<Edge> edges;
  edges.reserve(2 * Row * Col);
  for (unsigned x = 1 + RAD; x < Row - 2 - RAD; x++) {
    for (unsigned y = 1 + RAD; y < Col - 1 - RAD; y++) {
      edges.push_back(
          {calculate_weight(image.at<Vec3b>(x + 1, y), image.at<Vec3b>(x, y)),
           true, x * Col + y});
    }
  }
  for (unsigned x = 1 + RAD; x < Row - 1 - RAD; x++) {
    for (unsigned y = 1 + RAD; y < Col - 2 - RAD; y++) {
      edges.push_back(
          {calculate_weight(image.at<Vec3b>(x, y + 1), image.at<Vec3b>(x, y)),
           false, x * Col + y});
    }
  }

//   std::shuffle(edges.begin(), edges.end(),
//                       std::mt19937{std::random_device{}()});

  std::sort(
      std::execution::par_unseq, edges.begin(), edges.end(),
      [](Edge l, Edge r) { return get_edge_weight(l) < get_edge_weight(r); });

  // 节点
  // 最小到大 x+ y+ x- y- 父节点的方向（2位） 无用（2位）
  // graph.setTo(Scalar::all(0x00));

  // 并查集
  DisjointSet set(Row * Col);

  std::vector<bool> visited(2 * Row * Col, false);
  uint32_t count = 0;

  // 连接片段
  for (int i = 0; i < edges.size(); i++) {
    Edge e = edges[i];
    uint32_t p1, p2;
    get_points(e, Col, p1, p2);
    uint8_t w = get_edge_weight(e);
    if (set.disjoint(p1, p2) && merge_criterion(set, p1, p2, w)) {
      set.join(p1, p2, w);
      if (e.direction) {
        graph.at<uint8_t>(p1) |= 1 << 1;
        graph.at<uint8_t>(p2) |= 1 << 3;
      } else {
        graph.at<uint8_t>(p1) |= 1 << 0;
        graph.at<uint8_t>(p2) |= 1 << 2;
      }
      visited[i] = true;
      count++;
    }
  }
  // 构成全连接
  for (int i = 0; i < edges.size() &&
                  count != (Row - 2 - 2 * RAD) * (Col - 2 - 2 * RAD) - 1;
       i++) {
    if (visited[i]) continue;
    Edge e = edges[i];
    uint32_t p1, p2;
    get_points(e, Col, p1, p2);
    if (set.disjoint(p1, p2)) {
      set.join(p1, p2, get_edge_weight(e));
      if (e.direction) {
        graph.at<uint8_t>(p1) |= 1 << 1;
        graph.at<uint8_t>(p2) |= 1 << 3;
      } else {
        graph.at<uint8_t>(p1) |= 1 << 0;
        graph.at<uint8_t>(p2) |= 1 << 2;
      }
      count++;
    }
  }
  assert(count == (Row - 2 - 2 * RAD) * (Col - 2 - 2 * RAD) - 1);

  // 算法需要确定唯一的父节点
  // 最小到大 x+ y+ x- y- 父节点的方向（2位） 无用（2位）
  // 方向同样 0~3对应x+ y+ x- y- (左 上 右 下)
  std::stack<uint32_t> stack;
  stack.push(GRAPH_ROOT);
  //   cv::Mat temp{Row, Col, CV_8UC1, Scalar{0}};
  //   cvtColor(image, temp, CV_BGR2GRAY);
  while (!stack.empty()) {
    uint32_t top = stack.top();
    uint8_t node = graph.at<uint8_t>(top);
    stack.pop();
    // std::cout << top << ' ' << top / 1920 << ' ' << top % 1920 << ' '
    //           << (node >> 4) << std::endl;
    // temp.at<uint8_t>(top) = 244;
    // imshow("test", temp);
    // waitKey(1);
    if (node & (1 << 0) && ((node >> 4) != 0 || top == GRAPH_ROOT)) {
      uint32_t i = top + 1;
      stack.push(i);
      graph.at<uint8_t>(i) |= 2 << 4;
      // assert(temp.at<uint8_t>(i) == 0);
    }
    if (node & (1 << 1) && (node >> 4) != 1) {
      uint32_t i = top + Col;
      stack.push(i);
      graph.at<uint8_t>(i) |= 3 << 4;
      // assert(temp.at<uint8_t>(i) == 0);
    }
    if (node & (1 << 2) && (node >> 4) != 2) {
      uint32_t i = top - 1;
      stack.push(i);
      graph.at<uint8_t>(i) |= 0 << 4;
      // assert(temp.at<uint8_t>(i) == 0);
    }
    if (node & (1 << 3) && (node >> 4) != 3) {
      uint32_t i = top - Col;
      stack.push(i);
      graph.at<uint8_t>(i) |= 1 << 4;
      // assert(temp.at<uint8_t>(i) == 0);
    }
  }

//   show_tree(reinterpret_cast<const Mat3b&>(image),
//             reinterpret_cast<const Mat1b&>(graph));
}