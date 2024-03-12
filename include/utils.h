#ifndef JETSON_INCLUDE_UTILS_H_
#define JETSON_INCLUDE_UTILS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

struct Rect {
  int left;
  int top;
  int right;
  int bottom;
};

// struct Point {
//   int x;
//   int y;
// };

struct Bbox3DState {
  int x;
  int y;
  int z;
  int w;
  int h;
  int l;
  float alpha;
};

struct Bbox3D {
  Rect rect;
  cv::Point points[8];
  float thetas;
};

#endif // JETSON_INCLUDE_UTILS_H_