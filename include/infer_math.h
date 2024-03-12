#ifndef JETSON_INCLUDE_INFER_MATH_H_
#define JETSON_INCLUDE_INFER_MATH_H_

#include "inner_config.h"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
typedef struct xywhBox {
  int center_x;
  int center_y;
  int width;
  int height;
};

typedef struct xyxyBox {
  int left;
  int top;
  int right;
  int bottom;
};

typedef struct dim3Box {
  int front_left_top;
  int front_left_bottom;
  int front_right_top;
  int front_right_bottom;
  int back_left_top;
  int back_left_bottom;
  int back_right_top;
  int back_right_bottom;
};

int rotationMatrix(float yaw, float pitch, float roll,
                   std::vector<std::vector<float>> &ry);

float calcThetaRay(int width, xyxyBox xyxy);

void xywh2xyxy(xywhBox xywh, xyxyBox &xyxy);

void xyxy2xywh(xyxyBox xyxy, xywhBox &xywh);

void xyxy2xywh_cv(xyxyBox xyxy, cv::Rect &xywh);

std::vector<std::vector<int>> calcLocation3D(float *dim, xyxyBox rect,
                                             float alpha, float thetaRay);

#endif // JETSON_INCLUDE_INFER_MATH_H_