#include "infer_math.h"
#include "inner_config.h"

static const std::vector<std::vector<float>> calibMatrix = {
    {718.856, 0.0, 607.1928, 45.38225},
    {0.0, 718.856, 185.2157, -0.1130887},
    {0.0, 0.0, 1.0, 0.003779761}};

float calcThetaRay(int width, xyxyBox xyxy) {
  float fovx = 2 * std::atan(width / (2 * calibMatrix[0][0]));
  int center = xyxy.left + (xyxy.right - xyxy.left) / 2;
  int dx = center - width / 2;

  int mult = dx >= 0 ? 1 : -1;
  dx = dx >= 0 ? dx : -dx;

  float angle = std::atan((2 * dx * std::tan(fovx / 2.0)) / width);
  return angle * mult;
}

// designed from this math: https://en.wikipedia.org/wiki/Rotation_matrix
int rotationMatrix(float yaw, float pitch, float roll,
                   std::vector<std::vector<float>> &ry) {
  float tx = roll, ty = yaw, tz = pitch;

  std::vector<std::vector<float>> Rx(3, std::vector<float>(3, 0.0)),
      Ry(3, std::vector<float>(3, 0.0)), Rz(3, std::vector<float>(3, 0.0));

  Rx[0][0] = Ry[1][1] = Rz[2][2] = 1.0;

  {
    Rx[1][1] = std::cos(tx);
    Rx[2][1] = std::sin(tx);
    Rx[1][2] = -Rx[2][1];
    Rx[2][2] = Rx[1][1];
  }

  {
    Ry[0][0] = std::cos(ty);
    Ry[0][2] = std::sin(ty);
    Ry[2][0] = -Ry[0][2];
    Ry[2][2] = Ry[0][0];
  }

  {
    Rz[0][0] = std::cos(tz);
    Rz[1][0] = std::sin(tz);
    Rz[0][1] = -Rz[1][0];
    Rz[1][1] = Rz[0][0];
  }

  ry = Ry;
  return 0;
}

void xywh2xyxy(xywhBox xywh, xyxyBox &xyxy) {
  xyxy.left = xywh.center_x - xywh.width / 2;
  xyxy.top = xywh.center_y - xywh.height / 2;
  xyxy.right = xywh.center_x + xywh.width / 2;
  xyxy.bottom = xywh.center_y + xywh.height / 2;
}

void xyxy2xywh(xyxyBox xyxy, xywhBox &xywh) {
  xywh.center_x = xyxy.left + (xyxy.right - xyxy.left) / 2;
  xywh.center_y = xyxy.top + (xyxy.bottom - xyxy.top) / 2;
  xywh.width = xyxy.right - xyxy.left;
  xywh.height = xyxy.bottom - xyxy.top;
}

void xyxy2xywh_cv(xyxyBox xyxy, cv::Rect &xywh) {
  xywh.x = xyxy.left + (xyxy.right - xyxy.left) / 2;
  xywh.y = xyxy.top + (xyxy.bottom - xyxy.top) / 2;
  xywh.width = xyxy.right - xyxy.left;
  xywh.height = xyxy.bottom - xyxy.top;
}

bool doubleSame(std::vector<float> a, std::vector<float> b) {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

bool theSame(std::vector<float> a, std::vector<float> b, std::vector<float> c,
             std::vector<float> d) {
  if (doubleSame(a, b) || doubleSame(a, c) || doubleSame(a, d) ||
      doubleSame(b, c) || doubleSame(b, d) || doubleSame(c, d)) {
    return true;
  }
  return false;
}

std::vector<float> vecDot1x3f(std::vector<std::vector<float>> R,
                              std::vector<float> X) {
  std::vector<float> result;
  for (auto r : R) {
    float dot = std::inner_product(r.begin(), r.end(), X.begin(), 0.0f);
    result.push_back(dot);
  }
  return result;
}

std::vector<std::vector<float>>
vecDotCalib3x4f(std::vector<std::vector<float>> M) {
  std::vector<std::vector<float>> result(3, std::vector<float>(4, 0.0f));
  for (int i = 0; i < calibMatrix.size(); i++) {
    for (int j = 0; j < M[0].size(); j++) {
      for (int k = 0; k < M.size(); ++k) {
        result[i][j] += calibMatrix[i][k] * M[k][j];
      }
    }
  }
  return result;
}

std::vector<std::vector<float>> vecDot3x8f(std::vector<std::vector<float>> R,
                                           std::vector<float> x,
                                           std::vector<float> y,
                                           std::vector<float> z) {
  std::vector<std::vector<float>> result(3, std::vector<float>(8, 0.0f));
  for (int i = 0; i < R.size(); i++) {
    for (int j = 0; j < x.size(); j++) {
      result[i][j] = R[i][0] * x[j] + R[i][1] * y[j] + R[i][2] * z[j];
    }
  }
  return result;
}

void matrixAbMult(std::vector<std::vector<float>> &A, std::vector<float> &b,
                  std::vector<std::vector<float>> M, xyxyBox rect, int row,
                  int index) {
  int box_corner_row = 0;
  switch (row) {
  case 0: {
    box_corner_row = rect.left;
    break;
  }
  case 1: {
    box_corner_row = rect.top;
    break;
  }
  case 2: {
    box_corner_row = rect.right;
    break;
  }
  case 3: {
    box_corner_row = rect.bottom;
    break;
  }
  default: {
    break;
  }
  }
  for (int i = 0; i < A[0].size(); i++) {
    A[row][i] = M[index][i] - box_corner_row * M[2][i];
  }
  b[row] = box_corner_row * M[2][3] - M[index][3];
}

// 最小二乘法线性回归
float performLinearRegression(const std::vector<std::vector<float>> &X,
                              const std::vector<float> &y,
                              std::vector<float> &theta) {
  int numSamples = X.size();
  int numFeatures = X[0].size();

  // Create matrices and vectors
  std::vector<std::vector<float>> Xt(numFeatures,
                                     std::vector<float>(numSamples));
  std::vector<std::vector<float>> XtX(numFeatures,
                                      std::vector<float>(numFeatures));
  std::vector<float> Xty(numFeatures, 0.0);

  // Transpose X
  for (int i = 0; i < numSamples; i++) {
    for (int j = 0; j < numFeatures; j++) {
      Xt[j][i] = X[i][j];
    }
  }

  // Compute XtX and Xty
  for (int i = 0; i < numFeatures; i++) {
    for (int j = 0; j < numFeatures; j++) {
      for (int k = 0; k < numSamples; k++) {
        XtX[i][j] += Xt[i][k] * X[k][j];
      }
    }

    for (int k = 0; k < numSamples; k++) {
      Xty[i] += Xt[i][k] * y[k];
    }
  }

  // Solve for theta using Gaussian elimination
  for (int i = 0; i < numFeatures; i++) {
    for (int j = i + 1; j < numFeatures; j++) {
      float ratio = XtX[j][i] / XtX[i][i];
      for (int k = 0; k < numFeatures; k++) {
        XtX[j][k] -= ratio * XtX[i][k];
      }
      Xty[j] -= ratio * Xty[i];
    }
  }

  // Back substitution
  for (int i = numFeatures - 1; i >= 0; i--) {
    theta[i] = Xty[i];
    for (int j = i + 1; j < numFeatures; j++) {
      theta[i] -= XtX[i][j] * theta[j];
    }
    theta[i] /= XtX[i][i];
  }

  // Calculate residuals
  float residuals = 0.0;
  for (int i = 0; i < numSamples; i++) {
    float predicted = 0.0;
    for (int j = 0; j < numFeatures; j++) {
      predicted += theta[j] * X[i][j];
    }
    float diff = predicted - y[i];
    residuals += diff * diff;
  }

  return residuals;
}

std::vector<std::vector<float>> createCorners(float *dim, float orient,
                                              std::vector<float> location) {
  float dx = dim[2] / 2.0;
  float dy = dim[0] / 2.0;
  float dz = dim[1] / 2.0;

  std::vector<float> x_corners, y_corners, z_corners;
  int loop[2] = {1, -1};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        x_corners.push_back(dx * loop[i]);
        y_corners.push_back(dy * loop[j]);
        z_corners.push_back(dz * loop[k]);
      }
    }
  }

  std::vector<std::vector<float>> R;
  rotationMatrix(orient, 0.0, 0.0, R);

  std::vector<std::vector<float>> corners =
      vecDot3x8f(R, x_corners, y_corners, z_corners);
  for (int i = 0; i < corners.size(); i++) {
    for (int j = 0; j < corners[0].size(); j++) {
      corners[i][j] = corners[i][j] + location[i];
    }
  }

  // transpose matrix
  std::vector<std::vector<float>> tranposeCorners(
      corners[0].size(), std::vector<float>(corners.size()));

  for (int i = 0; i < tranposeCorners.size(); i++) {
    for (int j = 0; j < tranposeCorners[0].size(); j++) {
      tranposeCorners[i][j] = corners[j][i];
    }
  }

  return tranposeCorners; // corners: 3x8, 横竖需要交换
}

void project3DPoint(std::vector<float> pt, std::vector<int> &point) {
  pt.push_back(1);
  auto tmp = vecDot1x3f(calibMatrix, pt);
  // 限制范围，越界crash，640x640范围0-639
  point[0] = std::min(640, std::max(0, (int)(tmp[0] / tmp[2])));
  point[1] = std::min(640, std::max(0, (int)(tmp[1] / tmp[2])));
}

// designed from this math http://ywpkwon.github.io/pdf/bbox3d-study.pdf
std::vector<std::vector<int>> calcLocation3D(float *dim, xyxyBox rect,
                                             float alpha, float thetaRay) {
  float orient = alpha + thetaRay;
  std::vector<std::vector<float>> R;
  rotationMatrix(orient, 0, 0, R);

  std::vector<std::vector<float>> left_constraints;
  std::vector<std::vector<float>> right_constraints;
  std::vector<std::vector<float>> top_constraints;
  std::vector<std::vector<float>> bottom_constraints;

  float dx = *(dim + 2) / 2.0;
  float dy = *(dim + 0) / 2.0;
  float dz = *(dim + 1) / 2.0;

  float best_score = FLT_MAX;
  std::vector<std::vector<float>> result(5);

  int left_mult = 1, right_mult = -1;

  if (alpha < deg2rad_92 && alpha > deg2rad_88) {
    left_mult = right_mult = 1;
  } else if (alpha < deg2rad_88_minus && alpha > deg2rad_92_minus) {
    left_mult = right_mult = -1;
  } else if (alpha < deg2rad_90 && alpha > (-1.0 * deg2rad_90)) {
    left_mult = -1;
    right_mult = 1;
  }
  int switch_mult = alpha > 0 ? 1 : -1;
  int loopDir[2] = {-1, 1};
  for (int i = 0; i < 2; i++) {
    left_constraints.push_back(
        {left_mult * dx, loopDir[i] * dy, -switch_mult * dz});
    right_constraints.push_back(
        {right_mult * dx, loopDir[i] * dy, switch_mult * dz});
    for (int j = 0; j < 2; j++) {
      top_constraints.push_back({loopDir[i] * dx, -dy, loopDir[j] * dz});
      bottom_constraints.push_back({loopDir[i] * dx, dy, loopDir[j] * dz});
    }
  }
  // std::vector<std::vector<std::vector<float>>> constrains;
  for (auto left : left_constraints) {
    for (auto top : top_constraints) {
      for (auto right : right_constraints) {
        for (auto bottom : bottom_constraints) {
          // 去重，filter(lambda x: len(x) == len(set(tuple(i) for i in x))
          if (theSame(left, top, right, bottom)) {
            continue;
          } else {
            std::vector<std::vector<float>> X_array({left, top, right, bottom});
            std::vector<std::vector<std::vector<float>>> M_array = {
                {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
                {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
                {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
                {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}},
            };
            int indicies[4] = {0, 1, 0, 1};
            std::vector<std::vector<float>> A(4, std::vector<float>(3, 0));
            std::vector<float> b(4, 0);

            for (int i = 0; i < 4; i++) {
              std::vector<float> X = X_array[i];
              std::vector<std::vector<float>> M = M_array[i];
              std::vector<float> RX = vecDot1x3f(R, X);
              for (int j = 0; j < 3; j++) {
                M[j][3] = RX[j];
              }
              std::vector<std::vector<float>> dotM = vecDotCalib3x4f(M);
              matrixAbMult(A, b, dotM, rect, i, indicies[i]);
            }

            // linalg.lstsq
            std::vector<float> tmpLoc(3, 0.0);

            float score = performLinearRegression(A, b, tmpLoc);

            if (score < best_score) {
              best_score = score;
              result[0] = tmpLoc;
              result[1] = left;
              result[2] = top;
              result[3] = right;
              result[4] = bottom;
            }
          }
        }
      }
    }
  }
  std::vector<std::vector<float>> corners =
      createCorners(dim, orient, result[0]);

  std::vector<std::vector<int>> returnPoints;
  for (auto corner : corners) {
    std::vector<int> point(2);
    project3DPoint(corner, point);
    returnPoints.push_back(point);
  }

  return returnPoints;
}