#ifndef JETSON_INCLUDE_INFER_UTILS_H_
#define JETSON_INCLUDE_INFER_UTILS_H_

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "infer_math.h"
// #include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <sstream>

cv::Mat transposeInferImage(cv::Mat &img, int h, int w);

char *loadEngine(std::string engine_file_path, size_t &size);

float *blobFromImage(cv::Mat &img);

void classificationInference(nvinfer1::IExecutionContext &context, float *input,
                             float *output, const int output_size,
                             cv::Size input_shape);

std::vector<xyxyBox> classificationProbDecode(float *prob, const int size);

float *normalizeAndCropImage(cv::Mat img, xyxyBox xyxy);

void depthInference(nvinfer1::IExecutionContext &context, cv::Mat img,
                    std::vector<xyxyBox> rectList, size_t *output_size,
                    float *orientList, float *confList, float *dimList);

std::vector<std::vector<std::vector<int>>>
depthProbDecode(int listLength, size_t *reg_size, float *orientList,
                float *confList, float *dimList, std::vector<xyxyBox> rectList);

#endif // JETSON_INCLUDE_INFER_UTILS_H_
