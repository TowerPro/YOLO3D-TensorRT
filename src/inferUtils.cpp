#include "infer_utils.h"
#include "inner_config.h"

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cerr << "cuda failure: " << ret << std::endl;                       \
      abort();                                                                 \
    }                                                                          \
  } while (false);

const std::vector<float> mean = {0.485, 0.456, 0.406};
const std::vector<float> standard = {0.229, 0.224, 0.225}; // 标准差

const std::vector<float> averageItem = {1.5260834319115253, 1.6285898684850986,
                                        3.8839544916846327};
const std::vector<float> angleBins = {1.5707963267948966, 4.71238898038469};

cv::Mat transposeInferImage(cv::Mat &img, int h, int w) {
  // BGR2RGB
  // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  // letter box
  float r =
      std::min((h * 1.0) / (img.rows * 1.0), (w * 1.0) / (img.cols * 1.0));
  int unpaddingWidth = r * img.rows;
  int unpaddingHeight = r * img.cols;
  cv::Mat re(unpaddingWidth, unpaddingHeight, CV_8UC3);
  cv::resize(img, re, re.size());

  // std::cout << (h - unpaddingWidth) / 2 << ' ' << (w - unpaddingHeight) / 2
  //           << std::endl;

  cv::Mat out(h, w, CV_8UC3, cv::Scalar(114, 114, 114));

  int imgPosX = (h - unpaddingHeight) / 2;
  int imgPosY = (w - unpaddingWidth) / 2;
  re.copyTo(out(cv::Rect(imgPosX, imgPosY, re.cols, re.rows)));
  return out;
}

char *loadEngine(std::string engine_file_path, size_t &size) {
  std::ifstream file(engine_file_path, std::ios::binary);
  char *trtModelStream{nullptr};
  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
  } else {
    std::cerr << "[ERROR] load engine failed" << std::endl;
  }
  return trtModelStream;
}

float *blobFromImage(cv::Mat &img) {
  float *blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] =
            (float)img.at<cv::Vec3b>(h, w)(c) / 255.0;
      }
    }
  }
  return blob;
}

void classificationInference(nvinfer1::IExecutionContext &context, float *input,
                             float *output, const int output_size,
                             cv::Size input_shape) {
  const nvinfer1::ICudaEngine &engine = context.getEngine();
  assert(engine.getNbBindings() == 2);
  void *buffers[2];

  const int inputIndex = engine.getBindingIndex(CLASSIFICATION_INPUT_NAME);
  assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
  const int outputIndex = engine.getBindingIndex(CLASSIFICATION_OUTPUT_NAME);
  assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
  int mBatchSize = engine.getMaxBatchSize();

  CHECK(cudaMalloc(&buffers[inputIndex],
                   3 * input_shape.height * input_shape.width * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        3 * input_shape.height * input_shape.width *
                            sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(1, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        output_size * sizeof(float), cudaMemcpyDeviceToHost,
                        stream));
  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

bool isSmallTarget(cv::Rect rect) {
  return ((rect.width * rect.height) /
          (INNER_IMAGE_HEIGHT * INNER_IMAGE_WIDTH)) <
         CLASSIFICATION_FILTER_SMALL_TARGET_THRES;
}

std::vector<xyxyBox> classificationProbDecode(float *prob, const int size) {
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;

  int loopRange = size / (CLASSIFICATION_CLASSES + 5);
  for (int i = 0; i < loopRange; i++) {
    // conf
    float conf = prob[i * (CLASSIFICATION_CLASSES + 5) + 4];
    if (conf <= CLASSIFICATION_CONF_THRES) {
      continue;
    } else {
      // in this project just need car, simpilify the conf compute
      float car_conf = prob[i * (CLASSIFICATION_CLASSES + 5) + 4 +
                            CLASSIFICATION_TARGET_INDEX] *
                       conf;
      if (car_conf <= CLASSIFICATION_CONF_THRES) {
        continue;
      } else {
        cv::Rect rect;
        rect.x = (int)prob[i * (CLASSIFICATION_CLASSES + 5) + 0];
        rect.y = (int)prob[i * (CLASSIFICATION_CLASSES + 5) + 1];
        rect.width = (int)prob[i * (CLASSIFICATION_CLASSES + 5) + 2];
        rect.height = (int)prob[i * (CLASSIFICATION_CLASSES + 5) + 3];

        boxes.push_back(rect);
        scores.push_back(car_conf);
      }
    }
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, CLASSIFICATION_CONF_THRES,
                    CLASSIFICATION_IOU_THRES, indices);

  std::vector<xyxyBox> result;
  for (int i = 0, cnt = 0;
       cnt < CLASSIFICATION_MAX_TARGET_NUM, i < indices.size(); i++) {
    // if (isSmallTarget(boxes[indices[i]]))
    //   continue;
    ++cnt;
    result.push_back({
        std::min(640, std::max(0, boxes[indices[i]].x -
                                      boxes[indices[i]].width / 2)),
        std::min(640, std::max(0, boxes[indices[i]].y -
                                      boxes[indices[i]].height / 2)),
        std::min(640, std::max(0, boxes[indices[i]].x +
                                      boxes[indices[i]].width / 2)),
        std::min(640, std::max(0, boxes[indices[i]].y +
                                      boxes[indices[i]].height / 2)),
    });
  }

  return result;
}

float *normalizeAndCropImage(cv::Mat img, xyxyBox xyxy) {
  // crop image
  cv::Mat cropImg = img(cv::Rect(xyxy.left, xyxy.top, xyxy.right - xyxy.left,
                                 xyxy.bottom - xyxy.top));
  cv::resize(cropImg, cropImg, cv::Size(224, 224), 0, 0, cv::INTER_CUBIC);
  // cv::imwrite("regres.jpg", cropImg);

  // to tensor(float,0 - 1)
  float *tensorImg = blobFromImage(cropImg);

  // normalize
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 224; h++) {
      for (int w = 0; w < 224; w++) {
        tensorImg[c * 224 * 224 + h * 224 + w] =
            (tensorImg[c * 224 * 224 + h * 224 + w] - mean[c]) / standard[c];
      }
    }
  }

  return tensorImg;
}

void depthInference(nvinfer1::IExecutionContext &context, cv::Mat img,
                    std::vector<xyxyBox> rectList, size_t *output_size,
                    float *orientList, float *confList, float *dimList) {
  const nvinfer1::ICudaEngine &engine = context.getEngine();
  assert(engine.getNbBindings() == 4); // 1 input, 3 outputs
  void *buffers[4];

  const int inputIndex = engine.getBindingIndex(REGRESSION_INPUT_NAME);
  assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
  const int outputIndexOri = engine.getBindingIndex(REGRESSION_OUTPUT_ORIENT);
  assert(engine.getBindingDataType(outputIndexOri) ==
         nvinfer1::DataType::kFLOAT);
  const int outputIndexDim = engine.getBindingIndex(REGRESSION_OUTPUT_DIM);
  assert(engine.getBindingDataType(outputIndexDim) ==
         nvinfer1::DataType::kFLOAT);
  const int outputIndexConf = engine.getBindingIndex(REGRESSION_OUTPUT_CONF);
  assert(engine.getBindingDataType(outputIndexConf) ==
         nvinfer1::DataType::kFLOAT);
  // std::cout << inputIndex << ' ' << outputIndexOri << ' ' << outputIndexDim
  //           << ' ' << outputIndexConf << std::endl;

  CHECK(cudaMalloc(&buffers[inputIndex], 3 * 224 * 224 * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndexOri], output_size[0] * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndexConf], output_size[1] * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndexDim], output_size[2] * sizeof(float)));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  float *cropImg;
  // float *orient = new float[output_size[0]];
  // float *conf = new float[output_size[1]];
  // float *dim = new float[output_size[2]];

  for (int i = 0; i < rectList.size(); i++) {
    cropImg = normalizeAndCropImage(img, rectList[i]);
    CHECK(cudaMemcpyAsync(buffers[inputIndex], cropImg,
                          3 * 224 * 224 * sizeof(float), cudaMemcpyHostToDevice,
                          stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(
        orientList + output_size[0] * i, buffers[outputIndexOri],
        output_size[0] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(
        confList + output_size[1] * i, buffers[outputIndexConf],
        output_size[1] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(dimList + output_size[2] * i, buffers[outputIndexDim],
                          output_size[2] * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    delete[] cropImg;
    // std::cout << *(orientList + output_size[0] * i) << ' '
    //           << *(confList + output_size[1] * i) << ' '
    //           << *(dimList + output_size[2] * i) << std::endl;
    // std::cout << std::endl;
  }

  // delete[] orient;
  // delete[] conf;
  // delete[] dim;

  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndexOri]));
  CHECK(cudaFree(buffers[outputIndexConf]));
  CHECK(cudaFree(buffers[outputIndexDim]));
}

std::vector<std::vector<std::vector<int>>>
depthProbDecode(int listLength, size_t *reg_size, float *orientList,
                float *confList, float *dimList,
                std::vector<xyxyBox> rectList) {
  std::vector<int> oriIndexList;
  std::vector<float> alphaList;

  std::vector<std::vector<std::vector<int>>> result;

  for (int i = 0; i < listLength; i++) {
    dimList[i * reg_size[2] + 0] += averageItem[0];
    dimList[i * reg_size[2] + 1] += averageItem[1];
    dimList[i * reg_size[2] + 2] += averageItem[2];
    int index =
        confList[i * reg_size[1] + 0] > confList[i * reg_size[1] + 1] ? 0 : 1;
    oriIndexList.push_back(index);
    float alpha = std::atan2(orientList[i * reg_size[0] + 1],
                             orientList[i * reg_size[0] + 0]);
    alpha += angleBins[index];
    alpha -= M_PI;
    alphaList.push_back(alpha);
    float thetaRay = calcThetaRay(640, rectList[i]); // 固定尺寸图，640x640

    std::vector<std::vector<int>> location =
        calcLocation3D((dimList + i * 3), rectList[i], alpha, thetaRay);
    result.push_back(location);
  }
  return result;
}