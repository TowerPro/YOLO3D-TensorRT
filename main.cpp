#include "bbox_queue.h"
#include "cuda_runtime_api.h"
#include "infer_math.h"
#include "infer_utils.h"
#include "inner_config.h"
#include "tensorrt_logging.h"
#include "utils.h"
#include <NvInfer.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <vector>

// using ffmpeg
#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
#ifdef __cplusplus
}
#endif

AVPixelFormat get_pix_fmt(AVPixelFormat fmt) {
  switch (fmt) {
  case AV_PIX_FMT_YUVJ420P:
    return AV_PIX_FMT_YUV420P;
  case AV_PIX_FMT_YUVJ422P:
    return AV_PIX_FMT_YUV422P;
  case AV_PIX_FMT_YUVJ411P:
    return AV_PIX_FMT_YUV411P;
  case AV_PIX_FMT_YUVJ444P:
    return AV_PIX_FMT_YUVJ444P;
  default:
    return fmt;
  }
}

#define DUMP_IMAGE 1
#define RUN_3D 1

#define SAFE_FREE(x)                                                           \
  do {                                                                         \
    if (x != nullptr) {                                                        \
      free(x);                                                                 \
      x = nullptr;                                                             \
    }                                                                          \
  } while (false);

#define SET_CPU_AFFINITY(x, returnVal)                                         \
  do {                                                                         \
    cpu_set_t mask;                                                            \
    cpu_set_t get;                                                             \
    CPU_ZERO(&mask);                                                           \
    CPU_SET(x, &mask);                                                         \
    if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) == -1) {                \
      FUNC_LOG_INFO("warning: could not set cpu affinity...");                 \
      pthread_exit(returnVal);                                                 \
    }                                                                          \
  } while (false);

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cerr << "cuda failure: " << ret << std::endl;                       \
      abort();                                                                 \
    }                                                                          \
  } while (false);

std::atomic<bool> cameraThreadStopFlag(false);
std::atomic<bool> frameThreadStopFlag(false);
std::atomic<bool> generateThreadStopFlag(false);
std::atomic<bool> inferenceThreadStopFlag(false);
std::atomic<bool> hasDrawedRectangle(false);
std::atomic<bool> useCameraFlag(true);

std::mutex bufferMtx;

cv::Mat sharedBuf(SRC_IMAGE_HEIGHT, SRC_IMAGE_WIDTH, CV_8UC3);

bboxQue bboxQueue;
int cpuNumber = 0; // cpu数

static Logger gLogger;

void signalHandler(int signal) {
  cameraThreadStopFlag = true;
  frameThreadStopFlag = true;
  generateThreadStopFlag = true;
  inferenceThreadStopFlag = true;
}

void *draw3DBboxThread(void *arg) {
  SET_CPU_AFFINITY(0, NULL);
  while (true) {
    std::vector<cv::Point *> pointArrays;

    FUNC_LOG_INFO("QUEUE SIZE %d", bboxQueue.size());
    while (!bboxQueue.empty()) {
      Bbox3D bbox = {0};
      bboxQueue.pop(bbox);
      pointArrays.push_back(bbox.points);
    }

    if (!pointArrays.empty()) {
      FUNC_LOG_INFO("POINT ARRAY SIZE %d", pointArrays.size());
      {
        std::lock_guard<std::mutex> lock(bufferMtx);
        int npt[] = {8};
        for (auto p : pointArrays) {
          cv::polylines(sharedBuf, &p, npt, 1, true,
                        cv::Scalar(0.0, 0.0, 255.0), 2);
        }
        cv::imshow("Camera", sharedBuf);
      }
    }

    if (cv::waitKey(1) == 'q') {
      break;
    }
  }
  cv::destroyAllWindows();

  FUNC_LOG_INFO("EXIT 3DBOX DRAWING THREAD");
  pthread_exit(NULL);
}

void *imageThread(void *arg) {
  SET_CPU_AFFINITY(0, NULL);
  cv::Mat frame;
  char *imagePath = (char *)arg;
  frame = cv::imread(imagePath);
  while (!cameraThreadStopFlag) {
    frame = transposeInferImage(frame, 640, 640);
    {
      std::lock_guard<std::mutex> lock(bufferMtx);
      frame.copyTo(sharedBuf);
      hasDrawedRectangle = false;
    }
  }
  printf("EXIT IMAGE THREAD");
  pthread_exit(NULL);
}

void *cameraThread(void *arg) {
  SET_CPU_AFFINITY(0, NULL);
  static int flag = 0;
  int cnt = 0;
  cv::Mat frame;

  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    return NULL;
  }

  while (!cameraThreadStopFlag) {
    flag++;
    capture.read(frame);

    if (frame.empty()) {
      FUNC_LOG_INFO("CAMERA CAPTURE DATA EMPTY");
      break;
    }
    frame = transposeInferImage(frame, 640, 640);
    {
      std::lock_guard<std::mutex> lock(bufferMtx);
      frame.copyTo(sharedBuf);
      hasDrawedRectangle = false;
    }
  }
  frame.release();
  capture.release();

  FUNC_LOG_INFO("EXIT CAMERA CAPTURE THREAD");
  pthread_exit(NULL);
}

void *inferenceThread(void *arg) {
  SET_CPU_AFFINITY(1, NULL);

  char **modelPackage = (char **)arg;

  // init classification model
  size_t classificationSize = 0;
  char *trtModelStreamClassification =
      loadEngine(modelPackage[0], classificationSize);
  if (trtModelStreamClassification) {
    std::cout << "load classification model successfully" << std::endl;
  }

  nvinfer1::IRuntime *runtimeClassification =
      nvinfer1::createInferRuntime(gLogger);
  assert(runtimeClassification != nullptr);
  nvinfer1::ICudaEngine *engineClassification =
      runtimeClassification->deserializeCudaEngine(trtModelStreamClassification,
                                                   classificationSize);
  assert(engineClassification != nullptr);
  nvinfer1::IExecutionContext *contextClassification =
      engineClassification->createExecutionContext();
  assert(contextClassification != nullptr);
  delete[] trtModelStreamClassification;

  auto classification_out_dims = engineClassification->getBindingDimensions(1);

  auto classification_output_size = 1;
  for (int j = 0; j < classification_out_dims.nbDims; j++) {
    classification_output_size *= classification_out_dims.d[j];
  }

  FUNC_LOG_INFO("classification model output size: %d",
                classification_output_size);

  static float *prob = new float[classification_output_size];
  float *blob;
  int cnt = 0;

// init regression model
#if RUN_3D
  size_t regressionSize = 0;
  char *trtModelStreamRegression = loadEngine(modelPackage[1], regressionSize);
  if (trtModelStreamRegression) {
    std::cout << "load regression model successfully" << std::endl;
  }
  std::cout << regressionSize << std::endl;

  nvinfer1::IRuntime *runtimeRegression = nvinfer1::createInferRuntime(gLogger);
  assert(runtimeRegression != nullptr);
  nvinfer1::ICudaEngine *engineRegression =
      runtimeRegression->deserializeCudaEngine(trtModelStreamRegression,
                                               regressionSize);
  assert(engineRegression != nullptr);
  nvinfer1::IExecutionContext *contextRegression =
      engineRegression->createExecutionContext();
  assert(contextRegression != nullptr);
  delete[] trtModelStreamRegression;

  // three output head for orient, conf, dim
  size_t regress_out_size[3] = {1, 1, 1};
  for (int j = 0; j < 3; j++) {
    auto out_dims = engineRegression->getBindingDimensions(j + 1);
    for (int k = 0; k < out_dims.nbDims; k++) {
      regress_out_size[j] *= out_dims.d[k];
    }
  }

  FUNC_LOG_INFO("regression model output size: %d, %d, %d", regress_out_size[0],
                regress_out_size[1], regress_out_size[2]);

  float *orientList;
  float *confList;
  float *dimList;
#endif

  while (!inferenceThreadStopFlag) {
    cv::Mat resize_image;
    {
      std::lock_guard<std::mutex> lock(bufferMtx);
      resize_image = sharedBuf;
    }

    blob = blobFromImage(resize_image);
    auto start = std::chrono::system_clock::now();
    classificationInference(*contextClassification, blob, prob,
                            classification_output_size, resize_image.size());
    FUNC_LOG_INFO("prob[1]: %f", prob[1]);

    std::vector<xyxyBox> result =
        classificationProbDecode(prob, classification_output_size);

#if RUN_3D
    int resSize = result.size();
    orientList = new float[resSize * regress_out_size[0]];
    confList = new float[resSize * regress_out_size[1]];
    dimList = new float[resSize * regress_out_size[2]];

    depthInference(*contextRegression, resize_image, result, regress_out_size,
                   orientList, confList, dimList);
    // std::cout << "depthinference" << std::endl;
    // for (int i = 0; i < resSize; i++) {
    //   std::cout << orientList[i * regress_out_size[0]] << ' '
    //             << confList[i * regress_out_size[1]] << ' '
    //             << dimList[i * regress_out_size[2]] << std::endl;
    // }
    std::vector<std::vector<std::vector<int>>> location3Dlist = depthProbDecode(
        resSize, regress_out_size, orientList, confList, dimList, result);
    FUNC_LOG_INFO("3d boxes size: %d", location3Dlist.size());

    delete[] orientList;
    delete[] confList;
    delete[] dimList;

#endif

    auto end = std::chrono::system_clock::now();

    printf("inference fps: %.2f fps\n",
           (1000.0 /
            (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count())));
    if (!hasDrawedRectangle) {
      hasDrawedRectangle = true;
#if RUN_3D
      for (auto location : location3Dlist) {
        cv::line(resize_image, cv::Point(location[0][0], location[0][1]),
                 cv::Point(location[2][0], location[2][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[4][0], location[4][1]),
                 cv::Point(location[6][0], location[6][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[0][0], location[0][1]),
                 cv::Point(location[4][0], location[4][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[2][0], location[2][1]),
                 cv::Point(location[6][0], location[6][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);

        cv::line(resize_image, cv::Point(location[1][0], location[1][1]),
                 cv::Point(location[3][0], location[3][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[1][0], location[1][1]),
                 cv::Point(location[5][0], location[5][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[7][0], location[7][1]),
                 cv::Point(location[3][0], location[3][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[7][0], location[7][1]),
                 cv::Point(location[5][0], location[5][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);

        cv::line(resize_image, cv::Point(location[0][0], location[0][1]),
                 cv::Point(location[1][0], location[1][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[2][0], location[2][1]),
                 cv::Point(location[3][0], location[3][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[4][0], location[4][1]),
                 cv::Point(location[5][0], location[5][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
        cv::line(resize_image, cv::Point(location[7][0], location[7][1]),
                 cv::Point(location[6][0], location[6][1]),
                 cv::Scalar(0.0, 0.0, 255.0), 2);
      }
#endif
      for (auto rect : result) {
        cv::rectangle(resize_image, cv::Point(rect.left, rect.top),
                      cv::Point(rect.right, rect.bottom),
                      cv::Scalar(0.0, 255.0, 0.0), 2);
      }

#if DUMP_IMAGE
      char dumpPath[100] = "./dump.jpg";
      cv::imwrite(dumpPath, resize_image);
      FUNC_LOG_INFO("%s, %d, DUMP IMAGE: %s\n", __func__, __LINE__, dumpPath);
#endif
    }

    {
      std::lock_guard<std::mutex> lock(bufferMtx);
      resize_image.copyTo(sharedBuf);
    }
    resize_image.release();
    delete[] blob;

    // cv::imshow("Camera", resize_image);
  }

  delete[] prob;

  contextClassification->destroy();
  engineClassification->destroy();
  runtimeClassification->destroy();

#if RUN_3D
  contextRegression->destroy();
  engineRegression->destroy();
  runtimeRegression->destroy();
#endif

  sharedBuf.release();

  printf("EXIT INFERENCE THREAD");
  pthread_exit(NULL);
}

void *showThread(void *arg) {
  while (true) {
    {
      std::lock_guard<std::mutex> lock(bufferMtx);
      cv::imshow("3D Detection", sharedBuf);
      // cv::imwrite("./dump.jpg", sharedBuf);
    }
    if (cv::waitKey(1) == 'q') {
      break;
    }
  }
  cv::destroyAllWindows();
  printf("EXIT SHOW THREAD");
  pthread_exit(NULL);
}

void dumpUchar() {
  char savePath[100];
  int cnt = 0;
  while (!frameThreadStopFlag) {
    sprintf(savePath, "./uchar_frame_%d.jpg", (++cnt) % 25);
    {
      std::lock_guard<std::mutex> lock(bufferMtx);
      cv::imwrite(savePath, sharedBuf);
    }
  }
}

int main(int argc, char *args[]) {
  switch (argc) {
  case 3: {
    std::cout << "ruuning in the camera way" << std::endl;
    useCameraFlag = true;
    break;
  }
  case 4: {
    std::cout << "running in the image way" << std::endl;
    useCameraFlag = false;
    break;
  }
  default: {
    std::cerr << "input format error.\nrun in camera "
                 "model:\nexecpath/to/classification model path/to/regression "
                 "model\n\nrun in image model:\nexec path/to/classification "
                 "model path/to/regression model path/to/image"
              << std::endl;
    return -1;
  }
  }

  for (int i = 1; i < argc; i++) {
    printf("load file: %s\n", args[i]);
  }

  signal(SIGINT, signalHandler);
  cudaSetDevice(0);

  cv::namedWindow("3D Detection", cv::WINDOW_NORMAL);

  pthread_t data;
  pthread_t infer;
  pthread_t show;

  if (useCameraFlag) {
    pthread_create(&data, NULL, cameraThread, NULL);
  } else {
    pthread_create(&data, NULL, imageThread, args[3]);
  }
  pthread_create(&infer, NULL, inferenceThread, args + 1);
  pthread_create(&show, NULL, showThread, NULL);

  pthread_join(data, NULL);
  pthread_join(infer, NULL);
  pthread_join(show, NULL);

  return 0;
}
