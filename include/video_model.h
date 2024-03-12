#ifndef JETSON_INCLUDE_VIDEO_MODEL_H_
#define JETSON_INCLUDE_VIDEO_MODEL_H_

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sys/time.h>

#include "opencv2/opencv.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
#ifdef __cplusplus
}
#endif

class videoReformat {
private:
  // cv::Mat cvFrame;
  std::string videoPath;

  AVFormatContext *formatContext;
  AVCodecParameters *codecParams;
  AVCodecContext *codecContext;
  AVPacket packet;

  int videoStreamIndex;
  int totalFrameNum;

public:
  videoReformat(std::string videoPath);
  ~videoReformat();

  int initStream();
  int getDecodePlayer();
  int getFrame();
  int runDecode();

private:
  void releaseFormatContext();
  void releaseCodec();
};

void thread1();

#endif // JETSON_INCLUDE_VIDEO_MODEL_H_