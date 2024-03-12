#include "video_model.h"
#include "video_error.h"
#include <condition_variable>

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      printf("function: %s, line: %d error\n", __func__, __LINE__);            \
      abort();                                                                 \
    }                                                                          \
  } while (false)

cv::Mat sharedBuffer;
std::mutex sharedMutex;

static bool hasFrame = false;

uint64_t getTimeStamp() {
  struct timeval tv = {0};
  gettimeofday(&tv, NULL);
  return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

videoReformat::videoReformat(std::string videoPath) {
  printf("[%ld] init function --> videoReformat <--\n", getTimeStamp());
  this->videoPath = videoPath;
  this->videoStreamIndex = -1;
}

videoReformat::~videoReformat() {
  releaseFormatContext();
  releaseCodec();
}

void videoReformat::releaseFormatContext() {
  if (formatContext->iformat != nullptr) {
    avformat_close_input(&formatContext);
  }
  if (formatContext != nullptr) {
    avformat_free_context(formatContext);
  }
}

void videoReformat::releaseCodec() {
  if (codecParams != nullptr) {
    avcodec_parameters_free(&codecParams);
  }
  if (codecContext != nullptr) {
    avcodec_free_context(&codecContext);
  }
}

int videoReformat::initStream() {
  // init avformatContext
  formatContext = avformat_alloc_context();
  if (formatContext == nullptr) {
    return FFMPEG_FORMAT_INIT_ERROR;
  }

  if (avformat_open_input(&formatContext, videoPath.c_str(), nullptr,
                          nullptr) != 0) {
    releaseFormatContext();
    return FFMPEG_FORMAT_OPEN_ERROR;
  }

  if (avformat_find_stream_info(formatContext, nullptr) < 0) {
    releaseFormatContext();
    return FFMPEG_FORMAT_STREAM_ERROR;
  }

  for (unsigned int i = 0; i < formatContext->nb_streams; ++i) {
    if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoStreamIndex = i;
      break;
    }
  }
  if (videoStreamIndex == -1) {
    releaseFormatContext();
    return FFPMEG_STREAM_INDEX_ERROR;
  }

  return FFMPEG_OK;
}

int videoReformat::getDecodePlayer() {
  codecParams = avcodec_parameters_alloc();
  if (!codecParams) {
    releaseFormatContext();
    return FFMPEG_PARAM_INIT_ERROR;
  }

  if (avcodec_parameters_copy(
          codecParams, formatContext->streams[videoStreamIndex]->codecpar) <
      0) {
    releaseFormatContext();
    releaseCodec();
    return FFMPEG_PARAM_COPY_ERROR;
  }

  const AVCodec *codec = avcodec_find_decoder(codecParams->codec_id);
  if (codec == nullptr) {
    releaseFormatContext();
    releaseCodec();
    return FFMPEG_FIND_DECODER_ERROR;
  }

  codecContext = avcodec_alloc_context3(codec);
  if (codecContext == nullptr) {
    releaseFormatContext();
    releaseCodec();
    return FFMPEG_CODEC_CONTEXT_INIT_ERROR;
  }

  if (avcodec_parameters_to_context(codecContext, codecParams) < 0) {
    releaseFormatContext();
    releaseCodec();
    return FFMPEG_SET_DECODE_CONTEXT_ERROR;
  }

  if (avcodec_open2(codecContext, codec, nullptr) < 0) {
    releaseFormatContext();
    releaseCodec();
    return FFMPEG_OPEN_DECODE_ERROR;
  }

  totalFrameNum = formatContext->streams[videoStreamIndex]->nb_frames;

  return FFMPEG_OK;
}

int videoReformat::getFrame() {
  av_init_packet(&packet);
  while (av_read_frame(formatContext, &packet) >= 0) {
    if (packet.stream_index == videoStreamIndex) {
      if (avcodec_send_packet(codecContext, &packet) < 0) {
        std::cerr << "Failed to send packet to decoder" << std::endl;
        break;
      }

      AVFrame *frame = av_frame_alloc();
      AVFrame *rgbFrame = av_frame_alloc();
      rgbFrame->height = codecContext->height;
      rgbFrame->width = codecContext->width;

      int bufSize = av_image_get_buffer_size(AV_PIX_FMT_BGR24, rgbFrame->height,
                                             rgbFrame->width, 1);
      uint8_t *outBuffer = (uint8_t *)av_malloc(bufSize);
      av_image_fill_arrays(rgbFrame->data, rgbFrame->linesize, outBuffer,
                           AV_PIX_FMT_BGR24, rgbFrame->width, rgbFrame->height,
                           1);

      SwsContext *swsContext = sws_getContext(
          codecContext->width, codecContext->height, codecContext->pix_fmt,
          codecContext->width, codecContext->height, AV_PIX_FMT_BGR24,
          SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

      while (avcodec_receive_frame(codecContext, frame) == 0 &&
             codecContext->frame_num <= totalFrameNum) {
        sws_scale(swsContext, frame->data, frame->linesize, 0, frame->height,
                  rgbFrame->data, rgbFrame->linesize);
        cv::Mat cvFrame(cv::Size(codecContext->width, codecContext->height),
                        CV_8UC3);
        cvFrame.data = (uint8_t *)rgbFrame->data[0];

        {
          std::lock_guard<std::mutex> lock(sharedMutex);
          sharedBuffer = cvFrame;
        }
        if (!hasFrame)
          hasFrame = true;
        // cv::imwrite("./frame.jpg", cvFrame);
      }
      av_frame_unref(frame);
      av_frame_unref(rgbFrame);
      av_free(outBuffer);
      av_frame_free(&frame);
      av_frame_free(&rgbFrame);
      sws_freeContext(swsContext);
    }
  }
  av_packet_unref(&packet);
  return FFMPEG_OK;
}

int videoReformat::runDecode() {
  int ret = -1;
  if ((ret = initStream()) != FFMPEG_OK) {
    printf("initStream error, ret = %d\n", ret);
    return ret;
  }

  if ((ret = getDecodePlayer() != FFMPEG_OK)) {
    printf("getDecodePlayer error, ret = %d\n", ret);
    return ret;
  }

  ret = getFrame();
  return ret;
}

void thread1() {
  while (!hasFrame) {
  }
  while (true) {
    cv::Mat m;
    {
      std::lock_guard<std::mutex> lock(sharedMutex);
      m = sharedBuffer;
    }
    cv::imwrite("./frame.jpg", m);

    if (cv::waitKey(1) == 'q') {
      break;
    }
  }
}