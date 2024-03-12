#ifndef JETSON_INCLUDE_VIDEO_ERROR_H_
#define JETSON_INCLUDE_VIDEO_ERROR_H_

/**********************common define begin**********************/
#define FFMPEG_OK 0
/***********************common define end***********************/

/********************initStream define begin********************/
#define FFMPEG_FORMAT_INIT_ERROR -1000
#define FFMPEG_FORMAT_OPEN_ERROR FFMPEG_FORMAT_INIT_ERROR - 1
#define FFMPEG_FORMAT_STREAM_ERROR FFMPEG_FORMAT_INIT_ERROR - 2
#define FFPMEG_STREAM_INDEX_ERROR FFMPEG_FORMAT_INIT_ERROR - 3
/*********************initStream define end*********************/

/******************getDecodePlayer define begin******************/
#define FFMPEG_PARAM_INIT_ERROR -2000
#define FFMPEG_PARAM_COPY_ERROR FFMPEG_PARAM_INIT_ERROR - 1
#define FFMPEG_FIND_DECODER_ERROR FFMPEG_PARAM_INIT_ERROR - 2
#define FFMPEG_CODEC_CONTEXT_INIT_ERROR FFMPEG_PARAM_INIT_ERROR - 3
#define FFMPEG_SET_DECODE_CONTEXT_ERROR FFMPEG_PARAM_INIT_ERROR - 4
#define FFMPEG_OPEN_DECODE_ERROR FFMPEG_PARAM_INIT_ERROR - 5
/*******************getDecodePlayer define end*******************/

#endif // JETSON_INCLUDE_VIDEO_ERROR_H_