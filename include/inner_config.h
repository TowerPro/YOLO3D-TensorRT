#ifndef JETSON_INCLUDE_CONFIG_H_
#define JETSON_INCLUDE_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif
#include "log.h"
#include <stdio.h>
#if 0
#define FUNC_LOG_INFO(...) func_log_info(__VA_ARGS__)
#define FUNC_LOG_WARNING(...) func_log_warn(__VA_ARGS__)
#define FUNC_LOG_ERROR(...) func_log_error(__VA_ARGS__)
#else
#define FUNC_LOG_INFO(...)
#define FUNC_LOG_WARNING(...)
#define FUNC_LOG_ERROR(...)
#endif

#ifdef __cplusplus
}
#endif

/**************inferUtils define begin**************/
#define CLASSIFICATION_INPUT_NAME "images"
#define CLASSIFICATION_OUTPUT_NAME "output"
//  转模型的时候打错了，懒的改了
#define REGRESSION_INPUT_NAME "imgaes"
#define REGRESSION_OUTPUT_DIM "dim"
#define REGRESSION_OUTPUT_CONF "conf"
#define REGRESSION_OUTPUT_ORIENT "orient"

#define CLASSIFICATION_CONF_THRES 0.25
#define CLASSIFICATION_IOU_THRES 0.45
// 分类中选择需要过滤掉的小目标占比（和整张图相比）
#define CLASSIFICATION_FILTER_SMALL_TARGET_THRES 0.05

#define CLASSIFICATION_CLASSES 80
#define CLASSIFICATION_MAX_NMS 30000
#define CLASSIFICATION_MIN_WH 2
#define CLASSIFICAITON_MAX_WH 7680
#define CLASSIFICATION_TARGET_INDEX 3
// 分类中允许的最大检测到的目标数，可修改
#define CLASSIFICATION_MAX_TARGET_NUM 3
/***************inferUtils define end***************/

/**************inferMath define begin**************/
#define deg2rad_92 1.6057029118347832
#define deg2rad_88 1.53588974175501
#define deg2rad_90 1.5707963267948966
#define deg2rad_92_minus -1.0 * deg2rad_92
#define deg2rad_88_minus -1.0 * deg2rad_88
/***************inferMath define end***************/

/****************main define begin****************/
#define SRC_IMAGE_WIDTH 640
#define SRC_IMAGE_HEIGHT 640
#define SRC_IMAGE_SIZE (SRC_IMAGE_WIDTH * SRC_IMAGE_HEIGHT * 3)
#define SRC_IMAGE_SCALE 0.95
#define SRC_IMAGE_Y_OFFSET 104
#define THREAD_MAX_NUM 100

#define INNER_IMAGE_WIDTH SRC_IMAGE_WIDTH
#define INNER_IMAGE_HEIGHT 480
/*****************main define end*****************/

#endif // JETSON_INCLUDE_CONFIG_H_