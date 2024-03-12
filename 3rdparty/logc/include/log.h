/**
 * Copyright (c) 2020 rxi
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the MIT license. See `log.c` for details.
 */

#ifndef LOG_H
#define LOG_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#define LOG_VERSION "0.1.0"

typedef struct {
  va_list ap;
  const char *fmt;
  const char *file;
  struct tm *time;
  void *udata;
  int line;
  int level;
} log_Event;

typedef void (*log_LogFn)(log_Event *ev);
typedef void (*log_LockFn)(bool lock, void *udata);

enum {
  INNER_LOG_TRACE,
  INNER_LOG_DEBUG,
  INNER_LOG_INFO,
  INNER_LOG_WARN,
  INNER_LOG_ERROR,
  INNER_LOG_FATAL
};

#define func_log_trace(...)                                                    \
  func_log_log(INNER_LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define func_log_debug(...)                                                    \
  func_log_log(INNER_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define func_log_info(...)                                                     \
  func_log_log(INNER_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define func_log_warn(...)                                                     \
  func_log_log(INNER_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define func_log_error(...)                                                    \
  func_log_log(INNER_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define func_log_fatal(...)                                                    \
  func_log_log(INNER_LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

const char *log_level_string(int level);
void log_set_lock(log_LockFn fn, void *udata);
void log_set_level(int level);
void log_set_quiet(bool enable);
int log_add_callback(log_LogFn fn, void *udata, int level);
int log_add_fp(FILE *fp, int level);

void func_log_log(int level, const char *file, int line, const char *fmt, ...);

#endif