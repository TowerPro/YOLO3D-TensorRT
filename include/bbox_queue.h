#ifndef JETSON_INCLUDE_BBOX_QUEUE_H_
#define JETSON_INCLUDE_BBOX_QUEUE_H_

#include "utils.h"
#include <mutex>
#include <queue>

class bboxQue {
public:
  bboxQue() {}
  void push(Bbox3D &bbox);
  void pop(Bbox3D &bbox);
  int size();
  void clean();
  bool empty();

private:
  std::queue<Bbox3D> _que;
  std::mutex _mtx;
};

void bboxQue::push(Bbox3D &bbox) {
  std::lock_guard<std::mutex> lock(_mtx);
  _que.push(bbox);
}

void bboxQue::pop(Bbox3D &bbox) {
  std::lock_guard<std::mutex> lock(_mtx);
  if (!_que.empty()) {
    bbox = _que.front();
    _que.pop();
  }
}

int bboxQue::size() {
  int s = 0;
  {
    std::lock_guard<std::mutex> lock(_mtx);
    s = _que.size();
  }
  return s;
}

void bboxQue::clean() {
  std::lock_guard<std::mutex> lock(_mtx);
  while (!_que.empty()) {
    _que.pop();
  }
}

bool bboxQue::empty() {
  bool isEmpty = false;
  {
    std::lock_guard<std::mutex> lock(_mtx);
    isEmpty = _que.empty();
  }
  return isEmpty;
}

#endif // JETSON_INCLUDE_BBOX_QUEUE_H_