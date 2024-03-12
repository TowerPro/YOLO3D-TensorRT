/*********************************************/
/*      代码用于程序运行时监控内存使用情况      */
/*      防止出现内存泄漏等情况使得程序奔溃      */
/*编译方式：g++ mem_monitor.cpp -o mem_monitor*/
/*        使用方式： ./mem_monitor pid        */
/*        设置时间为100秒一次进行内存获取       */
/*********************************************/

#include <atomic>
#include <iostream>
#include <signal.h>
#include <unistd.h>

std::atomic<bool> stopFLag(false);

void signalHandler(int signal) { stopFLag = true; }

int main(int argc, char *args[]) {
  if (argc != 2) {
    std::cout << "please input pid" << std::endl;
    return -1;
  }

  signal(SIGINT, signalHandler);

  char cmd[100];
  sprintf(cmd, "cat /proc/%s/status | grep VmRSS", args[1]);
  while (!stopFLag) {
    system(cmd);
    usleep(100 * 1000 * 1000); // get every 100s
  }
  return 0;
}
