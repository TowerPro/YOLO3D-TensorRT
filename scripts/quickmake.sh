#!/bin/bash

script_path=$(dirname "$0")
config_file="${script_path}/config.txt"

# 读取config.txt文件并提取参数
while IFS=':' read -r key value; do
  # 去除空格
  key=$(echo "$key" | tr -d '[:space:]')
  value=$(echo "$value" | tr -d '[:space:]')

  # 提取冒号后面的内容
  case $key in
    gccPath) C_COMPILER="$value" ;;
    g++Path) CXX_COMPILER="$value" ;;
    ip) ip="$value" ;;
    foldPath) foldPath="$value" ;;
    *) ;;
  esac
done < "$config_file"


# 编译
cd ./build
make clean
cmake -DCMAKE_C_COMPILER=$C_COMPILER -DCMAKE_CXX_COMPILER=$CXX_COMPILER ..
make -j12

# 通过scp把文件传到板子上
scp ./demo dnn@$ip:$foldPath
echo "!!!scp done!!!"