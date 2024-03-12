#!/bin/bash

# 切换到代码目录

# 统计代码行数（排除 3rdparty 文件夹）
line_count=$(find . -type f -name '*.cpp' -or -name '*.h' -or -name '*.hpp' -or -name '*.py' | grep -v -E './3rdparty/|./opencv_with_gui|./mybuild|./build' | xargs cat | wc -l)

# 输出代码行数
echo "代码行数：$line_count"
