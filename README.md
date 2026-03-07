# 项目简介
使用 OpenCV 进行裂缝自动检测，计算裂缝长度和宽度，生成检测报告。

# 技术栈
Python 3.9
OpenCV 4.x
NumPy
Git

# 功能
图像预处理（灰度化、滤波、边缘检测）
裂缝轮廓提取
自适应+形态学操作
裂缝长度/宽度计算
批量处理多张图片
自动生成检测报告

# 运行方法
1. 安装依赖
pip install opencv-python numpy
2. 把裂缝照片放到 images/ 文件夹
3. 运行
python crack_test.py
