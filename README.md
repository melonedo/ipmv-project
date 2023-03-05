# ipmv-project

图像处理与机器视觉小组项目，实现[A non-local cost aggregation method for stereo matching](https://ieeexplore.ieee.org/document/6247827)。

## 编译

首先下载[middlebury 数据集](https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip)并解压到`data/`，并安装OpenCV 3.4.15、Eigen 3.3.9。

```shell
cmake -B build -D OpenCV_ROOT=XXX -D Eigen_INCLUDE_DIR=XXX
cmake --build build
```

## 运行

```shell
cmake --build build -t copy-dependency --config release # 复制dll，也可以手动复制
cmake --build build -t run --config release
```
