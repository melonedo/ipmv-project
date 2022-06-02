# ipmv-project

## 编译

首先下载[middlebury 数据集](https://vision.middlebury.edu/stereo/data/scenes2021/zip/all.zip)并解压到`data/`，并安装OpenCV 3.4.15、Eigen 3.3.9。

```shell
cmake -B build -D OpenCV_ROOT=XXX -D Eigen_INCLUDE_DIR=XXX
cmake --build build
```

## 运行

```shell
cmake --build build -t copy-dependency
cmake --build build -t run
```
