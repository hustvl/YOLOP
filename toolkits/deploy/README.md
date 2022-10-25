YoloP TensorRT Usage(简要说明)
=====


## 1. 准备构建环境 | Prepare building environments

Make sure you have install `c++`(support c++11)、 `cmake`、`opencv`(4.x)、`cuda`(10.x)、`nvinfer`(7.x).

Now, zedcam is not necessary!

And we can also use opencv that built without cuda.

## 2. 编译 | build

Go to `YOLOP/toolkits/deploy`.

```
mkdir build
cd build

cmake ..
make
```

Now you can get `yolov5` and `libmyplugins.so`.
If you have Zed installed, you can get `yolop`.


## 3. 生成和测试 trt | Generate and test trt

Go to build dir (`YOLOP/toolkits/deploy/build`).

### 3.1 gen wts
```
python3 ../gen_wts.py
```

### 3.2 gen trt
```
./yolov5 -s yolop.wts  yolop.trt s
```

### 3.3 test trt
```
mkdir ../results
./yolov5 -d yolop.trt  ../../../inference/images/
```

It will output like as follow if successful! (`Jetson Xavier NX - Jetpack 4.4`)
```
1601ms
26ms
26ms
26ms
26ms
28ms
```

![](build/results/_3c0e7240-96e390d2.jpg)




