# yolov5-trt
Minimal code for YOLO v5 inference with TensorRT (C++).

## Prerequisites

To build and run this program, the following items are required.
- a CUDA compatible GPU card
- NVIDIA driver: 470.82.01 or higher
- NVIDIA CUDA: 11.4.3
- NVIDIA cuDNN: 8.2.1
- NVIDIA TensorRT: 8.2.5.1
- OpenCV: 4.5.4
- cmake: 3.16 or higher
- an exported yolov5s ONNX model file

## Install

Clone this repo:

```bash
git clone https://github.com/yinguobing/yolov5-trt.git
```

## Build

Build with cmake:

```bash
cd yolov5-trt
mkdir build && cd build
cmake ..
make
```

## Run

Run inference with a sample image:

```bash
yolov5 yolov5s.engine input.jpg
```

.. then checkout the output image.

## Authors
Yin Guobing (尹国冰) - [yinguobing](https://yinguobing.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* https://github.com/cyrusbehr/tensorrt-cpp-api
