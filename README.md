# robot_parkingslot
This repo is for deployment of ParkingSlot Detection Algorithm [DeepPS](https://cslinzhang.github.io/deepps/).

## Environment:
### 1. Environment
Hardware: Nvidia AGX Xavier(ARM)  
Software: 
* JetPack 4.4
* CUDA 10.2
* TensorRT 7.1.3
* CUDNN 8.0.0
* onnx-tensorrt 7.2.1  

## Usage
* Model: Download onnx model from [model](https://pan.baidu.com/s/17fYGdUTsuCNzj5jjwp3FSQ)(code: w1bb) to models/  
run `sh trt.sh` to convert onnx to trt engine
* Run: `python3 src/main.py`
