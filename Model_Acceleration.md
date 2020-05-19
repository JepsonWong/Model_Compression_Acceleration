# Model_Acceleration

# Optimization method

## Automatic Mixed Precision (AMP)

# Optimization tools and libraries

## TensorRT (Nvidia)

Only for inference, 40x faster than CPU-only platforms during inference.

Optimizations:

* Weight & Activation Precision Calibration: TensorRT provides INT8 and FP16 optimizations for production deployments of deep learning inference applications
* Layer & Tensor Fusion: Optimizes use of GPU memory and bandwidth by fusing nodes in a kernel.
* Kernel Auto-Tuning: Selects best data layers and algorithms based on target GPU platform.
* Dynamic Tensor Memory: Minimizes memory footprint and re-uses memory for tensors efficiently.
* Multi-Stream Execution: Scalable design to process multiple input streams in parallel.

Useage:

* For Caffe and TensorFlow trained models, if the included operations are all supported by TensorRT, they can be directly optimized and reconstructed by TensorRT.
* For models trained by MXnet, PyTorch or other frameworks, if the operations included are all supported by TensorRT, you can use the TensorRT API to rebuild the network structure and indirectly optimize the reconstruction.
* After the models trained by other frameworks are converted to ONNX intermediate format, if the included operations are supported by TensorRT, the TensorRT-ONNX interface can be used to optimize.
* If the trained network model includes operations not supported by TensorRT:
  * TensorFlow models can be converted through `` tf.contrib.tensorrt``, and unsupported operations will be reserved as TensorFlow computing nodes; MXNet also supports similar conversion methods for computing graphs.
  * Unsupported operations can be customized through Plugin API and added to TensorRT calculation graph.
  * Divide the deep network into two parts, some of the operations included are supported by TensorRT and can be converted into TensorRT calculation graphs. The other part is implemented by other frameworks, such as MXnet or PyTorch, and it is recommended to use the C ++ API to ensure a more efficient runtime.

Reference:

* [tensorrt](https://developer.nvidia.com/tensorrt)
* [TensorRT为什么能让模型跑快快](https://zhuanlan.zhihu.com/p/64933639)
* [TensorRT加速tensorflow模型](https://zhuanlan.zhihu.com/p/105650617)
* [TensorRT(1)-介绍-使用-安装](https://arleyzhang.github.io/articles/7f4b25ce/)
* [高性能深度学习支持引擎实战——TensorRT](https://zhuanlan.zhihu.com/p/35657027)
* [TensorRT概述](https://zhuanlan.zhihu.com/p/138941917)
* [利用TensorRT对深度学习进行加速](https://oldpan.me/archives/use-tensorrt-speed-up-deep-learning-1) [利用TensorRT实现神经网络提速(读取ONNX模型并运行)](https://oldpan.me/archives/tensorrt-code-toturial-1)
* https://github.com/mileistone/study_resources/blob/master/engineering/tensorrt/tensorrt.md

## TVM (Tensor Virtual Machine)

## Tensor Comprehension (Facebook)

## Distiller (Intel)

