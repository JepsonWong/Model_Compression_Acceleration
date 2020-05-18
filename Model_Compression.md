# Model_Compression

## Pruning

The method of parameter pruning and sharing focuses on exploring redundant parts of the model parameters, and attempts to remove redundant and unimportant parameters.

channel pruning、deep compression、network slimming

## Quantization

Quantization is also focused on exploring redundant parts of model parameters and trying to remove redundant and unimportant parameters.

Principle of quantification:

* Quantization formula from floating-point to fixed-point, Unquantization formula from fixed-point to floating-point
* Unsaturated/Saturated quantizer, Kullback–Leibler divergence (KLD)
* Quantization, Unquantization, Requantize 

Reference:

* [Tensorflow模型量化(Quantization)原理及其实现方法](https://zhuanlan.zhihu.com/p/79744430)
* [fixed_point_quantization.md](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/quantization/fixed_point_quantization.md)
* https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/slim/quantization/README.md
* [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://blog.csdn.net/guvcolie/article/details/81286349)
* [模型量化技术原理及其发展现状和展望](https://zhuanlan.zhihu.com/p/141641433)
* [NCNN Conv量化详解](https://zhuanlan.zhihu.com/p/71881443)
* [神经网络量化方法](https://murphypei.github.io/blog/2019/11/neural-network-quantization)
* [Int8量化-介绍](https://zhuanlan.zhihu.com/p/58182172)
* [神经网络量化简介](https://zhuanlan.zhihu.com/p/64744154)
* [PaddleSlim](https://paddlepaddle.github.io/PaddleSlim/algo/algo.html#1-quantization-aware-training%E9%87%8F%E5%8C%96%E4%BB%8B%E7%BB%8D)
* [quantization](https://github.com/Ewenwan/MVision/blob/master/CNN/Deep_Compression/quantization/readme.md)
* Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

### 8bit Quantization

* Post-training quantization: offline
* Quantization-aware training: online

QA:

* [why is there different latency for post-training quantization and quantization-aware trained models?](https://github.com/tensorflow/tensorflow/issues/24744)

Tools:

* [Tensorflow Lite](https://www.tensorflow.org/lite/performance/model_optimization)
  * Post-training quantization: dynamic range quantization, float16 quantization, integer quantization
    * dynamic range quantization: Accuracy loss, 4x smaller, 2-3x speedup in CPU latency. Quantization statically **quantizes only the weights from floating point to integer**. At inference, weights are converted from 8-bits of precision to floating point and computed using floating-point kernels. **To further improve latency**, dynamically quantize activations based on their range to 8-bits and perform computations with 8-bit weights and activations.
    * integer quantization: Smaller accuracy loss, 4x smaller, 3x+ speedup in CPU latency. Quantizes all weights and activations during. At inference, compute using fixed-point kernels.
    * float16 quantization: Insignificant accuracy loss, 2x smaller, potential GPU acceleration. Quantizing the weights to float16. **By default, a float16 quantized model will "dequantize" the weights values to float32 when run on the CPU.**
  * Quantization-aware training: Smallest accuracy loss, 4x smaller, 1.5-4x speedup in CPU latency. The operation of quantization and dequantization is added to the forward propagation process of traditional neural networks, which undoubtedly introduces the accuracy error brought by quantization. I personally understand that the purpose of quantization training is to **adapt the parameter distribution to this error**.
    * scale factor
      * Dynamic strategy: For weight.
      * Static strategy: For activation. Calculate the average/maximum/moving average value of the maximum absolute value of activation in one window.

## Knowledge Distillation (KD)

Distillation loss:

* teacher network output's logits
* intermediate hidden layer parameter value: FitNet
* intermediate hidden layer output value: FSP

Reference:

* [从入门到放弃：深度学习中的模型蒸馏技术](https://zhuanlan.zhihu.com/p/93287223)

## Neural Architecture Search (NAS)

