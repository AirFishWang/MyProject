(1)参考 tensorRT官方demo， tensorRT安装后，在根目录下的sample/python下有提供的demo，densenet_trtrt参考的是end_to_end_tensorflow_mnist
(2)一般步骤：
   模型训练
   模型保存(保存为pb模型， 如果是keras模型，转换为pb模型)
   模型转换(使用convert-to-uff命令工具转换为uff模型， convert-to-uff **.pb)
   构建tftrt引擎执行推理
