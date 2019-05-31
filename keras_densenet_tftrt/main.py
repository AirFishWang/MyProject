# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :        wangchun
   date：          2019/5/6
-------------------------------------------------
   Change Activity:
                   2019/5/6:
-------------------------------------------------
"""
import keras
import time
import cv2
import uff
import common
import numpy as np
import keras.backend as K
from keras.utils import plot_model
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.models import load_model
from keras.preprocessing import image
# from tensorflow.contrib import tensorrt as tftrt
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.applications.densenet import DenseNet169
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorrt as trt
import pycuda.autoinit


warm_up = 10
step = 100


def keras_test():
    # model = DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,classes=1000)
    # plot_model(model, show_shapes=True, to_file='DenseNet169.png')
    # model.save("DenseNet169.h5")
    model = load_model("DenseNet169.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    for i in range(warm_up):
        preds = model.predict(x)
    t0 = time.time()
    for i in range(step):
        preds = model.predict(x)
    t1 = time.time()
    # 将结果解码为元组列表 (class, description, probability)
    # (一个列表代表批次中的一个样本）
    print('Predicted:', decode_predictions(preds, top=3)[0])
    print "keras time repeat {}: {}".format(step, t1-t0)
    # exit()


def convert_keras_to_pb():
    K.set_learning_phase(0)
    model = load_model("DenseNet169.h5")
    print ('input is:', [i.op.name for i in model.inputs])
    # change for multipy outputs
    print ('output is:', [i.op.name for i in model.outputs])
    sess = K.get_session()
    graph = sess.graph
    with graph.as_default():
        input_graph_def = graph.as_graph_def()
        frozen_graph = convert_variables_to_constants(sess, input_graph_def, ["fc1000/Softmax"])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    graph_io.write_graph(frozen_graph, './', "DenseNet169.pb", as_text=False)
    exit()


def tensorflow_test():
    pb_file = "DenseNet169.pb"
    with tf.Session() as sess:
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        x_tensor, y_tensor = tf.import_graph_def(graph_def=graph_def, name="", return_elements=[u"input_1:0", u"fc1000/Softmax:0"])

        img_path = 'elephant.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        for i in range(warm_up):
            y = sess.run(y_tensor, feed_dict={x_tensor: x})
        t0 = time.time()
        for i in range(step):
            y = sess.run(y_tensor, feed_dict={x_tensor: x})
        t1 = time.time()
        print "y.shape = ", y.shape
        print('Predicted:', decode_predictions(y, top=3)[0])
        print "tensorflow time repeat {}: {}".format(step, t1 - t0)

    exit()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_batch_size = 1
def build_engine(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_batch_size = trt_batch_size
        builder.max_workspace_size = common.GiB(1)  # Workspace size是builder在构建engine时候最大可以使用的内存大小，其越高越好
        # parser the UFF Nerwork
        parser.register_input('input_1', (3, 224, 224))
        parser.register_output("fc1000/Softmax")
        parser.parse(model_file, network)  # 载入模型，解析，填充tensorRT的network
        print "build engine..."
        return builder.build_cuda_engine(network)  # build network


def tensorrt_test():
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.transpose(x, (0, 3, 1, 2))

    batch_x = np.tile(x, (trt_batch_size, 1, 1, 1))

    with build_engine("DenseNet169.uff") as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            # np.copyto(inputs[0].host, x.ravel())
            np.copyto(inputs[0].host, batch_x.ravel())
            [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=trt_batch_size)
            for i in range(warm_up):
                [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=trt_batch_size)
            t0 = time.time()
            for i in range(step):
                [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=trt_batch_size)
            t1 = time.time()
            print 'Predicted:', decode_predictions(output.reshape(trt_batch_size, 1000), top=3)
            print "tensorrt time repeat {}: {}".format(step, t1 - t0)


def repeat_channel_test():
    gray_img = cv2.imread('elephant.jpg', 0)
    bgr_img = np.tile(np.expand_dims(gray_img, axis=-1), (1, 1, 3))
    batch_img = np.tile(np.expand_dims(bgr_img, axis=0), (10, 1, 1, 1))

    cv2.imshow("gray_img", gray_img)
    cv2.imshow("bgr_img", bgr_img)
    cv2.imshow("batch_img[0]", batch_img[0])
    cv2.waitKey(0)
    exit()


if __name__ == "__main__":
    # repeat_channel_test()

    # convert_keras_to_pb()
    # keras_test()
    # tensorflow_test()
    tensorrt_test()
    pass

