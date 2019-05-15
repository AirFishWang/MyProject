from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import keras.backend as K
from keras.models import load_model
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import numpy as np
import os

def Vgg16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


class Vgg16Frozen():
    def __init__(self):

        with gfile.FastGFile("vgg16.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x, self.y = tf.import_graph_def(graph_def=graph_def,
                                                 name="",
                                                 return_elements=["zero_padding2d_1_input:0", "dense_3/Softmax:0"])

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, use_per_session_threads=True)
            self.sess = tf.Session(config=config)

    def predict(self, x):
        return self.sess.run(self.y, feed_dict={self.x: x})


def frozen_model():
    K.set_learning_phase(0)
    xnet = Vgg16('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    print ('input is:', [i.op.name for i in xnet.inputs])
    print ('output is:', [i.op.name for i in xnet.outputs])

    sess = K.get_session()
    graph = sess.graph
    with graph.as_default():
        input_graph_def = graph.as_graph_def()
        frozen_graph = convert_variables_to_constants(sess, input_graph_def, ['dense_3/Softmax'])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    graph_io.write_graph(frozen_graph, './', "vgg16.pb", as_text=False)
    exit()


if __name__ == "__main__":
    # frozen_model()

    xnet = Vgg16()
    print ('input is:', [i.op.name for i in xnet.inputs])
    print ('output is:', [i.op.name for i in xnet.outputs])

    obj = Vgg16Frozen()
    imgfile = "test_images/cat.jpeg"
    # BGR
    im = cv2.resize(cv2.imread(imgfile), (224, 224)).astype(np.float32)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = im.reshape((1, 224, 224, 3))
    out = obj.predict(im)
    print np.argmax(out)
    pass
