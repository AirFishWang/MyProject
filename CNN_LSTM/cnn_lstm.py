# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cnn_lstm
   Description :
   Author :        wangchun
   date：          2019/6/4
-------------------------------------------------
   Change Activity:
                   2019/6/4
-------------------------------------------------
"""
frames = 7
channels = 3
rows = 224
columns = 224
classes = 5
batch_size = 8

import os
import keras
import random
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np


K.set_image_data_format('channels_first')
def create_model():
    # reference : https://riptutorial.com/keras/example/29812/vgg-16-cnn-and-lstm-for-video-classification
    video = Input(shape=(frames,
                         channels,
                         rows,
                         columns))
    cnn_base = VGG16(input_shape=(channels,
                                  rows,
                                  columns),
                     weights="imagenet",
                     include_top=False)
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(input=cnn_base.input, output=cnn_out)
    # cnn.trainable = True
    # cnn.trainable = False  # 默认的是True，需要设置为False时，这一句没有作用，需要将每一层都设置为False
    for layer in cnn.layers:
        layer.trainable = False

    cnn.summary()

    encoded_frames = TimeDistributed(cnn)(video)
    encoded_sequence = LSTM(256)(encoded_frames)
    hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
    outputs = Dense(output_dim=classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)
    optimizer = Nadam(lr=0.002,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      schedule_decay=0.004)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["categorical_accuracy"])
    return model


def generate():
    while True:
        batch_x = np.random.random((batch_size, frames, channels, rows, columns))
        batch_label = np.ones([batch_size, classes], dtype=np.float)
        for i in range(batch_size):
            label = random.randint(0, classes-1)  # [a, b]
            batch_label[i] = keras.utils.to_categorical(label, classes)
        yield batch_x, batch_label


if __name__ == "__main__":
    model = create_model()
    model.summary()
    plot_model(model, to_file='cnn_lstm_true.png', show_shapes=True, show_layer_names=True)

    train_gen = generate()
    valid_gen = generate()

    models_save_path = "snapshot_model"
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)
    checkpoint = ModelCheckpoint(filepath=os.path.join(models_save_path, 'cnn_lstm-{epoch:02d}-{val_loss:.4f}.h5'),
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=False,
                                 save_weights_only=False)
    model.fit_generator(train_gen,
                        steps_per_epoch=10,
                        epochs=15,
                        initial_epoch=0,
                        validation_data=valid_gen,
                        validation_steps=20,
                        callbacks=[checkpoint])

