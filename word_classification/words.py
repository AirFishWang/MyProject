# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     words
   Description :
   Author :        wangchun
   date：          18-12-27
-------------------------------------------------
   Change Activity:
                   18-12-27:
-------------------------------------------------
    reference: https://www.cnblogs.com/skyfsm/p/8051705.html
"""


import os
import time
import cv2
import shutil
import numpy as np
import tensorflow as tf
from squeezenet import SqueezeNet
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Convolution2D, Activation
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.utils import plot_model
from data import HEIGHT, WIDTH, BATCHSIZE, NCLASS, root_path, generator_v2, load_label

INIT_LR = 1e-4
EPOCHS = 50

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)


def create_squeezenet(nclass, train=True):
    input = Input(shape=(HEIGHT, WIDTH, 3), name="img_input")
    base_model = SqueezeNet(include_top=False, weights=None, input_tensor=input)
    x = base_model.output

    if train:
        base_model.load_weights("models/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5")
        x = Dropout(0.2, name='drop9')(x)
    x = Convolution2D(nclass, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)
    model = Model(input, x, name='squeezenet')
    return model


def train():
    train_dir = os.path.join(root_path, "train")
    train_label = os.path.join(root_path, "train.txt")
    test_dir = os.path.join(root_path, "test")
    test_label = os.path.join(root_path, "test.txt")

    train_generator = generator_v2(train_dir, train_label, BATCHSIZE)
    test_generator = generator_v2(test_dir, test_label, BATCHSIZE)

    model = create_squeezenet(NCLASS)
    plot_model(model, show_shapes=True, to_file='SqueezeNet.png')
    opt = Adam(lr=INIT_LR, decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    models_save_path = "./models"
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)

    checkpoint = ModelCheckpoint(filepath=os.path.join(models_save_path, 'word-{epoch:02d}-{val_acc:.4f}.h5'),
                                 monitor='val_acc',
                                 mode='max',
                                 save_best_only=True,
                                 save_weights_only=True)
    train_count = 9605
    test_count = 1600

    print '-----------Start training-----------'
    start = time.time()
    model.fit_generator(train_generator,
                        steps_per_epoch=train_count // BATCHSIZE,
                        epochs=EPOCHS,
                        initial_epoch=0,
                        validation_data=test_generator,
                        validation_steps=test_count // BATCHSIZE,
                        callbacks=[checkpoint])
    end = time.time()
    print "train finished, cost time = {} hours".format((end - start) / 3600.0)


def predict(model, src_image):
    """
    :param model: a classification model have loaded weight
    :param src_image: a bgr image
    :return:
    """
    h, w = src_image.shape[:2]
    width = int(w * HEIGHT / h)
    if width <= 0:  # the width is too small
        return 0
    resize_image = cv2.resize(src_image, (width, HEIGHT))
    h, w = resize_image.shape[:2]

    if w <= WIDTH:
        crop_image = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        left = (WIDTH - w)/2
        right = left + w
        crop_image[:, left:right] = resize_image
        image = crop_image.astype(np.float) / 255.0 - 0.5
        image = np.expand_dims(image, axis=0)
        out = model.predict(image)
        out = out[0].tolist()
        label = np.argmax(out)
        return label
    else:
        label_0 = 0
        label_1 = 0
        for i in range(3):
            if i == 1:
                left = 0
            elif i == 2:
                left = (w-WIDTH) / 2
            else:
                left = w - WIDTH
            right = left + WIDTH
            crop_image = resize_image[:, left:right]
            image = crop_image.astype(np.float) / 255.0 - 0.5
            image = np.expand_dims(image, axis=0)
            out = model.predict(image)
            out = out[0].tolist()
            label = np.argmax(out)

            if label == 0:
                label_0 += 1
            else:
                label_1 += 1
        return 0 if label_0 > label_1 else 1


def test_one_image():
    image_path = "error/9790140683116556062.png"

    model_path = "models/word-39-0.9962.h5"
    model = create_squeezenet(2, train=False)
    model.load_weights(model_path)
    src_image = cv2.imread(image_path)
    label = predict(model, src_image)
    print "label = {}".format(label)


def test():

    test_dir = os.path.join(root_path, "train")
    test_label_path = os.path.join(root_path, "train.txt")

    model_path = "models/word-39-0.9962.h5"
    test_labels = load_label(test_label_path)

    error_dir = "error"
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    model = create_squeezenet(2, train=False)
    model.load_weights(model_path)
    count = len(test_labels)
    correct = 0
    all_time = 0
    confusion_matrix = np.zeros((NCLASS, NCLASS), dtype=np.int16)
    for index, x in enumerate(test_labels):
        image_name, label = x.split(" ")
        src_image = cv2.imread(os.path.join(test_dir, image_name))
        start = time.time()
        p_label = predict(model, src_image)
        end = time.time()
        all_time += end - start
        if int(label) == p_label:
            correct += 1
            print "{}/{} {}  ground_truth = {}    predict={}".format(index, count, image_name, int(label), p_label)
        else:
            print "{}/{} {}  ground_truth = {}    predict={}   error".format(index, count, image_name, int(label), p_label)
            shutil.copyfile(os.path.join(test_dir, image_name), os.path.join(error_dir, image_name))
        confusion_matrix[int(label), p_label] += 1

    print "confusion_matrix = ", confusion_matrix
    # calculate recall and precision according confusion matrix
    for i in range(NCLASS):
        recall = confusion_matrix[i, i] * 1.0 / np.sum(confusion_matrix[i, :])
        precision = confusion_matrix[i, i] * 1.0 / np.sum(confusion_matrix[:, i])
        print "class {}   recall = {}   precision = {}".format(i, recall, precision)

    print "accuracy = {}/{} = {}".format(correct, count, correct*1.0/count)
    print "average time = {}".format(all_time/count)
    print "test finish"


if __name__ == "__main__":
    # train()
    test()
    #test_one_image()