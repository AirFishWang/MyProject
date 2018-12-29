# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Description :
   Author :        wangchun
   date：          18-12-19
-------------------------------------------------
   Change Activity:
                   18-12-19:
-------------------------------------------------
"""
import os
import cv2
import random
import heapq
import keras
import time
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from data import num_classes
from data import voc2012_root_path
from data import read_file
from data import classes
from multiprocessing import Process, Pipe
from common import logger

pretrain_model = "pretrain_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
HEIGHT = 244
WIDTH = 244
BATCHSIZE = 32
NCLASS = num_classes
INIT_LR = 1e-3
EPOCHS = 100

import imgaug
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random
imgaug.seed(random.randint(0, 100))

# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True   # if set allow_growth, cannot use multiple gpu memory

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)


def get_rotate_keypoints(angle):
    def rotate(images, random_state, parents, hooks):
        dst_images = []
        for image in images:
            src_image = image
            h, w = src_image.shape[:2]
            center_x = w / 2
            center_y = h / 2
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - center_x
            M[1, 2] += (nH / 2) - center_y
            dst_image = cv2.warpAffine(src_image, M, (nW, nH), borderValue=(255, 255, 255))
            dst_images.append(dst_image)
        return dst_images

    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images
    return rotate, func_keypoints


rotate_45, func_keypoints = get_rotate_keypoints(45)
rotate_135, func_keypoints = get_rotate_keypoints(135)
rotate_225, func_keypoints = get_rotate_keypoints(225)
rotate_315, func_keypoints = get_rotate_keypoints(315)

seq = iaa.Sequential([
        # iaa.PerspectiveTransform(scale=0.02, keep_size=True),
        iaa.SomeOf((3, 4), [
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Add((-10, 10)),
            iaa.Sometimes(0.5, iaa.SomeOf(1, [
                iaa.Scale(0.5),
                iaa.Scale(0.8),
                iaa.Scale(2.0)
                ])
            ),
            iaa.Sometimes(0.5, iaa.SomeOf(1, [
                iaa.Lambda(func_images=rotate_45, func_keypoints=func_keypoints),
                iaa.Lambda(func_images=rotate_135, func_keypoints=func_keypoints),
                iaa.Lambda(func_images=rotate_225, func_keypoints=func_keypoints),
                iaa.Lambda(func_images=rotate_315, func_keypoints=func_keypoints),
                ])
            ),
            iaa.Sharpen(),
            iaa.SomeOf(1, [
                 iaa.Fliplr(1.0),
                 iaa.Flipud(1.0)
            ]),
            iaa.OneOf([
                iaa.AverageBlur(k=(1, 3)),
                iaa.MedianBlur(k=(1, 3)),
                iaa.GaussianBlur(sigma=iap.Uniform(0.8, 1.2)),
            ]),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)   # 改变对比度
        ])
    ], random_order=True)


def create_resnet50(nclass, train_phase=True):
    input = Input(shape=(HEIGHT, WIDTH, 3), name="img_input")
    base_model = ResNet50(input_tensor=input, include_top=False, weights=None)
    output = base_model.output

    output = Flatten()(output)
    if train_phase:
        output = Dropout(0.2)(output)
    output = Dense(nclass, activation='softmax', name='fc{}'.format(nclass))(output)
    model = Model(input, output, name='resnet50')

    return model


def generator(data_root, label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    all_labels = [x.strip() for x in lines if len(x.strip()) > 0]
    count = len(all_labels)
    while True:
        index = range(count)
        random.shuffle(index)
        groups = [[index[x % count] for x in range(i, i + BATCHSIZE)] for i in range(0, count, BATCHSIZE)]
        for g in groups:
            image_batch = np.zeros((BATCHSIZE, HEIGHT, HEIGHT, 3), dtype=np.float)
            label_batch = np.ones([BATCHSIZE, NCLASS], dtype=np.float)

            for i, x in enumerate(g):
                image_name, label = all_labels[x].split(" ")
                image = cv2.imread(os.path.join(data_root, image_name))
                if random.randint(0, 1) == 0:  # augment image in 50 percentage
                    image = seq.augment_image(image)
                image = cv2.resize(image, (WIDTH, HEIGHT))
                image = image.astype(np.float) / 255.0 - 0.5
                image_batch[i] = image
                label_batch[i] = keras.utils.to_categorical(int(label), NCLASS)
            yield image_batch, label_batch


def create_gen_process(data_root, num_process):
    def gen_data(child_side, data_root):
        while True:
            image_list = child_side.recv()  # process block
            if isinstance(image_list, str) and image_list == "stop":
                break
            list_size = len(image_list)

            image_batch = np.zeros((list_size, HEIGHT, HEIGHT, 3), dtype=np.float)
            label_batch = np.ones([list_size, NCLASS], dtype=np.float)

            for i, image_and_label in enumerate(image_list):
                image_name, label = image_and_label.split(" ")
                image = cv2.imread(os.path.join(data_root, image_name))
                if random.randint(0, 1) == 0:  # augment image in 50 percentage
                    image = seq.augment_image(image)
                image = cv2.resize(image, (WIDTH, HEIGHT))
                image = image.astype(np.float) / 255.0 - 0.5
                image_batch[i] = image
                label_batch[i] = keras.utils.to_categorical(int(label), NCLASS)

            child_side.send([image_batch, label_batch])

    process_worker = []
    for i in range(num_process):
        parent_side, child_side = Pipe()
        p = Process(target=gen_data, args=(child_side, data_root,))
        p.start()
        process_worker.append([parent_side, child_side, p])
    return process_worker


def generator_v2(data_root, label_file, num_process, process_list):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    all_labels = [x.strip() for x in lines if len(x.strip()) > 0]
    count = len(all_labels)

    # process_worker: [[parent_side, child_side, p], [parent_side, child_side, p], ...]
    process_worker = create_gen_process(data_root, num_process)
    for p in process_worker:
        process_list.append(p)

    while True:
        index = range(count)
        random.shuffle(index)
        groups = [[index[x % count] for x in range(i, i + BATCHSIZE)] for i in range(0, count, BATCHSIZE)]
        for g in groups:
            image_and_label = [all_labels[x] for x in g]
            # map
            for i in range(num_process):
                tmp_list = image_and_label[i::num_process]  # if length < count_son_process, the tmp_list = [], son process return []
                process_worker[i][0].send(tmp_list)
            recv_list = [process_worker[i][0].recv() for i in range(num_process)]  # wait for son process

            image_batch = tuple(recv_list[i][0] for i in range(num_process))
            label_batch = tuple(recv_list[i][1] for i in range(num_process))

            image_batch = np.concatenate(image_batch, axis=0)
            label_batch = np.concatenate(label_batch, axis=0)
            yield image_batch, label_batch


def stop_child_process(process_worker):
    for parent_side, child_side, p in process_worker:
        parent_side.send("stop")


def train():
    train_data_root = os.path.join(voc2012_root_path, "JPEGImages")
    train_data_label = "train.txt"

    test_data_root = os.path.join(voc2012_root_path, "JPEGImages")
    test_data_label = "test.txt"

    model = create_resnet50(NCLASS)
    logger.info("pretrain_model loading")
    model.load_weights(pretrain_model, by_name=True)
    logger.info("pretrain_model load finished")

    plot_model(model, show_shapes=True, to_file='resnet50.png')
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    models_save_path = "./models"
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)
    # checkpoint = ModelCheckpoint(filepath=os.path.join(models_save_path, 'voc2012-{epoch:02d}-{val_loss:.2f}.h5'),
    #                              monitor='val_loss',
    #                              save_best_only=True,
    #                              save_weights_only=True)

    checkpoint = ModelCheckpoint(filepath=os.path.join(models_save_path, 'voc2012-{epoch:02d}-{val_acc:.4f}.h5'),
                                 monitor='val_acc',
                                 mode='max',
                                 save_best_only=True,
                                 save_weights_only=True)

    use_multiprocessing = False
    if use_multiprocessing:
        train_process_worker = []
        test_process_worker = []
        train_loader = generator_v2(train_data_root, train_data_label, 2, train_process_worker)
        test_loader = generator_v2(test_data_root, test_data_label, 2, test_process_worker)
    else:
        train_loader = generator(train_data_root, train_data_label)
        test_loader = generator(test_data_root, test_data_label)

    train_count = 5717
    test_count = 5283

    logger.info('-----------Start training-----------')
    start = time.time()
    model.fit_generator(train_loader,
                        steps_per_epoch=train_count // BATCHSIZE,
                        epochs=EPOCHS,
                        initial_epoch=0,
                        validation_data=test_loader,
                        validation_steps=test_count // BATCHSIZE,
                        callbacks=[checkpoint])
    end = time.time()
    logger.info("train finished, cost time = {} hours".format((end - start)/3600.0))

    if use_multiprocessing:
        stop_child_process(train_process_worker+test_process_worker)


def test():
    model_path = "./models/voc2012-64-0.55.h5"
    model = create_resnet50(NCLASS)
    model.load_weights(model_path)

    test_data_root = os.path.join(voc2012_root_path, "JPEGImages")
    test_data_label = "test.txt"
    all_labels = read_file(test_data_label)
    count = len(all_labels)
    correct = 0
    top_k = 3
    for index, image_and_label in enumerate(all_labels):
        image_name, ground_truth = image_and_label.split(" ")
        image_path = os.path.join(test_data_root, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        image = image.astype(np.float) / 255.0 - 0.5
        image = np.expand_dims(image, axis=0)
        out = model.predict(image)
        out = out[0].tolist()
        labels = list(map(out.index, heapq.nlargest(top_k, out)))
        if int(ground_truth) in labels:
            correct += 1
        predict = " ".join([classes[x] for x in labels])
        logger.info("{}/{} {} ground_truth = {:<12} predict = {:<12}".format(index+1, count, image_name, classes[int(ground_truth)], predict))

    logger.info("accuracy = {}/{} = {}".format(correct, count, correct*1.0/count))


def data_augment_test():
    image_path = os.path.join(voc2012_root_path, "JPEGImages/2008_000008.jpg")
    src_image = cv2.imread(image_path)

    rotate, func_keypoints = get_rotate_keypoints(50)
    local_seq = iaa.Sequential([
        # iaa.Affine(rotate=random.randint(0, 360), cval=imgaug.ALL)
        iaa.Lambda(func_images=rotate, func_keypoints=func_keypoints)
    ])
    for i in range(100):
        dst_image = seq.augment_image(src_image)
        print dst_image.shape
        cv2.imshow("src_image", src_image)
        cv2.imshow("dst_image", dst_image)
        cv2.waitKey(1000)
    exit()


if __name__ == "__main__":
    # data_augment_test()
    train()
    test()