# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data
   Description :
   Author :        wangchun
   date：          18-12-27
-------------------------------------------------
   Change Activity:
                   18-12-27:
-------------------------------------------------
"""
import os
import cv2
import keras
import random
import shutil
import numpy as np

root_path = "/home/wangchun/Desktop/DataSet/word_images"
HEIGHT = 64
WIDTH = 128
NCLASS = 2    # chi: 0   num: 1
BATCHSIZE = 64



def select_sample(label_path, class_index, test_count=800):

    with open(label_path, 'r') as f:
        lines = f.readlines()
        label = [x.strip().replace("\\", '/').lstrip("/") for x in lines if len(x.strip()) > 0]
    random.shuffle(label)
    test_label = []
    for index, x in enumerate(label[0:test_count]):
        image_name = x.split("/")[-1]
        src_path = os.path.join(root_path, x)
        dst_path = os.path.join(root_path, 'test', image_name)
        shutil.copyfile(src_path, dst_path)
        test_label.append('{} {}\n'.format(image_name, class_index))
        print "{}/{} copy {} to test".format(index+1, test_count, image_name)

    train_label = []
    for index, x in enumerate(label[test_count:]):
        image_name = os.path.split(x)[-1]
        src_path = os.path.join(root_path, x)
        dst_path = os.path.join(root_path, 'train', image_name)
        shutil.copyfile(src_path, dst_path)
        train_label.append('{} {}\n'.format(image_name, class_index))
        print "{}/{} copy {} to train".format(index+1, len(label)-test_count, image_name)
    return train_label, test_label


def load_label(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    labels = [x.strip() for x in lines if len(x.strip()) > 0]
    return labels


def random_crop_image(src_image):
    h, w = src_image.shape[:2]
    resize_image = cv2.resize(src_image, (int(w * HEIGHT / h), HEIGHT))
    h, w = resize_image.shape[:2]
    if w > WIDTH:
        left = random.randint(0, w - WIDTH)
        right = left + WIDTH
        crop_image = resize_image[:, left:right]
    else:
        crop_image = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        left = random.randint(0, WIDTH-w)
        right = left + w
        crop_image[:, left:right] = resize_image
    return crop_image


def generator(data_root, label_path, batch_size):
    labels = load_label(label_path)
    count = len(labels)
    while True:
        index = range(count)
        random.shuffle(index)
        groups = [[index[x % count] for x in range(i, i + batch_size)] for i in range(0, count, batch_size)]
        for g in groups:
            image_batch = np.zeros((batch_size, HEIGHT, WIDTH, 3), dtype=np.float)
            label_batch = np.ones([batch_size, NCLASS], dtype=np.float)

            for i, x in enumerate(g):
                image_name, label = labels[x].split(" ")
                image = cv2.imread(os.path.join(data_root, image_name))
                image = random_crop_image(image)
                # cv2.imshow("crop", image)
                # cv2.waitKey(500)
                image = image.astype(np.float) / 255.0 - 0.5
                image_batch[i] = image
                label_batch[i] = keras.utils.to_categorical(int(label), NCLASS)
            yield image_batch, label_batch


def generator_v2(data_root, label_path, batch_size):        # read all image at once
    labels = load_label(label_path)
    count = len(labels)
    all_images = []
    print "load images in {} start".format(label_path)
    for x in labels:
        image_name, label = x.split(" ")
        image = cv2.imread(os.path.join(data_root, image_name))
        all_images.append([image, int(label)])
    print "load images in {} finish".format(label_path)
    test_tmp_count = 1000
    while True:
        index = range(count)
        random.shuffle(index)
        groups = [[index[x % count] for x in range(i, i + batch_size)] for i in range(0, count, batch_size)]
        for g in groups:
            image_batch = np.zeros((batch_size, HEIGHT, WIDTH, 3), dtype=np.float)
            label_batch = np.ones([batch_size, NCLASS], dtype=np.float)
            for i, x in enumerate(g):
                image = all_images[x][0]
                label = all_images[x][1]
                image = random_crop_image(image)
                if test_tmp_count > 0:
                    if label == 0:
                        cv2.imwrite("./chi/{}.png".format(test_tmp_count), image)
                    else:
                        cv2.imwrite("./num/{}.png".format(test_tmp_count), image)
                    test_tmp_count -= 1
                # cv2.imshow("crop", image)
                # cv2.waitKey(0)
                image = image.astype(np.float) / 255.0 - 0.5
                image_batch[i] = image
                label_batch[i] = keras.utils.to_categorical(label, NCLASS)
            yield image_batch, label_batch


if __name__ == "__main__":
    # src_image = cv2.imread("/home/wangchun/Desktop/DataSet/word_images/test/9247735280436670782.png")
    # for i in range(100):
    #     dst_image = random_crop_image(src_image)
    #     cv2.imshow("src", src_image)
    #     cv2.imshow("dst", dst_image)
    #     print dst_image.shape
    #     cv2.waitKey(300)

    # chi_train_label, chi_test_label = select_sample(os.path.join(root_path, 'chi.txt'), 0)
    # num_train_label, num_test_label = select_sample(os.path.join(root_path, 'num.txt'), 1)
    # with open("train.txt", 'w') as f:
    #     f.writelines(chi_train_label+num_train_label)
    #
    # with open("test.txt", 'w') as f:
    #     f.writelines(chi_test_label+num_test_label)

    test_dir = os.path.join(root_path, "test")
    test_label = os.path.join(root_path, "test.txt")
    gen = generator_v2(test_dir, test_label, BATCHSIZE)
    for i in range(10):
        images, labels = gen.next()