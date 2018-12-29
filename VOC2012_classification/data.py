# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_preprocess
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
import xml.etree.ElementTree as ET

num_classes = 20
classes = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
class_to_index = dict(zip(classes, xrange(num_classes)))


voc2012_root_path = "/home/wangchun/Desktop/VOC2012/VOCdevkit/VOC2012"

train_file = os.path.join(voc2012_root_path, "ImageSets/Main/train.txt")
test_file = os.path.join(voc2012_root_path, "ImageSets/Main/val.txt")
xml_dir = os.path.join(voc2012_root_path, "Annotations")


def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines if len(x.strip()) > 0]
    return lines


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


def get_label(xml_root_dir, annotations_file, output_label):
    def select_main_object(objects):
        count = len(objects)
        if count == 0:
            raise Exception("no object")
        elif count == 1:
            return objects[0]['name']
        else:
            max_area = -1
            class_name = ""
            for obj in objects:
                bbox = obj['bbox']
                area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                if area > max_area:
                    max_area = area
                    class_name = obj['name']
        return class_name

    image_names = read_file(annotations_file)
    count = len(image_names)
    with open(output_label, 'w') as f:
        for index, image_name in enumerate(image_names):
            xml = os.path.join(xml_root_dir, image_name+".xml")
            objects = parse_rec(xml)
            class_name = select_main_object(objects)
            f.write("{} {}\n".format(image_name+".jpg", class_to_index[class_name]))
            print "{}/{} {} {:2} {}".format(index+1, count, image_name, class_to_index[class_name], class_name)


def show_image():
    image_path = os.path.join(voc2012_root_path, "JPEGImages/2008_000008.jpg")
    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    cv2.waitKey(0)


def label_count(label_file):
    all_laebl = read_file(label_file)
    count_dict = {}
    for x in all_laebl:
        image_name, label = x.split(" ")
        class_name = classes[int(label)]
        if class_name in count_dict:
            count_dict[class_name] += 1
        else:
            count_dict[class_name] = 0

    print "{} all_count = {}".format(label_file, len(all_laebl))
    for c in classes:
        print "{} {}".format(c, count_dict[c])


if __name__ == "__main__":
    # show_image()
    # get_label(xml_dir, train_file, "train.txt")
    # get_label(xml_dir, test_file, "test.txt")
    label_count("train.txt")
    label_count("test.txt")