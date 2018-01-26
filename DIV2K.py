#!/usr/bin/env python

# encoding: utf-8
import os
import tensorflow as tf
import numpy as np
import PIL
import scipy
import cv2
import random
import skimage


class Dataset():
    pass


class DIV2K(Dataset):
    def __init__(self):
        self.DIV2K_path = '/fds/sr/dataset/DIV2K'
        self.DIV2K_train_HR_path = os.path.join(self.DIV2K_path, 'DIV2K_train_HR')
        self.DIV2K_train_LR_path = os.path.join(self.DIV2K_path, 'DIV2K_train_LR_bicubic/X2')

        self.DIV2K_valid_HR_path = os.path.join(self.DIV2K_path, 'DIV2K_valid_HR')
        self.DIV2K_valid_LR_path = os.path.join(self.DIV2K_path, 'DIV2K_valid_LR_bicubic/X2')

        self.DIV2K_train_HR_filename_list = list(map(lambda x: os.path.join(self.DIV2K_train_HR_path, str(x) + '.png'), ['{:0>4}'.format(i + 1) for i in range(800)]))
        self.DIV2K_train_LR_filename_list = list(map(lambda x: os.path.join(self.DIV2K_train_LR_path, str(x) + '.png'), ['{:0>4}x2'.format(i + 1) for i in range(800)]))


        self.DIV2K_train_HR_filename_list = list(map(lambda x: os.path.join(self.DIV2K_train_HR_path, str(x) + '.png'), ['{:0>4}'.format(i + 1) for i in range(800)]))
        self.DIV2K_train_LR_filename_list = list(map(lambda x: os.path.join(self.DIV2K_train_LR_path, str(x) + '.png'), ['{:0>4}x2'.format(i + 1) for i in range(800)]))

    def __len__(self):
        return len(self.DIV2K_train_HR_filename_list)


    def __getitem__(self, item):
        return [self.DIV2K_train_LR_filename_list[item], self.DIV2K_train_HR_filename_list[item]]


    def get_test(self):
        lr_images = map(scipy.misc.imread, self.DIV2K_train_LR_filename_list[0:10])
        hr_images = map(scipy.misc.imread, self.DIV2K_train_HR_filename_list[0:10])
        return [lr_images, hr_images]


    def generate_div2k_tfrecord(self):
        writer = tf.python_io.TFRecordWriter(os.path.join(self.DIV2K_path, 'DIV2K.tfrecords'))
        for index in range(len(self)):
            # img=PIL.Image.open(self.DIV2K_train_LR_filename_list[index])
            # print(type(img))
            # img= img.resize((128,128))
            # img_raw=img.tobytes()
            # print(type(img), type(img_raw))
            input = PIL.Image.open(self.DIV2K_train_LR_filename_list[index])
            label = PIL.Image.open(self.DIV2K_train_HR_filename_list[index])
            # print(type(input), type(input.tobytes()))
            height = input.height
            width = input.width
            # print(input.shape)
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input.tobytes()])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
            }))
            writer.write(example.SerializeToString())
        writer.close()


# class DIV2K_test(Dataset):
#     def __init__(self):
#         self.DIV2K_path = '/fds/sr/dataset/DIV2K'
#         self.DIV2K_train_HR_path = os.path.join(self.DIV2K_path, 'DIV2K_train_HR')
#         self.DIV2K_train_LR_path = os.path.join(self.DIV2K_path, 'DIV2K_train_LR_bicubic/X4')
#
#         self.DIV2K_valid_HR_path = os.path.join(self.DIV2K_path, 'DIV2K_valid_HR')
#         self.DIV2K_valid_LR_path = os.path.join(self.DIV2K_path, 'DIV2K_valid_LR_bicubic/X4')
#
#         self.DIV2K_train_HR_filename_list = list(map(lambda x: os.path.join(self.DIV2K_train_HR_path, str(x) + '.png'), ['{:0>4}'.format(i + 1) for i in range(800)]))
#         self.DIV2K_train_LR_filename_list = list(map(lambda x: os.path.join(self.DIV2K_train_LR_path, str(x) + '.png'), ['{:0>4}x4'.format(i + 1) for i in range(800)]))
#
#     def __len__(self):
#         return len(self.DIV2K_train_HR_filename_list)
#
#
#     def __getitem__(self, item):
#         return [self.DIV2K_train_LR_filename_list[item], self.DIV2K_train_HR_filename_list[item]]
#
#
#     def generate_div2k_tfrecord(self):
#         writer = tf.python_io.TFRecordWriter(os.path.join(self.DIV2K_path, 'DIV2K.tfrecords'))
#         for index in range(len(self)):
#             # img=PIL.Image.open(self.DIV2K_train_LR_filename_list[index])
#             # print(type(img))
#             # img= img.resize((128,128))
#             # img_raw=img.tobytes()
#             # print(type(img), type(img_raw))
#             input = PIL.Image.open(self.DIV2K_train_LR_filename_list[index])
#             label = PIL.Image.open(self.DIV2K_train_HR_filename_list[index])
#             # print(type(input), type(input.tobytes()))
#             height = input.height
#             width = input.width
#             # print(input.shape)
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
#                 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
#                 'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[input.tobytes()])),
#                 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
#             }))
#             writer.write(example.SerializeToString())
#         writer.close()


class DIV2K_iterator(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.reinit()

    def reinit(self):
        self.finished = False
        self.index_list = list(range(len(self.dataset)))
        random.shuffle(self.index_list)
        self.now_index = 0

    def __iter__(self):
        return self

    def random_crop_pair(self, input, label, cropw, croph, scale):
        iw, ih, ic = input.shape
        lw, lh, lc = label.shape
        offw = random.randint(0, iw - 1 - cropw)
        offh = random.randint(0, ih - 1 - croph)
        crop_input = input[offw:offw + cropw, offh: offh + croph, :]
        crop_label = label[offw * scale: (offw + cropw) * scale, offh * scale: (offh + croph) * scale, :]
        return crop_input, crop_label



    def __next__(self):
        input_list = []
        label_list = []
        if self.finished:
            raise StopIteration
        start = self.now_index
        end = min(self.now_index + self.batch_size, len(self.dataset))
        print(start, end, len(self.dataset))
        for i in range(start, end):
            dataset_index = self.index_list[i]
            input = scipy.misc.imread(self.dataset[dataset_index][0])
            label = scipy.misc.imread(self.dataset[dataset_index][1])
            # crop_input, crop_label = self.random_crop_pair(input, label, 48, 48, 4)
            crop_input, crop_label = input, label
            input_list.append(crop_input)
            label_list.append(crop_label)
            self.now_index = end
        if self.now_index >= len(self.dataset):
            self.finished = True
        # print(len(input_list), len(label_list))
        return [input_list, label_list]

    next = __next__