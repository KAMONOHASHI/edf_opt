# Copyright 2018 NS Solutions Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np
import os

import tensorflow as tf


import pickle

"""
Cifar10 Data Manager
"""

FLAGS = tf.app.flags.FLAGS


def _load_cifar10_data(dir_data, names, label_key):
    """ Returns cifar10 dataset (value range [0-255] for images)

    :param str data_dir: directory containing data
    :param list[str] names : filenames
    :param label_key: "labels" for cifar10, "fine_labels" for cifar100
    :rtype: (np.ndarray, np.ndarray)
    :return:
      (images, labels)
    """
    list_data = []
    list_labels = []

    for cur_name in names:
        cur_path = os.path.join(dir_data, cur_name)

        with open(cur_path, "rb") as f:
            # for PYTHON3
            # data_dict = pickle.load(f, encoding="bytes")
            data_dict = pickle.load(f)


        num_data = len(data_dict[label_key])
        new_data = np.reshape(data_dict[b'data'], (num_data, 3, 32, 32))
        new_data = new_data.transpose([0, 2, 3, 1])

        # preprocess
        image = new_data.astype(np.float32) / 255

        list_data.append(image)
        list_labels.append(data_dict[label_key])

    data = np.vstack(list_data)
    label = np.concatenate(list_labels)

    return data, label


def get_cifar10_data(dir_data, sub_mean):
    """returns cifar10 dataset

    :param is_train:
    :return:
    """
    train_names = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5"]


    test_names = ["test_batch"]

    train_data, train_label = _load_cifar10_data(dir_data, train_names, label_key="labels")
    test_data, test_label = _load_cifar10_data(dir_data, test_names, label_key="labels")

    # mean subtract
    if sub_mean:
        mean = np.mean(train_data, axis=0)
        train_data = train_data - mean
        test_data = test_data - mean

    return train_data, train_label, test_data, test_label

def get_cifar100_data(dir_data, sub_mean):
    """returns cifar10 dataset

    :param is_train:
    :return:
    """
    train_names = ["train"]
    test_names = ["test"]

    train_data, train_label = _load_cifar10_data(dir_data, train_names, label_key="fine_labels")
    test_data, test_label = _load_cifar10_data(dir_data, test_names, label_key="fine_labels")

    # mean subtract
    if sub_mean:
        mean = np.mean(train_data, axis=0)
        train_data = train_data - mean
        test_data = test_data - mean

    return train_data, train_label, test_data, test_label

def get_cifar100_mean(dir_data):
    """returns cifar10 dataset

    :param is_train:
    :return:
    """
    train_names = ["train"]
    train_data, train_label = _load_cifar10_data(dir_data, train_names, label_key="fine_labels")
    mean = np.mean(train_data, axis=0)
    return mean

def image_augmentation(images):
    def _augmentation(image):
        _HEIGHT, _WIDTH = 32, 32

        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       _HEIGHT + 8, _WIDTH + 8)

        image = tf.random_crop(image, [_HEIGHT, _WIDTH, 3])
        image = tf.image.random_flip_left_right(image)

        # MODIFIED
        #image = tf.image.per_image_standardization(image)

        return image

    images = tf.map_fn(_augmentation, images, back_prop=False)

    return images

class Feeder(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    @property
    def num_data(self):
        return len(self.images)

    def iter(self, batch_size, shuffle=False, seed=None, force_align=False):
        """

        :param batch_size:
        :param shuffle:
        :param seed:
        :param force_align: add data to make dataset be multiple of batch_size
        :return:
        """

        if seed is not None:
            np.random.seed(seed)

        images = self.images
        labels = self.labels

        if force_align:
            mod = self.num_data % batch_size

            if mod != 0:
                plus_data = batch_size - mod
                indices = np.random.randint(0, self.num_data, plus_data)
                images = np.concatenate((images, self.images[indices]), axis=0)
                labels = np.concatenate((labels, self.labels[indices]), axis=0)

        if shuffle:
            indices = range(len(images))
            perm = np.random.permutation(indices)
            images = images[perm]
            labels = labels[perm]

        for batch_i in range(0, len(images), batch_size):
            batch_images = images[batch_i:batch_i + batch_size]
            batch_labels = labels[batch_i:batch_i + batch_size]

            yield batch_images, batch_labels



