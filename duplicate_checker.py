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

import os
import argparse
import pickle
import numpy as np
import cv2
import shutil

from cifar10_input import get_cifar100_data

def deprocess(image):
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def reconstruct_pickle(path_pickle, except_list, path_out):
    with open(path_pickle, "rb") as f:
        data_dict = pickle.load(f)

    dict_pure = {}

    for k, v in data_dict.items():
        new_v = [value for idx,value in enumerate(v) if idx not in except_list]
        dict_pure[k] = new_v

    print ("len(pure) = {}, len(except_list) = {}".format(len(new_v), except_list))

    with open(path_out, "wb") as f:
        pickle.dump(dict_pure, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="dir_in")
    parser.add_argument("-o", dest="dir_out")
    args = parser.parse_args()

    train_data, train_label, test_data, test_label = get_cifar100_data(args.dir_in, sub_mean=False)

    dict_hash = {}

    if args.dir_out is not None:
        if not os.path.exists(args.dir_out):
            os.makedirs(args.dir_out)

    except_list = []

    for idx, data in enumerate(train_data):
        assert isinstance(data, np.ndarray)

        key = data.tobytes()

        if key in dict_hash:
            idx_vs = dict_hash[key]

            print ("detect duplicate {}({}) vs {}({})".format(idx_vs, train_label[idx_vs],
                                                              idx,    train_label[idx]))

            cur_out_vs = os.path.join(args.dir_out, "{}_{}.png".format(idx_vs, train_label[idx_vs]))
            cur_out = os.path.join(args.dir_out, "{}_{}.png".format(idx, train_label[idx]))

            cv2.imwrite(cur_out_vs, deprocess(train_data[idx_vs]))
            cv2.imwrite(cur_out, deprocess(train_data[idx]))

            if train_label[idx_vs] != train_label[idx]:
                except_list.append(idx_vs)
                except_list.append(idx)

        else:
            dict_hash[key] = idx

    print ("len(except) = {}".format(len(except_list)))
    print (except_list)
    path_pickle = os.path.join(args.dir_in, "train")
    path_out    = os.path.join(args.dir_out, "train")
    reconstruct_pickle(path_pickle, except_list, path_out)

    print ("copy test data")
    path_test_in = os.path.join(args.dir_in, "test")
    path_test_out = os.path.join(args.dir_out, "test")
    shutil.copyfile(path_test_in, path_test_out)

if __name__ == "__main__":
    main()