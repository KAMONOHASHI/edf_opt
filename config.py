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


import tensorflow as tf


# input data parameters ########
tf.app.flags.DEFINE_string('images', './data/cifar-10-batches-py/',
                           """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_integer('data_size', 50000,
                            "dataset size")

tf.app.flags.DEFINE_boolean('cifar100', False,
                            "use cifar100")
tf.app.flags.DEFINE_boolean('augment', False,
                            "use data augmentation")


# training parameters ########

tf.app.flags.DEFINE_float('lr', 1.0,
                          """initial learning rate.""")

tf.app.flags.DEFINE_integer('imax', 2,
                            "imax")


tf.app.flags.DEFINE_integer('batch', 250,
                            "batch_size")

tf.app.flags.DEFINE_integer('epoch', 100,
                            "training epoch")

tf.app.flags.DEFINE_integer('decay_epoch', 999,
                            "training decay epoch")

LEARNING_RATE_DECAY_FACTOR = 0.1    # Learning rate decay factor.


tf.app.flags.DEFINE_float('bn_decay', 0.9,
                          "batch normalization running mean parameter")

tf.app.flags.DEFINE_string('optimizer', 'edf',
                           "optimizer (edf|adam|adagrad)")

tf.app.flags.DEFINE_string('pretrained_dir', None,
                           "pretrained data directory")



# model parameters ########

# Resnet parmeters (Resnet14=2, Resnet32=5, Resnet110=18)
tf.app.flags.DEFINE_integer('block_per_layer', 2,
                            "block per layer (Resnet)")

tf.app.flags.DEFINE_integer('resnet_version', 1,
                            "resnet version")


# output parameters #######
tf.app.flags.DEFINE_string('train_log_dir', './output/train_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('save_dir', './output/parameter',
                           "Directory where to write parameters.")

tf.app.flags.DEFINE_string('loss_log', './output/parameter/loss_log.csv',
                           "log file path")



def get_config_str():
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    res = {}
    for key, value in tf.app.flags.FLAGS.__flags.items():
        res[key] = value.value

    return pp.pformat(res)
