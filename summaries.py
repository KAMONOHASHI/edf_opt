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

import pandas as pd
import tensorflow as tf

def add_paramter_summary():
    vars = tf.trainable_variables()

    for v in vars:
        tf.summary.histogram(v.op.name, v)

class LossLogger(object):
    def __init__(self, path_out):
        self.list_loss_log = []
        self.path_out = path_out

    def add_loss(self, num_epoch, train_loss, train_acc, test_loss, test_acc):

        loss_entry = {
            "epoch": num_epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc
        }

        self.list_loss_log.append(loss_entry)

        df = pd.DataFrame(self.list_loss_log)

        df = df[["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]]

        # write files for the end of each epoch
        df.to_csv(self.path_out, index=False,encoding="shift-jis")


