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

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)

    list_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    )

    cross_entropy = tf.reduce_mean(list_cross_entropy, name="cross_entropy")
    return cross_entropy

def sum_crossentropy_loss(logits, labels):
    with tf.name_scope("sum_loss"):
        labels = tf.cast(labels, tf.int64)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        sum_entropy = tf.reduce_sum(cross_entropy, name='cross_entropy')

    return sum_entropy


def corrects(logits, labels):
    with tf.name_scope("sum_loss"):
        labels = tf.cast(labels, tf.int64)
        predicts = tf.argmax(logits, axis=1)

        return tf.equal(labels, predicts)



