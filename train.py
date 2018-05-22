#!/usr/bin/python

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
import time
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import config
import losses
import optimize
import summaries
from cifar10_input import Feeder, get_cifar10_data, get_cifar100_data, image_augmentation
from resnet_model import Cifar10Model

FLAGS = tf.app.flags.FLAGS

def calc_full_batch_loss(feeder, batch_size, images, labels, sess, inf_sum_loss, inf_correct, verbose=False):
    """ calculate full-batch loss

    :param feeder:
    :param batch_size:
    :param images:
    :param labels:
    :param sess:
    :param inf_sum_loss:
    :param inf_correct:
    :param verbose:
    :rtype: (float, float)
    :return:
      loss_mean : average of cross_entropy over full data.
      acc_mean  : accuracy for full data.
    """

    sum_entropy = 0.0
    sum_correct = 0

    loop_count = 0

    for cur_images, cur_labels in feeder.iter(batch_size=batch_size, shuffle=False):
        loop_count += 1
        if verbose:
            print ("loop {}: {}".format(loop_count, np.sum(cur_images)))


        cur_sum_entropy, cur_corrects = sess.run([inf_sum_loss, inf_correct], feed_dict={
            images: cur_images, labels: cur_labels})
        sum_entropy += cur_sum_entropy
        sum_correct += np.sum(cur_corrects)

    loss_mean = sum_entropy / feeder.num_data
    acc_mean = float(sum_correct) / feeder.num_data

    return loss_mean, acc_mean

def restore(path_checkpoint, saver, sess):
    """ return restore operator

    :param path_checkpoint: path to pickle file
    :rtype: int
    :return:
      global step of checkpoint file.
    """
    ckpt = tf.train.get_checkpoint_state(path_checkpoint)
    global_step = None

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path
        print ("restore from {0}".format(ckpt_path))
        saver.restore(sess, ckpt_path)

        global_step = ckpt_path.split('/')[-1].split('-')[-1]

    return global_step

def get_optimizer(labels, logits, lr):
    """

    :param labels:
    :param logits:
    :param lr:
    :rtype: (tf.Operator, tf.Operator, tf.Operator)
    :return:
      epoch_init, train, epoch_finish

      epoch_init : initializing operation at each epoch
      train : train operation
      epoch_finish : finish operation at each epoch
    """
    if FLAGS.optimizer == "adam":
        train_op = optimize.adam_train(labels, logits, lr)
        epoch_init, epoch_finish = tf.no_op(), tf.no_op()
    elif FLAGS.optimizer == "adagrad":
        train_op = optimize.adagrad_train(labels, logits, lr)
    elif FLAGS.optimizer == "momentum":
        train_op = optimize.momentum_train(labels, logits, lr)
    elif FLAGS.optimizer == "edf":
        train_op = optimize.train_edf(labels, logits, lr, imax=FLAGS.imax)
    elif FLAGS.optimizer == "edf_while":
        train_op = optimize.train_edf_while(labels, logits, lr, imax=FLAGS.imax)
    else:
        raise ValueError("invalid optimizer {}".format(FLAGS.optimizer))

    return train_op


def start_train():
    logger = logging.getLogger(__name__)

    if not FLAGS.cifar100:
        full_data, full_label, test_data, test_label = get_cifar10_data(FLAGS.images, sub_mean=True)
        num_classes = 10
    else:
        full_data, full_label, test_data, test_label = get_cifar100_data(FLAGS.images, sub_mean=True)
        num_classes = 100


    num_data = len(full_data)

    if num_data < FLAGS.data_size:
        logger.warning("data_size is bigger than real data {} > {}".format(FLAGS.data_size, num_data))
        examples = num_data
    else:
        examples = FLAGS.data_size

    batch_size = FLAGS.batch
    full_data = full_data[:examples]
    full_label = full_label[:examples]

    feeder = Feeder(full_data, full_label)
    test_feeder = Feeder(test_data, test_label)


    with tf.Graph().as_default():
        # Set random seed for reproducibility.
        # Some GPU operators (ex. reduce_sum) are non-deterministic.
        # For detail, see https://github.com/tensorflow/tensorflow/issues/3103
        tf.set_random_seed(1234)
        np.random.seed(5678)

        global_step = tf.train.create_global_step()
        global_step_inc = tf.assign_add(global_step, 1)


        network = Cifar10Model(num_blocks=FLAGS.block_per_layer, num_classes=num_classes,
                               data_format="channels_first", version=FLAGS.resnet_version)

        # training network
        with tf.name_scope("train_network"):
            train_images = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3], name="input_image")
            train_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="input_labels")

            if FLAGS.augment:
                logger.info("use augmented image")
                aug_images = image_augmentation(train_images)
                logits = network(aug_images, training=True)
            else:
                logits = network(train_images, training=True)


            boundaries = [FLAGS.decay_epoch]
            vals = [FLAGS.lr*v  for v in [1.0,config.LEARNING_RATE_DECAY_FACTOR]]
            lr =  tf.train.piecewise_constant(global_step, boundaries, vals)

            tf.summary.scalar("lr", lr)

            train_op = get_optimizer(train_labels, logits, lr)
            bn_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op] + bn_update_op)

        # inference network
        with tf.name_scope("inf_network"):
            eval_images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="input_image")
            eval_labels = tf.placeholder(dtype=tf.int32, shape=[None], name="input_labels")

            inf_logits = network(eval_images, training=False, reuse=True)
            inf_sum_loss = losses.sum_crossentropy_loss(inf_logits, eval_labels)
            inf_correct  =  losses.corrects(inf_logits, eval_labels)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=1)
        summaries.add_paramter_summary()
        summary_op = tf.summary.merge_all()

        loss_logger = summaries.LossLogger(FLAGS.loss_log)

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        session_config = tf.ConfigProto(gpu_options=gpu_options)

        max_epoch = FLAGS.epoch

        with tf.Session(config=session_config) as sess:
            summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir,
                                                   sess.graph)

            sess.run(tf.variables_initializer(tf.global_variables() + tf.local_variables()))

            if FLAGS.pretrained_dir is not None:
                logger.info("try to load pretrained directory {}.".format(FLAGS.pretrained_dir))
                # load only model variables
                model_variables = network.get_model_variables()

                restore_saver = tf.train.Saver(model_variables)
                restore(FLAGS.pretrained_dir, restore_saver, sess)

            tf.train.start_queue_runners(sess=sess)
            train_loss, train_acc = calc_full_batch_loss(feeder, batch_size, eval_images, eval_labels, sess, inf_sum_loss,
                                                         inf_correct)
            logger.info ("Initial Loss: {} ".format(train_loss))

            for num_epoch in range(max_epoch):
                for cur_images, cur_labels in feeder.iter(batch_size=batch_size, shuffle=True, seed=num_epoch, force_align=True):
                    start_time = time.time()
                    _ = sess.run([train_op], feed_dict={train_images:cur_images, train_labels:cur_labels})

                    duration = time.time() - start_time

                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration

                sec_per_batch = float(duration)

                # evaluation
                train_loss, train_acc = calc_full_batch_loss(feeder, batch_size, eval_images, eval_labels, sess, inf_sum_loss, inf_correct)
                test_loss, test_acc = calc_full_batch_loss(test_feeder, batch_size, eval_images, eval_labels, sess, inf_sum_loss, inf_correct, verbose=False)

                loss_logger.add_loss(num_epoch, train_loss, train_acc, test_loss, test_acc)


                logger.info("step = {} loss = {:.16f} acc_train = {:.4f} acc_test = {:.4f} ({:.1f} examples/sec; {:.1f} sec/batch)"
                            .format(num_epoch, train_loss, train_acc, test_acc, examples_per_sec, sec_per_batch))

                summary = tf.Summary()
                summary.value.add(tag='train_loss', simple_value=train_loss)
                summary.value.add(tag='test_loss', simple_value=test_loss)
                summary.value.add(tag='train_acc', simple_value=train_acc)
                summary.value.add(tag='test_acc', simple_value=test_acc)
                summary_writer.add_summary(summary, num_epoch)

                sum_str = sess.run(summary_op)
                summary_writer.add_summary(sum_str, num_epoch)

                summary_writer.flush()
                saver.save(sess, os.path.join(FLAGS.save_dir, "checkpoint"), global_step=num_epoch)

                sess.run(global_step_inc)


def main(argv=None):
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    if gfile.Exists(FLAGS.save_dir):
        gfile.DeleteRecursively(FLAGS.save_dir)
    if gfile.Exists(FLAGS.train_log_dir):
        gfile.DeleteRecursively(FLAGS.train_log_dir)

    gfile.MakeDirs(FLAGS.save_dir)
    gfile.MakeDirs(FLAGS.train_log_dir)

    logger = logging.getLogger(__name__)
    logger.info(config.get_config_str())
    
    start_train()


if __name__ == '__main__':
    tf.app.run()
