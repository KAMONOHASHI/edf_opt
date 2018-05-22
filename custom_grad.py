# -*- coding:utf-8 -*-

"""
R-operation and L-operation

For detail, See https://github.com/renmengye/tensorflow-forward-ad/issues/2.
"""

import tensorflow as tf

def Lop(f, x, v):
    with tf.name_scope("Lop", values=[f,x,v]):
        return tf.gradients(f, x, grad_ys=v)

def Rop(f, x, v):
    with tf.name_scope("Rop", values=[f,x,v]):
        if isinstance(f, list):
            w = [tf.ones_like(tmp) for tmp in f]
        else:
            w = tf.ones_like(f)

        lop = tf.gradients(f, x, grad_ys=w)
        return tf.gradients(lop, w, grad_ys=v)
