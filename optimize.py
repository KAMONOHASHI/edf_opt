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
import tensorflow as tf

from custom_grad import Lop, Rop

import losses

FLAGS = tf.app.flags.FLAGS

def apply_gradients(W, dW, rate):
    update_ops = []
    for cur_W, cur_dW in zip(W, dW):
        assign_op = tf.assign_sub(cur_W, cur_dW*rate)
        update_ops.append(assign_op)

    return tf.group(update_ops)

def train_edf(labels, logits, lr, imax=2, momentum = 0.9):
    """ make training operator for Exponential Decaying Flows (Type G).

    :param tf.Tensor labels: training labels. (shape=N)
    :param tf.Tensor logits: network output (shape=[N, D])
    :param lr: learning rate
    :param imax: the number of minres loop
    :param momentum: momentum
    :return:
    """
    batch_size, D = logits.shape.as_list()

    labels = tf.one_hot(labels, D)
    F_size = D * batch_size

    Z = tf.trainable_variables()

    with tf.name_scope("minres"):
        # M : batch_size, D : output dim
        # (M*D)
        Phi_NN = tf.reshape(logits, [-1])
        F_m = tf.nn.softmax(logits) - labels

        F = tf.reshape(F_m, [-1])

        # Phi_NN = logits
        F_div_N = F * (1.0 / batch_size)
        grad = Lop(Phi_NN, Z, F_div_N)
        lam = norm_list_tensor(grad)

        beta = tf.norm(F)
        V1 = F / beta
        xi  = beta

        V0  = np.zeros(F_size, dtype=np.float32)
        gamma0 = tf.constant(1.0, dtype=tf.float32)
        gamma1 = tf.constant(1.0, dtype=tf.float32)
        sigma0 = tf.constant(0.0, dtype=tf.float32)
        sigma1 = tf.constant(0.0, dtype=tf.float32)
        W0 = np.zeros([F_size], dtype=np.float32)
        W1 = np.zeros([F_size], dtype=np.float32)
        V = np.zeros([F_size], dtype=np.float32)

        # MINRES loop
        for cur_loop in range(imax):
            JtxF = Lop(Phi_NN, Z, V1)
            HxF  =  Rop(F, Z, JtxF)[0]

            U = (1.0 / batch_size) * HxF + lam * V1

            alpha = tf.reduce_sum(V1 * U)

            V3 = U - alpha * V1 - beta * V0

            beta_New = tf.norm(V3)
            V1_New = V3 / beta_New
            V0_New = V1

            delta = gamma1*alpha - gamma0 * sigma1 * beta
            rho1 = tf.sqrt(delta*delta + beta_New*beta_New)
            rho2 = sigma1*alpha + gamma0*gamma1*beta
            rho3 = sigma0*beta

            gamma0_New = gamma1
            gamma1_New = delta / rho1
            sigma0_New = sigma1
            sigma1_New  = beta_New / rho1

            W0_New = W1
            W1_New = (1.0 /rho1) * (V1 - rho2*W1 - rho3 * W0)
            V_New = V + gamma1_New * xi * W1_New
            xi_New = -sigma1_New * xi

            V = V_New
            V0 = V0_New
            V1 = V1_New
            W0 = W0_New
            W1 = W1_New
            beta = beta_New
            xi = xi_New
            gamma0 = gamma0_New
            gamma1 = gamma1_New
            sigma0 = sigma0_New
            sigma1 = sigma1_New

        lastV = V

        #### Update W ####
        lastV = (1.0 / batch_size) * lastV
        dZ = Lop(Phi_NN, Z, lastV)

        Z_new = []
        m_new = []
        momentums = []

        # Update variables.
        for cur_Z, cur_dZ in zip(Z, dZ):
            # TODO : change Variable to get_variable
            m = tf.Variable(np.zeros(cur_Z.shape), "momentum_{}".format(cur_Z.op.name), dtype=tf.float32)
            momentums.append(m)


            cur_new_m = momentum * m + (1-momentum) * cur_dZ
            cur_Z_new = cur_Z - lr * cur_new_m

            Z_new.append(cur_Z_new)
            m_new.append(cur_new_m)

        update_m = []
        update_Z = []

        with tf.control_dependencies(m_new + Z_new):
            for cur_m, cur_m_new in zip(momentums, m_new):
                cur_update_m = tf.assign(cur_m, cur_m_new)
                update_m.append(cur_update_m)

            for cur_Z, cur_Z_new in zip(Z, Z_new):
                cur_update_Z = tf.assign(cur_Z, cur_Z_new)
                update_Z.append(cur_update_Z)

        all_update = update_m + update_Z
        update_op = tf.group(*all_update)

    return update_op

def train_edf_while(labels, logits, lr, imax=2, momentum=0.9):
    """ Exponential Decaying Flows (using while_loop for MINRES).

    EDF optimizes with while_loop.
    When imax is big, using while_loop reduces memory consumption.

    :param tf.Tensor labels: training labels. (shape=N)
    :param tf.Tensor logits: network output (shape=[N, D])
    :param lr: learning rate
    :param imax: the number of minres loop
    :param momentum: momentum
    :return:
    """
    batch_size, D = logits.shape.as_list()

    labels = tf.one_hot(labels, D)
    F_size = D * batch_size

    Z = tf.trainable_variables()

    with tf.name_scope("minres"):
        # M : batch_size, D : output dim
        # (M*D)
        Phi_NN = tf.reshape(logits, [-1])
        F_m = tf.nn.softmax(logits) - labels

        F = tf.reshape(F_m, [-1])

        # Phi_NN = logits
        F_div_N = F * (1.0 / batch_size)
        grad = Lop(Phi_NN, Z, F_div_N)
        lam = norm_list_tensor(grad)

        beta_ini = tf.norm(F)
        V1_ini = F / beta_ini
        xi_ini  = beta_ini

        V0_ini  = np.zeros(F_size, dtype=np.float32)
        gamma0_ini = tf.constant(1.0, dtype=tf.float32)
        gamma1_ini = tf.constant(1.0, dtype=tf.float32)
        sigma0_ini = tf.constant(0.0, dtype=tf.float32)
        sigma1_ini = tf.constant(0.0, dtype=tf.float32)
        W0_ini = np.zeros([F_size], dtype=np.float32)
        W1_ini = np.zeros([F_size], dtype=np.float32)
        V_ini = np.zeros([F_size], dtype=np.float32)

        def minres_loop(loop_count, V, V0, V1, W0, W1, beta, xi, gamma0, gamma1, sigma0, sigma1):
            JtxF = Lop(Phi_NN, Z, V1)
            HxF = Rop(F, Z, JtxF)[0]

            U = (1.0 / batch_size) * HxF + lam * V1

            alpha = tf.reduce_sum(V1 * U)

            V3 = U - alpha * V1 - beta * V0

            beta_New = tf.norm(V3)
            V1_New = V3 / beta_New
            V0_New = V1

            delta = gamma1*alpha - gamma0 * sigma1 * beta
            rho1 = tf.sqrt(delta*delta + beta_New*beta_New)
            rho2 = sigma1*alpha + gamma0*gamma1*beta
            rho3 = sigma0*beta

            gamma0_New = gamma1
            gamma1_New = delta / rho1
            sigma0_New = sigma1
            sigma1_New  = beta_New / rho1

            W0_New = W1
            W1_New = (1. /rho1) * (V1 - rho2*W1 - rho3 * W0)
            V_New = V + gamma1_New * xi * W1_New
            xi_New = -sigma1_New * xi
            loop_count = loop_count + 1

            return loop_count, V_New, V0_New, V1_New, W0_New, W1_New, \
                   beta_New, xi_New, gamma0_New, gamma1_New, sigma0_New, sigma1_New

        def condition(loop_count, V, V0, V1, W0, W1, beta, xi, gamma0, gamma1, sigma0, sigma1):
            return tf.less(loop_count, imax)

        loop_vars = [0, V_ini, V0_ini, V1_ini, W0_ini, W1_ini, beta_ini, xi_ini, gamma0_ini, gamma1_ini, sigma0_ini, sigma1_ini]

        results = tf.while_loop(condition, minres_loop, loop_vars, back_prop=False, parallel_iterations=1)
        lastV = results[1]

        #### Update W ####
        lastV = (1.0 / batch_size) * lastV
        dZ = Lop(Phi_NN, Z, lastV)

        Z_new = []
        m_new = []
        momentums = []

        # Update variables.
        for cur_Z, cur_dZ in zip(Z, dZ):
            # TODO : change Variable to get_variable
            m = tf.Variable(np.zeros(cur_Z.shape), "momentum_{}".format(cur_Z.op.name), dtype=tf.float32)
            momentums.append(m)

            cur_new_m = momentum * m + (1 - momentum) * cur_dZ
            cur_Z_new = cur_Z - lr * cur_new_m

            Z_new.append(cur_Z_new)
            m_new.append(cur_new_m)

        update_m = []
        update_Z = []

        with tf.control_dependencies(m_new + Z_new):
            for cur_m, cur_m_new in zip(momentums, m_new):
                cur_update_m = tf.assign(cur_m, cur_m_new)
                update_m.append(cur_update_m)

            for cur_Z, cur_Z_new in zip(Z, Z_new):
                cur_update_Z = tf.assign(cur_Z, cur_Z_new)
                update_Z.append(cur_update_Z)

        all_update = update_m + update_Z
        update_op = tf.group(*all_update)

    return update_op

def norm_list_tensor(list_tensor):
    """ calculates norm of a vector whose components come from list_tensor.

    :param list[tf.Tensor] list_tensor:
    :return:
    """
    elements = []

    for tensor in list_tensor:
        cur_sum = tf.reduce_sum(tensor * tensor)
        elements.append(cur_sum)

    return tf.sqrt(tf.add_n(elements), name="norm_list")

def adam_train(labels, logits, lr):
    optimizer = tf.train.AdamOptimizer(lr)
    total_loss = losses.loss(logits, labels)

    minimizer = optimizer.minimize(total_loss)

    return minimizer

def momentum_train(labels, logits, lr, momentum=0.9):
    optimizer = tf.train.MomentumOptimizer(lr, momentum=momentum)

    total_loss = losses.loss(logits, labels)

    minimizer = optimizer.minimize(total_loss)

    return minimizer

def adagrad_train(labels, logits, lr):
    optimizer = tf.train.AdagradOptimizer(lr)
    total_loss = losses.loss(logits, labels)

    minimizer = optimizer.minimize(total_loss)

    return minimizer

