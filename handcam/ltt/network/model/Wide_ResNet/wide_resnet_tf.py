# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages
from handcam.ltt.network.model.Wide_ResNet import (
    wide_resnet_tf_official as resnet_model_official,
)

HParams = namedtuple(
    "HParams", "batch_size, num_classes, " "weight_decay_rate, relu_leakiness"
)


class ResNet(object):
    """ResNet model."""

    def __init__(
        self,
        hps,
        images,
        labels,
        mode,
        batch_size,
        model_type,
        input_modality,
        resnet_size,
        seq_len=60,
        imu_data=None,
    ):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        assert mode in ["train", "eval", "inference", "frozen_train"]
        assert model_type in ["sequence", "single_frames"]
        assert input_modality in ["rgb", "depth", "rgbd"]
        assert resnet_size in [18, 50]
        valid_seq_len = (
            False if (model_type == "sequence" and seq_len is None) else True
        )
        assert valid_seq_len

        self.hps = hps
        self.imu_data = imu_data
        self.with_imu = self.imu_data is not None
        self._images = images
        self.labels = labels
        self.mode = mode
        self.is_training = self.mode == "train"
        self.reuse = self.mode == "eval"
        self.model_type = model_type
        self.allow_backprop = self.mode != "frozen_train"
        self.input_modality = input_modality
        self.resnet_size = resnet_size
        self.batch_size = batch_size
        self.seq_len = tf.shape(self._images)[1]

        image_shape = list(self._images.shape)
        image_shape[0] = batch_size

        self.image_shape = image_shape
        # self.image_shape[1] = self.seq_len
        # self._images.set_shape(self.image_shape)

        if self.model_type == "sequence":
            print("Batch size: %d" % self.batch_size)
            # print("Seq len: %d" % self.seq_len)
            print("Image shape: " + str(self.image_shape))
            # TODO: sequence length is unknown at compile time when running validations on random seq_len
            self._images = tf.reshape(
                self._images,
                [
                    tf.shape(self._images)[0] * tf.shape(self._images)[1],
                    image_shape[2],
                    image_shape[3],
                    image_shape[4],
                ],
            )

        if self.mode != "frozen_train":
            # self.labels.set_shape([batch_size, self.labels.shape[1]])

            print(self._images.shape)
            print(self.labels.shape)
        # else:
        # self.is_training = False
        # self.reuse = tf.AUTO_REUSE
        # self._images = tf.reshape(self._images, [-1, 112, 112, 3])
        # self._images.reshape([-1, 112, 112, 3])
        self.weight_decay = 1e-4

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        # self.global_step = tf.train.get_or_create_global_step()
        # self._build_model()
        # if self.mode == 'frozen_train':
        if self.model_type == "sequence":
            self._build_model_official_sequence()
        else:
            self._build_model_official()
            self.loss()

            # self.loss_official()
        # if self.mode == 'training':
        #     self._build_train_op()
        # self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model_official(self):
        with tf.variable_scope("WRN", reuse=self.reuse):
            if self.resnet_size == 18:
                self.wrn_model = resnet_model_official.Model(
                    resnet_size=18,
                    bottleneck=False,
                    num_classes=self.hps.num_classes,
                    num_filters=64,
                    kernel_size=7,
                    conv_stride=2,
                    first_pool_size=3,
                    first_pool_stride=2,
                    second_pool_size=7,
                    second_pool_stride=1,
                    block_sizes=[2, 2, 2, 2],
                    block_strides=[1, 2, 2, 2],
                    final_size=512,
                    version=2,
                    data_format="channels_first",
                    input_modality=self.input_modality,
                )
            else:
                # resnet-50
                self.wrn_model = resnet_model_official.Model(
                    resnet_size=50,
                    bottleneck=True,
                    num_classes=self.hps.num_classes,
                    num_filters=64,
                    kernel_size=7,
                    conv_stride=2,
                    first_pool_size=3,
                    first_pool_stride=2,
                    second_pool_size=7,
                    second_pool_stride=1,
                    block_sizes=[3, 4, 6, 3],
                    block_strides=[1, 2, 2, 2],
                    final_size=2048,
                    version=2,
                    data_format="channels_first",
                    input_modality=self.input_modality,
                )

            self.logits = self.wrn_model(self._images, training=self.is_training)

    def _build_model_official_sequence(self):
        with tf.variable_scope("WRN", reuse=self.reuse):
            if self.resnet_size == 18:
                self.wrn_model = resnet_model_official.Model(
                    resnet_size=18,
                    bottleneck=False,
                    num_classes=self.hps.num_classes,
                    num_filters=64,
                    kernel_size=7,
                    conv_stride=2,
                    first_pool_size=3,
                    first_pool_stride=2,
                    second_pool_size=7,
                    second_pool_stride=1,
                    block_sizes=[2, 2, 2, 2],
                    block_strides=[1, 2, 2, 2],
                    final_size=512,
                    version=2,
                    data_format="channels_first",
                    input_modality=self.input_modality,
                )
            else:
                # resnet-50
                self.wrn_model = resnet_model_official.Model(
                    resnet_size=50,
                    bottleneck=True,
                    num_classes=self.hps.num_classes,
                    num_filters=64,
                    kernel_size=7,
                    conv_stride=2,
                    first_pool_size=3,
                    first_pool_stride=2,
                    second_pool_size=7,
                    second_pool_stride=1,
                    block_sizes=[3, 4, 6, 3],
                    block_strides=[1, 2, 2, 2],
                    final_size=2048,
                    version=2,
                    data_format="channels_first",
                    input_modality=self.input_modality,
                )
            final_shape = 512 if self.resnet_size == 18 else 2048
            if self.input_modality == "rgbd":
                final_shape = final_shape * 2

            self.cnn_representations = tf.reshape(
                self.wrn_model(
                    self._images, training=self.is_training, is_sequence=True
                ),
                [self.batch_size, self.seq_len, final_shape],
            )

            if self.with_imu:
                self.cnn_representations = tf.concat(
                    [self.cnn_representations, self.imu_data], axis=2
                )

            if not self.allow_backprop:
                print("Stopping gradients from flowing into resnet")
                self.cnn_representations = tf.stop_gradient(self.cnn_representations)

            print("CNN representations shape: " + str(self.cnn_representations.shape))

    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope(
            "resnet_model",
            reuse=self.reuse,
            initializer=tf.contrib.layers.xavier_initializer(),
        ):
            strides = [1, 2, 2, 2]
            res_units = [3, 4, 6, 3]
            filters = [16, 16, 32, 64]
            phase_train = tf.placeholder(tf.bool, name="phase_train")

            with tf.variable_scope("init", reuse=self.reuse):
                x = self._images
                x = self._conv(
                    name="conv0",
                    x=x,
                    filter_size=7,
                    in_filters=4,
                    out_filters=64,
                    strides=2,
                    padding=3,
                    use_bias=False,
                )

                x = self._batch_norm_fused(x, 64, "conv0", phase_train)
                x = self._relu(x, self.hps.relu_leakiness)

                x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
                x = tf.nn.max_pool(
                    x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID"
                )

            # Group 0, 3 units [0, 1, 2]
            # print('group0')
            o_g0 = self.group(
                x,
                stride=1,
                base="group0",
                n=res_units[0],
                group_num=0,
                phase_train=phase_train,
            )
            # print('group1')
            o_g1 = self.group(
                o_g0,
                stride=2,
                base="group1",
                n=res_units[1],
                group_num=1,
                phase_train=phase_train,
            )

            o_g2 = self.group(
                o_g1,
                stride=2,
                base="group2",
                n=res_units[2],
                group_num=2,
                phase_train=phase_train,
            )
            #
            o_g3 = self.group(
                o_g2,
                stride=2,
                base="group3",
                n=res_units[3],
                group_num=3,
                phase_train=phase_train,
            )

            o = tf.nn.avg_pool(
                o_g3, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID"
            )
            # o = tf.reshape(o, [-1, 2048])

            with tf.variable_scope("logit", reuse=self.reuse):
                self.logits = self._fully_connected(o, self.hps.num_classes)

    def loss(self):
        with tf.variable_scope("loss"):
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels
            )
            self.loss = tf.reduce_mean(xent, name="xent")
            # self.loss += self._decay()

        with tf.variable_scope("accuracy"):
            self.predictions = tf.nn.softmax(self.logits)
            truth = tf.argmax(self.labels, axis=1)
            predictions = tf.argmax(self.predictions, axis=1)
            self.batch_accuracy = tf.reduce_mean(
                tf.to_float(tf.equal(predictions, truth))
            )

    # def _build_train_op(self):
    #     """Build training specific ops for the graph."""
    #     self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    #     tf.summary.scalar('learning_rate', self.lrn_rate)
    #
    #     trainable_variables = tf.trainable_variables()
    #     grads = tf.gradients(self.loss, trainable_variables)
    #
    #     if self.hps.optimizer == 'sgd':
    #         optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    #     elif self.hps.optimizer == 'mom':
    #         optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    #     elif self.hps.optimizer == 'adam':
    #         optimizer=tf.train.AdamOptimizer(self.lrn_rate)
    #
    #     apply_op = optimizer.apply_gradients(
    #         zip(grads, trainable_variables),
    #         global_step=self.global_step, name='train_step')
    #
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #
    #     print(type(apply_op))
    #     print(type(update_ops))
    #
    #     train_ops = [apply_op] + self._extra_train_ops + update_ops
    #     self.train_op = tf.group(*train_ops)
    #
    #     total_parameters = 0
    #     for variable in tf.trainable_variables():
    #         # shape is an array of tf.Dimension
    #         shape = variable.get_shape()
    #         # print(shape)
    #         # print(len(shape))
    #         variable_parameters = 1
    #         for dim in shape:
    #             # print(dim)
    #             variable_parameters *= dim.value
    #         # print(variable_parameters)
    #         total_parameters += variable_parameters
    #     print("Total params: " + str(total_parameters))

    def group(self, input, stride, base, n, group_num, phase_train):
        with tf.variable_scope(base, reuse=self.reuse):
            o = input
            kernel0 = 2 ** (7 + group_num)
            kernel_1 = 2 ** (7 + group_num)
            kernel_2 = 2 ** (8 + group_num)
            kernel_dim = 2 ** (8 + group_num)
            in_filter = kernel_1
            if group_num == 0:
                in_filter = int(kernel0 / 2)

            for i in range(0, n):
                with tf.variable_scope("block%d" % i, reuse=self.reuse):
                    # print('block%d' % i)
                    # print(in_filter)
                    b_base = "conv"
                    x = o
                    # print('conv0: ' +str([kernel_1, in_filter, 1, 1]))
                    o = self._conv(
                        name=b_base + "0",
                        x=x,
                        filter_size=1,
                        in_filters=in_filter,
                        out_filters=kernel_1,
                        use_bias=False,
                    )
                    o = self._batch_norm_fused(o, kernel0, "conv0", phase_train)
                    o = self._relu(o, self.hps.relu_leakiness)

                    # print('conv1: ' + str([kernel_1, kernel_1, 3, 3]))
                    o = self._conv(
                        name=b_base + "1",
                        x=o,
                        filter_size=3,
                        in_filters=kernel_1,
                        out_filters=kernel_1,
                        strides=i == 0 and stride or 1,
                        padding=1,
                        use_bias=False,
                    )
                    o = self._batch_norm_fused(o, kernel_1, "conv1", phase_train)
                    o = self._relu(o, self.hps.relu_leakiness)

                    # print('conv2: ' + str([kernel_2, kernel_1, 1, 1]))
                    o = self._conv(
                        name=b_base + "2",
                        x=o,
                        filter_size=1,
                        in_filters=kernel_1,
                        out_filters=kernel_2,
                        use_bias=False,
                    )
                    o = self._batch_norm_fused(o, kernel_2, "conv2", phase_train)

                    if i == 0:
                        # print('conv_dim: ' + str([kernel_dim, in_filter, 1, 1]))
                        o += self._conv(
                            name=b_base + "_dim",
                            x=x,
                            filter_size=1,
                            in_filters=in_filter,
                            out_filters=kernel_dim,
                            use_bias=False,
                            strides=stride,
                        )
                    else:
                        o += x

                    o = self._relu(o, self.hps.relu_leakiness)

                    in_filter = kernel_2
            # print("group " + str(group_num) + "out shape: " + str(o.shape))
            return o

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm_fused(self, x, n_out, name, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope("bn" + name, reuse=self.reuse):
            # phase_train = tf.placeholder(tf.bool, name='phase_train')

            normed = tf.contrib.layers.batch_norm(
                x,
                is_training=self.is_training,
                fused=True,
                updates_collections=None,
                reuse=self.reuse,
                scope="bn" + name,
            )
            return normed

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r"DW") > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(
        self,
        name,
        x,
        filter_size,
        in_filters,
        out_filters,
        use_bias=False,
        strides=1,
        padding=0,
    ):
        """Convolution."""
        strides = self._stride_arr(strides)
        with tf.variable_scope(name, reuse=self.reuse):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                "DW",
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)),
            )

            x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            z = tf.nn.conv2d(x, kernel, strides, padding="VALID")
            if use_bias:
                return tf.nn.bias_add(z, bias=0.01)
            else:
                return z

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        print(x.shape)
        x = tf.reshape(x, [self.hps.batch_size, -1])
        print(x.shape)
        w = tf.get_variable(
            # 'DW', [x.get_shape()[1], out_dim],
            "DW",
            [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
        )
        b = tf.get_variable("biases", [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
