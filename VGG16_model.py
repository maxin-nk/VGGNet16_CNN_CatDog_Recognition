# ! /usr/bin/env python
# coding:utf-8
# python interpreter:3.6.2
# author: admin_maxin


# 训练模型
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
n_class = 10


class Vgg16(object):
    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fclayers()
        self.probs = tf.nn.softmax(self.fc8)

    def saver(self):
        """
        定义模型存储器
        :return:
        """
        return tf.train.Saver()

    def variable_summaries(self, var, name):
        """
        生成变量监控信息并定义生成监控信息日志的操作
        :param var: 输入变量
        :param name: 变量名称
        :return:
        """
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)

    def conv(self, name, input_data, out_channel):
        """
        定义卷积组
        :param name:
        :param input_data:
        :param out_channel:
        :return:
        """
        in_channel = input_data.get_shape()[-1]

        # 定义变量命名空间
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=False)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=False)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
            self.parameters += [kernel, biases]
        return out

    def fc(self, name, input_data, out_channel, trainable=True):
        """
        定义全连接组(展开图像数据)
        :param name:
        :param input_data:
        :param out_channel:
        :return:
        """
        shape = input_data.get_shape().as_list()
        # 获取img纬度
        if len(shape) == 4:
            size = shape[-1]*shape[-2]*shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data, [-1, size])

        # 定义变量命名空间
        with tf.variable_scope(name):
            weights = tf.get_variable("weights", [size, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=trainable)
            res = tf.nn.bias_add(tf.matmul(input_data_flat, weights), biases)
            out = tf.nn.relu(res, name=name)
            self.parameters += [weights, biases]
        return out

    def maxpool(self, name, input_data):
        """
        定义池化层
        :param name:
        :param input_data:
        :return:
        """
        with tf.variable_scope(name):
            out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name=name)
        return out

    def convlayers(self):
        """
        定义卷积模型
        :return:
        """
        # conv1
        self.conv1_1 = self.conv("conv1_1", self.imgs, 64)
        self.conv1_2 = self.conv("conv1_2", self.conv1_1, 64)
        self.pool1 = self.maxpool("pool1", self.conv1_2)

        # conv2
        self.conv2_1 = self.conv("conv2_1", self.pool1, 128)
        self.conv2_2 = self.conv("conv2_2", self.conv2_1, 128)
        self.pool2 = self.maxpool("pool2", self.conv2_2)

        # conv3
        self.conv3_1 = self.conv("conv3_1", self.pool2, 256)
        self.conv3_2 = self.conv("conv3_2", self.conv3_1, 256)
        self.conv3_3 = self.conv("conv3_3", self.conv3_2, 256)
        self.pool3 = self.maxpool("pool3", self.conv3_3)

        # conv4
        self.conv4_1 = self.conv("conv4_1", self.pool3, 512)
        self.conv4_2 = self.conv("conv4_2", self.conv4_1, 512)
        self.conv4_3 = self.conv("conv4_3", self.conv4_2, 512)
        self.pool4 = self.maxpool("pool4", self.conv4_3)

        # conv5
        self.conv5_1 = self.conv("conv5_1", self.pool4, 512)
        self.conv5_2 = self.conv("conv5_2", self.conv5_1, 512)
        self.conv5_3 = self.conv("conv5_3", self.conv5_2, 512)
        self.pool5 = self.maxpool("pool5", self.conv5_3)

    def fclayers(self):
        """
        定义全连接模型
        :return:
        """
        self.fc6 = self.fc("fc6", self.pool5, 4096, trainable=False)
        self.fc7 = self.fc("fc7", self.fc6, 4096, trainable=False)
        self.fc8 = self.fc("fc8", self.fc7, n_class)

    def load_weight(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))
        print("--------------------all done--------------------")