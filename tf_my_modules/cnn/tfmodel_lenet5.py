#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: tfmodel_lenet5.py

    description: providing a tensorflow model of LeNet5


    author: Jaewook Kang
    final date  : 2018 Mar

'''

import tensorflow as tf

from cnn_layer_modules import Conv2dLayer
from cnn_layer_modules import PoolingLayer
from cnn_layer_modules import FcLayer




class Lenet5(object):

    def __init__(self,dropout_keeprate_for_fc,dtype=tf.float32):

        self.dtype = dtype
        self.dropout_keeprate_for_fc = dropout_keeprate_for_fc
        # stacking CNN layers
        # In convolutional layer num_filter == num_output_channels
        self.c1_layer = Conv2dLayer(layer_index=1,
                               num_input_channels=1,
                               filter_size=5,
                               num_filters=6,
                               filter_stride=1,
                               filter_padding="SAME",
                               activation_type='tanh',
                               dtype=self.dtype)

        self.s2_layer = PoolingLayer(layer_index=2,
                                   num_input_channels=self.c1_layer.num_output_channels,
                                    tile_size=2,
                                    pool_stride=2,
                                    pool_padding='VALID',
                                    pooling_type='avg')

        self.c3_layer = Conv2dLayer(layer_index=3,
                                  num_input_channels=self.s2_layer.num_output_channels,
                                   filter_size=5,
                                   num_filters=16,
                                   filter_stride=1,
                                   filter_padding="VALID",
                                   activation_type='tanh',
                                   dtype=self.dtype)

        self.s4_layer = PoolingLayer(layer_index=4,
                                   num_input_channels=self.c3_layer.num_output_channels,
                                    tile_size=2,
                                    pool_stride=2,
                                    pool_padding='VALID',
                                    pooling_type='avg')

        self.c5_layer = Conv2dLayer(layer_index=5,
                                   num_input_channels=self.s4_layer.num_output_channels,
                                   filter_size=5,
                                   num_filters=120,
                                   filter_stride=1,
                                   filter_padding="VALID",
                                   activation_type='tanh',
                                   dtype=self.dtype)

        self.f6_layer = FcLayer(layer_index=6,
                                  num_input_nodes=self.c5_layer.num_output_channels,
                                  num_output_nodes= 84,
                                  dropout_keep_prob =self.dropout_keeprate_for_fc,
                                  activation_type='tanh',
                                   dtype= self.dtype)

        self.out_layer = FcLayer(layer_index=7,
                                num_input_nodes = self.f6_layer.num_output_nodes,
                                num_output_nodes = 10,
                                dropout_keep_prob = self.dropout_keeprate_for_fc,
                                activation_type = 'none',
                                dtype=self.dtype)

        # layer outputs
        self.c1_layer_out   = None
        self.s2_layer_out   = None
        self.c3_layer_out   = None
        self.s4_layer_out   = None
        self.c5_layer_out   = None
        self.f6_layer_out   = None
        self.out_layer_out  = None

        # cost and optimizer
        self.tf_cost        = None
        self.tf_optimizer   = None







    def get_tf_model(self,input_nodes):

        print('====================================')
        print('[Lenet5] Stacking CNN layers!')
        self.c1_layer_out    = self.c1_layer.make_conv2dlayer( layer_input =input_nodes)
        self.s2_layer_out    = self.s2_layer.make_poolinglayer(layer_input =self.c1_layer_out)
        self.c3_layer_out    = self.c3_layer.make_conv2dlayer( layer_input =self.s2_layer_out)
        self.s4_layer_out    = self.s4_layer.make_poolinglayer(layer_input =self.c3_layer_out)
        self.c5_layer_out    = self.c5_layer.make_conv2dlayer( layer_input =self.s4_layer_out)

        c5_layer_out_shape = self.c5_layer_out.get_shape().as_list()

        # data Tensor reshaping for conv_layer to fc_layer pipeline
        #
        # [batchsize, height, width, channelnum]  ==> [batchsize, height * width * channelnum]
        #  in lenet5, which is [ batchsize, 1*1*120]
        # c5_layer_out_reshape = tf.reshape(tensor=self.c5_layer_out,
        #                                   shape =tf.TensorShape([c5_layer_out_shape[0],
        #                                           c5_layer_out_shape[1]*c5_layer_out_shape[2]*c5_layer_out_shape[3]]))

        c5_layer_out_reshape = tf.contrib.layers.flatten(self.c5_layer_out)
        self.f6_layer_out         = self.f6_layer.make_fclayer(layer_input = c5_layer_out_reshape)
        self.out_layer_out        = self.out_layer.make_fclayer(layer_input = self.f6_layer_out)

        return self.out_layer_out






    def get_tf_cost_fuction(self,train_labels_node,is_l2_loss=False,epsilon=0.0):
        # design cost function======================================
        # in this code the label \in [0, num_label-1]
        # such that we go with sparse_softmax_cross_entropy_with_logits()
        # rather than softmax_cross_entropy_with_logits() which use one-hot coding for the label data
        self.tf_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels_node,
                                                                                    logits=self.out_layer_out))
        # l2 regularization for fully-connected layers
        if is_l2_loss == True:
            fc_regularizers = (tf.nn.l2_loss(self.f6_layer.layer_weight) +
                               tf.nn.l2_loss(self.f6_layer.layer_bias)   +
                               tf.nn.l2_loss(self.out_layer.layer_weight)+
                               tf.nn.l2_loss(self.out_layer.layer_bias))
            self.tf_cost += epsilon* fc_regularizers

        return self.tf_cost







    def get_tf_optimizer(self,opt_type,learning_rate,total_batch_size,minibatch_size,is_exp_decay=False,decay_rate = 1.0):

        if is_exp_decay == True:

            # Decay once per epoch, using an exponential schedule starting at 0.01
            batch = tf.Variable(0.,dtype= self.dtype)
            exp_decay_global_step = batch * minibatch_size
            tf_leaning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                         global_step= exp_decay_global_step,
                                                         decay_steps=total_batch_size,
                                                         decay_rate=decay_rate,
                                                         staircase=True)
            global_step = batch
        else:
            tf_leaning_rate = learning_rate
            global_step = None

        # chose optimizer
        if opt_type == 'Gradient_descent':
            self.tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf_leaning_rate).minimize(
                loss=self.tf_cost,
                global_step=global_step)
            print('[Lenet5] Using GradientDescent optimizer.')

        elif opt_type == 'Adam':
            self.tf_optimizer = tf.train.AdamOptimizer(learning_rate=tf_leaning_rate).minimize(
                loss=self.tf_cost,
                global_step=global_step)
            print('[Lenet5] Using Adam optimizer.')

        else:
            print('[LeNet5] opt_type = %s is not supported.' % opt_type)


        return self.tf_optimizer



