#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: cnn_layer_modules.py

    description: providing a set of basic cnn layer modules
    - conv2d layers
    - pooling layers


    author: Jaewook Kang
    final date  : 2018 Mar

'''
import numpy as np
import tensorflow as tf

# set for random seed
SEED = 66478

class PoolingLayer(object):

    def __init__(self,layer_index,
                 num_input_channels,
                 tile_size,
                 pool_stride,
                 pool_padding='VALID',
                 pooling_type='max'):

        self.layer_index        = layer_index
        self.layer_name         = pooling_type + '_pool_' + str(layer_index)

        self.num_input_channels   = num_input_channels
        self.num_output_channels  = num_input_channels
        self.tile_size          = tile_size
        self.pool_stride        = pool_stride
        self.pooling_type       = pooling_type

        self.pooling_shape          = [1,tile_size,tile_size,1]
        self.pooling_stride_shape   = [1,pool_stride,pool_stride,1]
        self.pooling_padding        = pool_padding

        self.pooling_out        = None
        #
        # print ('[%s] Input channel num = %s'   %  (self.layer_name,self.num_input_channels))
        # print ('[%s] Output channel num = %s'  %  (self.layer_name,self.num_input_channels))
        # print ('[%s] Tile size = [%s X %s]'  % (self.layer_name,self.tile_size, self.tile_size))
        # print ('[%s] Pooling stride = %s'       % (self.layer_name,self.pool_stride))
        # print ('[%s] Pooling padding = %s'      % (self.layer_name,self.pooling_padding))
        # print ('[%s] ----------------------------------' % self.layer_name)
        #



    def make_poolinglayer(self,layer_input):
        if self.pooling_type == 'max':
            self.pooling_out    = tf.nn.max_pool(value=layer_input,
                                                 ksize=self.pooling_shape,
                                                 strides=self.pooling_stride_shape,
                                                 padding=self.pooling_padding,
                                                 name=self.layer_name)
            print('[%s] Making max pooling layer' % self.layer_name)
        elif self.pooling_type == 'avg':
            self.pooling_out    = tf.nn.avg_pool(value=layer_input,
                                                 ksize=self.pooling_shape,
                                                 strides=self.pooling_stride_shape,
                                                 padding=self.pooling_padding,
                                                 name=self.layer_name)
            print ('[%s] Making avg pooling layer' % self.layer_name)

        else:
            print('[%s] pooling type = %s is not supported.' % (self.layer_name, self.pooling_type))

        print ('[%s] (In chs, Out_chs, tilesize, stride)=(%s,%s,%s,%s)' %
               (self.layer_name, self.num_input_channels, self.num_output_channels, self.tile_size,
                self.pool_stride))
        print ('-------------------------')

        return self.pooling_out




class Conv2dLayer(object):

    def __init__(self,layer_index,
                 num_input_channels,
                 filter_size,
                 num_filters,
                 filter_stride,
                 filter_padding="SAME",
                 activation_type='relu',
                 dtype          = tf.float32):

        self.layer_index        = layer_index
        self.variable_scope_name=  'conv_' + activation_type + '_' + str(self.layer_index)

        # conv layer config
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_filters
        self.num_filters         = num_filters
        self.filter_size        = filter_size

        self.filter_shape       = [filter_size,filter_size,num_input_channels,num_filters]
        self.bias_shape         = num_filters
        self.activation_type    = activation_type

        # filtering operation config
        self.filter_stride_shape      = [1,filter_stride,filter_stride,1]
        self.filter_padding     = filter_padding

        # tf variables
        self.filter_weight      = None
        self.layer_bias        = None
        self.layer_logit        = None
        self.layer_out          = None
        self.dtype              = dtype

        self.get_conv2dlayer_tfvariables()





    def get_conv2dlayer_tfvariables(self):
        with tf.variable_scope(name_or_scope=self.variable_scope_name):
            self.filter_weight  = tf.get_variable(name= "weights",
                                                 shape= self.filter_shape,
                                                 dtype=self.dtype,
                                                 initializer=tf.random_normal_initializer(mean=0.0,
                                                                                          stddev=0.1,
                                                                                          seed=SEED))

            self.layer_bias    = tf.get_variable(name= "bias",
                                                  shape= self.bias_shape,
                                                  dtype=self.dtype,
                                                  initializer=tf.random_normal_initializer(mean=0.0,
                                                                                           stddev=0.1,
                                                                                           seed=SEED))

        # print('[%s] filter shape = %s, bias shape = %s' % (self.variable_scope_name,self.filter_shape,self.bias_shape))
        # print ('[%s] Input channel num = %s'   %  (self.variable_scope_name,self.num_input_channels ))
        # print ('[%s] Output channel num = %s'  %  (self.variable_scope_name,self.num_filters))
        # print ('[%s] Filter size = [%s X %s]'  % (self.variable_scope_name,self.filter_size, self.filter_size))
        # print ('[%s] Filter stride = %s'       % (self.variable_scope_name,self.filter_stride_shape))
        # print ('[%s] Filter padding = %s'      % (self.variable_scope_name,self.filter_padding))
        # print ('[%s] ----------------------------------' % self.variable_scope_name)
        return self.filter_weight, self.layer_bias




    def activation(self):

        if self.activation_type == 'relu':
            self.layer_out = tf.nn.relu(features=self.layer_logit,
                                        name=self.variable_scope_name + '_actfunc')
            # print('[%s] Activation function = %s ' % (self.variable_scope_name,self.activation_type))

        elif self.activation_type == 'softmax':
            self.layer_out  = tf.nn.softmax(logits=self.layer_logit,
                                            name=self.variable_scope_name + '_actfunc')
            # print('[%s] Activation function = %s ' % (self.variable_scope_name,self.activation_type))

        elif self.activation_type == 'tanh':
            self.layer_out  = tf.nn.tanh(x=self.layer_logit,
                                         name=self.variable_scope_name + '_actfunc')
            # print('[%s] Activation function = %s ' % (self.variable_scope_name,self.activation_type))
        elif self.activation_type == 'none':
            self.layer_out  = self.layer_logit
            # print('[%s] Configure to no activation' % self.variable_scope_name)

        else:
            print('[%s] Activation function named %s is not supported' % (self.variable_scope_name,self.activation_type))

        return self.layer_out





    def make_conv2dlayer(self,layer_input):

        with tf.name_scope(self.variable_scope_name):
            conv = tf.nn.conv2d(input=layer_input,
                                filter=self.filter_weight,
                                strides=self.filter_stride_shape,
                                padding=self.filter_padding)

            self.layer_logit = tf.nn.bias_add(conv, self.layer_bias)
            self.layer_out   = self.activation()
        print ('[%s] Making conv2d layer  ' % self.variable_scope_name)
        print ('[%s] (In chs, Out_chs, filter_size, stride,activation_type)=(%s,%s,%s,%s,%s)' %
               (self.variable_scope_name,
                self.num_input_channels,
                self.num_output_channels,
                self.filter_size,
                self.filter_stride_shape[1],
                self.activation_type))
        print ('-------------------------')

        return self.layer_out







class FcLayer(object):

    def __init__(self,layer_index,
                 num_input_nodes,
                 num_output_nodes,
                 dropout_keep_prob,
                 activation_type='relu',
                 dtype= tf.float32):

        self.layer_index        = layer_index
        self.variable_scope_name=  'fc_' + activation_type+ '_' + str(self.layer_index)

        # conv layer config
        self.num_input_nodes     = num_input_nodes
        self.num_output_nodes    = num_output_nodes

        self.layer_shape       = [self.num_input_nodes, self.num_output_nodes]
        self.activation_type    = activation_type
        self.dropout_keep_prob = dropout_keep_prob

        # tf variables
        self.layer_weight      = None
        self.layer_bias        = None
        self.layer_logit        = None
        self.layer_out          = None
        self.dtype              = dtype
        self.get_fclayer_tfvariables()




    def get_fclayer_tfvariables(self):
        with tf.variable_scope(name_or_scope=self.variable_scope_name):
            self.layer_weight  = tf.get_variable(name= "weights",
                                                 shape= self.layer_shape,
                                                 dtype= self.dtype,
                                                 initializer=tf.random_normal_initializer(mean=0.0,
                                                                                          stddev=0.1,
                                                                                          seed=SEED))

            self.layer_bias    = tf.get_variable(name= "bias",
                                                  shape= self.num_output_nodes,
                                                  dtype= self.dtype,
                                                  initializer=tf.random_normal_initializer(mean=0.0,
                                                                                           stddev=0.1,
                                                                                           seed=SEED))

        # print ('[%s] Input node num = %s'   %  (self.variable_scope_name,self.num_input_nodes))
        # print ('[%s] Output node num = %s'  %  (self.variable_scope_name,self.num_output_nodes))
        # print ('[%s] Dropout keep rate = %s'   %  (self.variable_scope_name,self.dropout_keep_prob))
        # print ('[%s] ----------------------------------' % self.variable_scope_name)

        return self.layer_weight, self.layer_bias




    def activation(self):

        if self.activation_type == 'relu':
            self.layer_out = tf.nn.relu(features=self.layer_logit,
                                        name=self.variable_scope_name + '_actfunc')
            # print('[%s] Activation function = %s ' % (self.variable_scope_name,self.activation_type))

        # elif self.activation_type == 'softmax':
            self.layer_out  = tf.nn.softmax(logits=self.layer_logit,
                                            name=self.variable_scope_name + '_actfunc')
            # print('[%s] Activation function = %s ' % (self.variable_scope_name,self.activation_type))

        elif self.activation_type == 'tanh':
            self.layer_out  = tf.nn.tanh(x=self.layer_logit,
                                         name=self.variable_scope_name + '_actfunc')
            # print('[%s] Activation function = %s ' % (self.variable_scope_name,self.activation_type))
        elif self.activation_type == 'none':
            self.layer_out = self.layer_logit
            # print ('[%s] Configure to no activation' % self.variable_scope_name)

        else:
            print('[%s] Activation function named %s is not supported' % (self.variable_scope_name,self.activation_type))

        return self.layer_out




    def make_fclayer(self,layer_input):

        with tf.name_scope(self.variable_scope_name):
            logit_before_dropout    =   tf.matmul(a=layer_input,
                                                  b=self.layer_weight) + self.layer_bias

            self.layer_logit        = tf.nn.dropout(x=logit_before_dropout,
                                                    keep_prob=self.dropout_keep_prob,
                                                    name=self.variable_scope_name + '_dropout')
            self.layer_out = self.activation()
        print ('[%s] Making fc layer  ' % self.variable_scope_name)
        print ('[%s] (In chs, Out_chs, activation_type)=(%s,%s,%s)' %
               (self.variable_scope_name,self.num_input_nodes,self.num_output_nodes,self.activation_type))
        print ('-------------------------')

        return self.layer_out
