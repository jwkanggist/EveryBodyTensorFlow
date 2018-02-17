#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: bbrbm.py

    This is for Bernoulli-Bernoulli Restricted Boltzammn
    Machine implementation in TensorFlow

    - Handling binary visible and hidden units
    - Pre-training algorithm is implemented by contrastive divergence
    with K gibbs sampling iteration
    - Bernoulli-Bernoulli RBM is good for
    Bernoulli-distributed binary input data. MNIST, for example.

    ref1: https://github.com/blackecho/Deep-Learning-TensorFlow/blob/master/yadlt/models/boltzmann/rbm.py
    ref2: https://github.com/meownoid/tensorfow-rbm
    written by Jaewook Kang @ Dec 2017
#------------------------------------------------------------
'''

from os import getcwd
import matplotlib.pyplot as plt

import numpy as np
from sklearn import metrics
import tensorflow as tf


class BBRBM (object):

    def __init__(self, num_visible,num_hidden,\
                 learning_rate,batch_size,\
                 num_epoch,gibbs_sampling_num_iter ):

        self._num_visible               = num_visible
        self._num_hidden                = num_hidden
        self._learning_rate             = learning_rate
        self._batch_size                = batch_size
        self._num_epoch                 = num_epoch
        self._gibbs_sampling_num_iter   = gibbs_sampling_num_iter

        # input data from training set
        self._input_data = np.empty(shape=(0,0))

        # weight and bias
        self._weight    = np.zeros(shape=(num_visible,num_hidden))
        self._biases    = None

        # variable and hidden units
        self._visible_state     = None
        self._hidden0_state     = None
        self._hidden1_prob      = None


        # tf loss function
        self._loss              = None
        self._reconst_error     = None
        self._variable_init     = None



    def fit_model(self,training_set,validation_set):

        # visible units
        self._input_data = tf.placeholder("float", [None, self._num_visible])

        self._visible_state = tf.Variable("float",\
                                          tf.zeros([self._num_visible]),\
                                          name='visible_state')
        self._hidden0_state = tf.Variable("float",\
                                          tf.zeros([self._num_hidden]) ,\
                                          name='hidden_state0')
        self._hidden1_prob = tf.Variable("float",\
                                          tf.zeros([self._num_hidden]) ,\
                                          name='hidden_prob1')

        self._weight = tf.Variable(tf.random_normal([self._num_visible,self._num_hidden],mean=0.0,stddev=0.01))
        self._biases = \
        {
            'b': tf.Variable(tf.zeros([self._num_visible] ),name='hidden_bias'),
            'c': tf.Variable(tf.zeros([self._num_hidden]),name='visible_bias')
        }

        self._build_model()

        #--------------- 여기까지 171211 ---------------#
        self._train_model()

        self._report_result()


    def _train_model(self):



    def _get_reconst_error(self):

    def _get_free_energy(self):

    def _sample_prob(self,probs, rand):
        """Get samples from a tensor of probabilities.
        :param probs: tensor of probabilities
        :param rand: tensor (of the same shape as probs) of random values
        :return: binary sample of probabilities
            """
        return tf.nn.relu(tf.sign(probs - rand))


    def _build_model(self):

        # first trial of gibb sampling
        hidden_prob0, hidden_state0, visible_prob, visible_state = \
            self._run_single_step_gibbs_sampling(self._input_data)
        gibbsampler_input   = visible_prob
        hidden_prob1        = hidden_prob0
        hidden_state1       = hidden_prob1

        # residual gibb sampling
        for gibbsampling_iter in range(self._gibbs_sampling_num_iter - 1):
            hidden_prob1, hidden_state1, visible_prob, visible_state = \
                self._run_single_step_gibbs_sampling(gibbsampler_input)

            gibbsampler_input = visible_prob

        positive = tf.matmul(tf.transpose(self._input_data),hidden_state0)
        negative = tf.matmul(tf.transpose(visible_prob),hidden_prob1)

        # update weight and bias gradient by contrastive divergence
        self.weight_update       = self._weight.     assign_add(self._learning_rate * (positive - negative))
        self.hidden_bias_update  = self._biases['c'].assign_add(self._learning_rate * tf.reduce_mean(hidden_prob0 - hidden_prob1,0) )
        self.visible_bias_update = self._biases['b'].assign_add(self._learning_rate * tf.reduce_mean(self._input_data - visible_prob,0))

        self.hidden0_state_update = tf.assign(self._hidden0_state    ,hidden_state0)
        self.hidden1_prob_update  = tf.assign(self._hidden1_prob     ,hidden_prob1)
        self.visible_state_update = tf.assign(self._visible_state    ,visible_state)

        self._loss =  tf.sqrt(tf.reduce_mean(tf.square(self._input_data - visible_prob)))


    def _run_single_step_gibbs_sampling(self,vstate_from_data):

        # calculate p(h|v)
        hidden_prob  = tf.nn.sigmoid(tf.matmul(vstate_from_data,self._weight) + self._biases['c'])

        # generate hidden variable on the basis of p(h|v)
        hidden_state = self._sample_prob(hidden_prob,np.random.rand(self._num_hidden))

        # calculate p(v|h)
        visible_prob    = tf.nn.sigmoid(tf.matmul(self._weight,hidden_state) + self._biases['b'])

        # generate visible variable on the basis of p(v|h)
        visible_state   = self._sample_prob(visible_prob,np.random.rand(self._num_visible))

        return hidden_prob, hidden_state, visible_prob, visible_state