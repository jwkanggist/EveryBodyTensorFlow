#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_tf_basic_lenet5_mnist.py

    description: simple end-to-end LetNet5 implementation
        - For the purpose of EverybodyTensorFlow tutorial
            -
        - training with Mnist data set from Yann's website.
        - the benchmark test error rate is 0.95% which is given by LeCun 1998

        - references:
            - https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
            - https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb


    author: Jaewook Kang
    date  : 2018 Feb.

'''
# Anybody know why we should include "__future__" code conventionally?
# anyway I include the below:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import time
from datetime import datetime
from os import getcwd

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

sys.path.insert(0, getcwd()+'/tf_my_modules/cnn')

from mnist_data_loader import DataFilename
from mnist_data_loader import MnistLoader




# configure training parameters =====================================
TRAININGSET_SIZE     = 5000
VALIDATIONSET_SIZE   = 1000
TESTSET_SIZE         = 1000

class TrainConfig(object):
    def __init__(self):

        self.learning_rate      = 0.01
        self.is_learning_rate_decay = True
        self.learning_rate_decay_rate =0.99
        self.opt_type='Adam'

        self.training_epochs    = 100
        self.minibatch_size     = 1000

        # the number of step between evaluation
        self.display_step   = 5
        self.total_batch    = int(TRAININGSET_SIZE / self.minibatch_size)


        # batch norm config
        self.batch_norm_decay   =  0.999
        self.batch_norm_fused   =  True
        self.FLAGS              = None

        # FC layer config
        self.dropout_keeprate       = 0.8
        self.fc_weights_initializer = tf.contrib.layers.xavier_initializer
        self.fc_weights_regularizer = tf.contrib.layers.l2_regularizer(4E-5)


        # conv layer config
        self.weights_initializer = tf.contrib.layers.xavier_initializer()
        self.weights_regularizer = None
        self.biases_initializer  = slim.init_ops.zeros_initializer()

        self.is_trainable       = True
        self.activation_fn      = tf.nn.relu
        self.normalizer_fn      = slim.batch_norm


        self.random_seed        = 66478
        self.tf_data_type       = tf.float32

        # tensorboard config
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root_logdir = getcwd() + '/export/lenet5_slim/'

        self.ckptdir  = self.root_logdir + '/pb_and_ckpt/'
        self.tflogdir = "{}/run-{}/".format(self.root_logdir+'/tf_logs', now)




def get_model(model_in,
              dropout_keeprate_node,
              train_config,
              scope):

    model_chout_num = \
    {
        'c1': 6,
        'c3': 16,
        'c5': 120,
        'f6': 84,
        'out':10
    }

    net = model_in
    with tf.variable_scope(name_or_scope=scope,values=[model_in]):

        # batch norm arg_scope
        with slim.arg_scope([train_config.normalizer_fn],
                            decay=train_config.batch_norm_decay,
                            fused=train_config.batch_norm_fused,
                            is_training=train_config.is_trainable,
                            activation_fn=train_config.activation_fn):

            if train_config.normalizer_fn == None:
                conv_activation_fn = train_config.activation_fn
            else:
                conv_activation_fn = None
            # max_pool arg_scope
            with slim.arg_scope([slim.max_pool2d],
                                stride      = [2,2],
                                kernel_size = [2,2],
                                padding     = 'VALID'):

                # convolutional layer arg_scope
                with slim.arg_scope([slim.conv2d],
                                        kernel_size = [5,5],
                                        stride      = [1,1],
                                        weights_initializer = train_config.weights_initializer,
                                        weights_regularizer = train_config.weights_regularizer,
                                        biases_initializer  = train_config.biases_initializer,
                                        trainable           = train_config.is_trainable,
                                        activation_fn       = conv_activation_fn,
                                        normalizer_fn       = train_config.normalizer_fn):


                    net = slim.conv2d(inputs=    net,
                                     num_outputs= model_chout_num['c1'],
                                     padding    = 'SAME',
                                     scope      = 'c1_conv')

                    net = slim.max_pool2d(inputs=   net,
                                          scope ='s2_pool')

                    net = slim.conv2d(inputs        = net,
                                      num_outputs   = model_chout_num['c3'],
                                      padding       = 'VALID',
                                      scope         = 'c3_conv')

                    net = slim.max_pool2d(inputs    = net,
                                          scope     = 's4_pool')

                    net  = slim.conv2d(inputs       = net,
                                       num_outputs  = model_chout_num['c5'],
                                       padding      = 'VALID',
                                       scope        = 'c5_conv')


        # output layer by fully-connected layer
        with slim.arg_scope([slim.fully_connected],
                            trainable=      train_config.is_trainable):

            with slim.arg_scope([slim.dropout],
                                keep_prob   =dropout_keeprate_node,
                                is_training=train_config.is_trainable):

                net = slim.fully_connected(inputs        =net,
                                           num_outputs  = model_chout_num['f6'],
                                           activation_fn= train_config.activation_fn,
                                           scope        ='f6_fc')

                net = slim.dropout(inputs=net,
                                   scope='f6_dropout')

                net = slim.fully_connected(inputs       =net,
                                           num_outputs  =model_chout_num['out'],
                                           activation_fn=None,
                                           scope        ='out_fc')

                out_logit = slim.dropout(inputs=net,
                                         scope='out_dropout')

                out_logit = tf.reshape(out_logit,
                                       shape=[-1,
                                              model_chout_num['out']])

        return out_logit




if __name__ == '__main__':

    # worker instance declaration
    datafilename_worker = DataFilename()
    mnist_data_loader = MnistLoader()
    trainconfig_worker = TrainConfig()

    # Download the data
    train_data_filepathname = mnist_data_loader.download_mnist_dataset(
        filename=datafilename_worker.trainingimages_filename)
    train_labels_filepathname = mnist_data_loader.download_mnist_dataset(
        filename=datafilename_worker.traininglabels_filename)

    test_data_filepathname = mnist_data_loader.download_mnist_dataset(filename=datafilename_worker.testimages_filename)
    test_labels_filepathname = mnist_data_loader.download_mnist_dataset(
        filename=datafilename_worker.testlabels_filename)

    # extract data from gzip files into numpy arrays
    train_data = mnist_data_loader.extract_data(filename=train_data_filepathname,
                                                num_images=TRAININGSET_SIZE + VALIDATIONSET_SIZE)
    train_labels = mnist_data_loader.extract_label(filename=train_labels_filepathname,
                                                   num_images=TRAININGSET_SIZE + VALIDATIONSET_SIZE)

    test_data = mnist_data_loader.extract_data(filename=test_data_filepathname,
                                               num_images=TESTSET_SIZE)
    test_labels = mnist_data_loader.extract_label(filename=test_labels_filepathname,
                                                  num_images=TESTSET_SIZE)

    # prepare validation by spliting training set
    validation_data = train_data[:VALIDATIONSET_SIZE, ...]
    validation_labels = train_labels[:VALIDATIONSET_SIZE]

    train_data = train_data[VALIDATIONSET_SIZE:, ...]
    train_labels = train_labels[VALIDATIONSET_SIZE:]


    # network model construction ======================
    # TF computational graph construction
    lenet5_tf_graph = tf.Graph()

    with lenet5_tf_graph.as_default():
        # training nodes (data,label) placeholders
        lenet5_model_in = tf.placeholder(dtype=trainconfig_worker.tf_data_type,
                                   shape=[None, mnist_data_loader.IMAGE_SIZE,
                                                mnist_data_loader.IMAGE_SIZE,
                                                mnist_data_loader.NUM_CHANNELS])
        lenet5_label = tf.placeholder(dtype=tf.int64,
                                     shape=[None, ])

        dropout_keeprate_node = tf.placeholder(dtype=trainconfig_worker.tf_data_type)

        model_out = get_model(model_in              = lenet5_model_in,
                              dropout_keeprate_node =dropout_keeprate_node,
                              train_config          = trainconfig_worker,
                              scope                 = 'model')

        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lenet5_label,
                                                                                logits=model_out))

        train_op = tf.train.AdamOptimizer(learning_rate=trainconfig_worker.learning_rate)\
                                            .minimize(loss=loss_op)


        with tf.name_scope('model_out'):
            model_pred = tf.nn.softmax(model_out)

        with tf.name_scope('eval_performance'):
            error             = tf.equal(tf.argmax(model_pred,1),lenet5_label)
            tf_pred_accuracy     = tf.reduce_mean(tf.cast(error,tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

    ## file writing for Tensorboard
    file_writer = tf.summary.FileWriter(logdir=trainconfig_worker.tflogdir)
    file_writer.add_graph(lenet5_tf_graph)

    ## Summary for Tensorboard visualization
    tb_summary_accuracy = tf.summary.scalar('accuracy', tf_pred_accuracy)
    tb_summary_cost     = tf.summary.scalar('loss', loss_op)


    # network model training ==============================

    train_error_rate        = np.zeros(shape=np.ceil(trainconfig_worker.training_epochs/trainconfig_worker.display_step).astype(np.int16),
                                       dtype=np.float32)
    validation_error_rate   = np.zeros(shape=np.ceil(trainconfig_worker.training_epochs/trainconfig_worker.display_step).astype(np.int16),
                                       dtype=np.float32)
    test_error_rate         = np.zeros(shape=np.ceil(trainconfig_worker.training_epochs/trainconfig_worker.display_step).astype(np.int16),
                                       dtype=np.float32)

    with tf.Session(graph=lenet5_tf_graph) as sess:

        # Run the variable initializer
        sess.run(init)
        print("-------------------------------------------")
        rate_record_index = 0

        for epoch in range(trainconfig_worker.training_epochs):
            avg_cost = 0.
            avg_minibatch_error_rate = 0.

            start_time = time.time()

            # [data shuffling here]
            for i in range(trainconfig_worker.total_batch):
                data_start_index  = i * trainconfig_worker.minibatch_size
                data_end_index    = (i + 1) * trainconfig_worker.minibatch_size

                batch_data  = train_data  [data_start_index:data_end_index, ...]
                batch_label = train_labels[data_start_index:data_end_index]

                _, minibatch_cost = sess.run([train_op,loss_op],
                                             feed_dict={lenet5_model_in:  batch_data,
                                                        lenet5_label:     batch_label,
                                                        dropout_keeprate_node: trainconfig_worker.dropout_keeprate})

                # compute average cost and error rate
                avg_cost                    += minibatch_cost


            avg_cost = avg_cost /  trainconfig_worker.total_batch


            if trainconfig_worker.display_step == 0:
                continue
            elif (epoch + 1) % trainconfig_worker.display_step == 0:
                elapsed_time = time.time() - start_time

                train_error_rate[rate_record_index]      = (1.0 - tf_pred_accuracy.eval(feed_dict={lenet5_model_in: train_data,
                                                                                                  lenet5_label: train_labels,
                                                                                                  dropout_keeprate_node: 1.0})) *100.0

                validation_error_rate[rate_record_index] = (1.0 - tf_pred_accuracy.eval(feed_dict={lenet5_model_in: validation_data,
                                                                                                  lenet5_label: validation_labels,
                                                                                                  dropout_keeprate_node: 1.0})) * 100.0

                test_error_rate[rate_record_index]       = (1.0 - tf_pred_accuracy.eval(feed_dict={lenet5_model_in: test_data,
                                                                                                  lenet5_label: test_labels,
                                                                                                  dropout_keeprate_node: 1.0})) * 100.0

                # tb_summary_cost_result, tb_summary_accuracy_result  = sess.run([tb_summary_cost,tb_summary_accuracy],
                #                                                                feed_dict={lenet5_model_in: train_data,
                #                                                                           lenet5_label: train_labels,
                #                                                                           dropout_keeprate_node:1.0})
                #         file_writer.add_summary(summary_str,step)
                print('At epoch = %d, elapsed_time = %.1f ms' % (epoch, elapsed_time))

                print("Training set avg cost (avg over minibatches)=%.2f" % avg_cost)
                print("Training set Err rate (avg over minibatches)= %.2f %%  " % (train_error_rate[rate_record_index]))
                print("Validation set Err rate (total batch)= %.2f %%" % (validation_error_rate[rate_record_index]))
                print("Test Set Err. rate (total batch)     = %.2f %%" % (test_error_rate[rate_record_index]) )
                print("--------------------------------------------")

                rate_record_index += 1

        print("Training finished!")

    #file_writer.close()

    # Training result visualization ===============================================


    hfig1 = plt.figure(1, figsize=(10, 10))
    err_rate_index = np.array([elem for elem in range(train_error_rate.shape[0])])
    plt.plot(err_rate_index, train_error_rate,      label='Training err', color='r', marker='o')
    plt.plot(err_rate_index, validation_error_rate, label='Validation err', color='b', marker='x')
    plt.plot(err_rate_index, test_error_rate,       label='Test err', color='g', marker='d')
    plt.legend()
    plt.title('Train/Valid/Test Error rate')
    plt.xlabel('Iteration epoch')
    plt.ylabel('error Rate')



    plt.show()