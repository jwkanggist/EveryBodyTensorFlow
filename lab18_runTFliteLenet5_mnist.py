#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: lab18_runTFliteLenet5_mnist.py

    description: Simple end-to-end LetNet5 TFlite implementation
        - For the purpose of EverybodyTensorFlow tutorial
            -
        - training with Mnist data set from Yann's website.
        - the benchmark test error rate is 0.95% which is given by LeCun 1998

        - references:
            - https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
            - https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb


    author: Jaewook Kang
    date  : 2018 Mar.

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

sys.path.insert(0, getcwd()+'/tf_my_modules/cnn')
from tfmodel_lenet5 import Lenet5
from mnist_data_loader import DataFilename
from mnist_data_loader import MnistLoader




# configure training parameters =====================================
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
        self.batch_norm_epsilon = 1E-5
        self.batch_norm_decay   = 0.99
        self.FLAGS              = None

        # FC layer config
        self.dropout_keeprate   = 0.8
        self.fc_layer_l2loss_epsilon = 5E-5

        self.tf_data_type       = tf.float32


        # tensorboard config
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = 'tf_logs'
        self.logdir = "{}/run-{}/".format(root_logdir, now)

        self.ckpt_period = -1


# data size config
# TRAININGSET_SIZE     = 50000
# VALIDATIONSET_SIZE   = 10000
# TESTSET_SIZE         = 10000

TRAININGSET_SIZE     = 5000
VALIDATIONSET_SIZE   = 1000
TESTSET_SIZE         = 1000

# worker instance declaration
datafilename_worker = DataFilename()
mnist_data_loader   = MnistLoader()
trainconfig_worker  = TrainConfig()


# Download the data
train_data_filepathname = mnist_data_loader.download_mnist_dataset(filename=datafilename_worker.trainingimages_filename)
train_labels_filepathname = mnist_data_loader.download_mnist_dataset(filename=datafilename_worker.traininglabels_filename)

test_data_filepathname = mnist_data_loader.download_mnist_dataset(filename=datafilename_worker.testimages_filename)
test_labels_filepathname = mnist_data_loader.download_mnist_dataset(filename=datafilename_worker.testlabels_filename)

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

# [data set should be zipped here]


# network model construction ======================
# TF computational graph construction
lenet5_tf_graph = tf.Graph()

with lenet5_tf_graph.as_default():
    # training nodes (data,label) placeholders
    data_node = tf.placeholder(dtype=trainconfig_worker.tf_data_type,
                               shape=[None, mnist_data_loader.IMAGE_SIZE,
                                            mnist_data_loader.IMAGE_SIZE,
                                            mnist_data_loader.NUM_CHANNELS])
    labels_node = tf.placeholder(dtype=tf.int64,
                                 shape=[None, ])

    dropout_keeprate_node = tf.placeholder(dtype=trainconfig_worker.tf_data_type)

    lenet5_model_builder = Lenet5(dropout_keeprate_for_fc=dropout_keeprate_node,
                                 dtype=trainconfig_worker.tf_data_type)
    lenet5_model_builder.get_tf_model(input_nodes=data_node)

    with tf.name_scope("cost_func"):
        lenet5_model_builder.get_tf_cost_fuction(train_labels_node = labels_node,
                                                is_l2_loss=True,
                                                epsilon=trainconfig_worker.fc_layer_l2loss_epsilon)

    with tf.name_scope('optimizer'):
        lenet5_model_builder.get_tf_optimizer(opt_type=trainconfig_worker.opt_type,
                                         learning_rate=trainconfig_worker.learning_rate,
                                         total_batch_size=TRAININGSET_SIZE,
                                         minibatch_size=trainconfig_worker.minibatch_size,
                                         is_exp_decay=trainconfig_worker.is_learning_rate_decay,
                                         decay_rate=trainconfig_worker.learning_rate_decay_rate)


    with tf.name_scope('model_out'):
        model_pred = tf.nn.softmax(lenet5_model_builder.out_layer_out)

    with tf.name_scope('eval_performance'):
        error             = tf.equal(tf.argmax(model_pred,1),labels_node)
        tf_pred_accuracy     = tf.reduce_mean(tf.cast(error,tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # For pb and ckpt saving
    lenet5_model_builder.set_model_saver()

# file writing for Tensorboard
#file_writer = tf.summary.FileWriter(logdir=trainconfig_worker.logdir)
#file_writer.add_graph(lenet5_tf_graph)

# Summary for Tensorboard visualization
#tb_summary_accuracy = tf.summary.scalar('accuracy', tf_pred_accuracy)
#tb_summary_cost     = tf.summary.scalar('loss', lenet5_model_builder.tf_cost)


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

    # save graph structure in pb / pbtxt
    lenet5_model_builder.save_tfgraph_pb(sess_graph_def=sess.graph_def)
    print("-------------------------------------------")

    for epoch in range(trainconfig_worker.training_epochs):
        avg_cost = 0.
        avg_minibatch_error_rate = 0.

        start_time = time.time()

        # [data shuffling here]
        for i in range(trainconfig_worker.total_batch):
            data_start_index  = i * trainconfig_worker.minibatch_size
            data_end_index    = (i + 1) * trainconfig_worker.minibatch_size

            minibatch_data  = train_data  [data_start_index:data_end_index, ...]
            minibatch_label = train_labels[data_start_index:data_end_index]

            _, minibatch_cost = sess.run([lenet5_model_builder.tf_optimizer,lenet5_model_builder.tf_cost],
                                         feed_dict={data_node:      minibatch_data,
                                                    labels_node:     minibatch_label,
                                                    dropout_keeprate_node: trainconfig_worker.dropout_keeprate})

            # compute average cost and error rate
            avg_cost                    += minibatch_cost


        avg_cost = avg_cost /  trainconfig_worker.total_batch


        if trainconfig_worker.display_step == 0:
            continue
        elif (epoch + 1) % trainconfig_worker.display_step == 0:
            elapsed_time = time.time() - start_time

            train_error_rate[rate_record_index]      = (1.0 - tf_pred_accuracy.eval(feed_dict={data_node: train_data,
                                                                                              labels_node: train_labels,
                                                                                              dropout_keeprate_node: 1.0})) *100.0

            validation_error_rate[rate_record_index] = (1.0 - tf_pred_accuracy.eval(feed_dict={data_node: validation_data,
                                                                                              labels_node: validation_labels,
                                                                                              dropout_keeprate_node: 1.0})) * 100.0

            test_error_rate[rate_record_index]       = (1.0 - tf_pred_accuracy.eval(feed_dict={data_node: test_data,
                                                                                              labels_node: test_labels,
                                                                                              dropout_keeprate_node: 1.0})) * 100.0

            # tb_summary_cost_result, tb_summary_accuracy_result  = sess.run([tb_summary_cost,tb_summary_accuracy],
            #                                                                feed_dict={data_node: train_data,
            #                                                                           labels_node: train_labels,
            #                                                                           dropout_keeprate_node:1.0})
            #         file_writer.add_summary(summary_str,step)
            print('At epoch = %d, elapsed_time = %.1f ms' % (epoch, elapsed_time))

            print("Training set avg cost (avg over minibatches)=%.2f" % avg_cost)
            print("Training set Err rate (avg over minibatches)= %.2f %%  " % (train_error_rate[rate_record_index]))
            print("Validation set Err rate (total batch)= %.2f %%" % (validation_error_rate[rate_record_index]))
            print("Test Set Err. rate (total batch)     = %.2f %%" % (test_error_rate[rate_record_index]) )
            print("--------------------------------------------")

            rate_record_index += 1

        if trainconfig_worker.ckpt_period > 0:
            if epoch % ckpt_period == 0:
                lenet5_model_builder.save_ckpt(sess=sess,epoch=epoch)

    if trainconfig_worker.ckpt_period < 0:
        lenet5_model_builder.save_ckpt(sess=sess)


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