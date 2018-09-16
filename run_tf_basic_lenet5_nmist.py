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
        self.batch_norm_epsilon = 1E-5
        self.batch_norm_decay   = 0.99
        self.FLAGS              = None

        # FC layer config
        self.dropout_keeprate   = 0.8
        self.fc_layer_l2loss_epsilon = 5E-5

        self.tf_data_type       = tf.float32

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.random_seed        = 66478

        # tensorboard config
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.root_logdir = getcwd() + '/export/lenet5/'

        self.ckptdir  = self.root_logdir + '/pb_and_ckpt/'
        self.tflogdir = "{}/run-{}/".format(self.root_logdir+'/tf_logs', now)



def conv_layer(layer_in,
               kernel_shape,
               kernel_stride,
               kernel_padding,
               train_config,
               scope=None):

    with tf.variable_scope(name_or_scope=scope,values=[layer_in]):

        weight = tf.get_variable(name='weight',
                                 shape=kernel_shape,
                                 dtype=train_config.tf_data_type,
                                 initializer= train_config.weight_initializer)

        bias = tf.get_variable(name='bias',
                               shape=kernel_shape[3],
                               dtype=train_config.tf_data_type,
                               initializer=train_config.weight_initializer)

        conv_out = tf.nn.conv2d(input= layer_in,
                                filter=weight,
                                strides =kernel_stride,
                                padding=kernel_padding)

        logit_out = tf.nn.bias_add(value=conv_out,
                                   bias=bias)

        return logit_out




def get_model(model_in,
              dropout_keeprate_node,
              train_config,
              scope):

    chin_num = model_in.get_shape().as_list()[3]

    model_shape ={
        'c1_shape': [5,5,chin_num,6],
        's2_shape': [1,2,2,1],
        'c3_shape': [5,5,6,16],
        's4_shape': [1,2,2,1],
        'c5_shape': [5,5,16,120],
        'f6_shape': [120,84],
        'out_shape': [84,10]
    }


    net = model_in
    with tf.variable_scope(name_or_scope=scope,values=[model_in]):

        c1_logit = conv_layer(net,
                            kernel_shape=model_shape['c1_shape'],
                            kernel_stride=[1,1,1,1],
                            kernel_padding='SAME',
                            train_config=train_config,
                            scope='c1_conv')
        c1_out = tf.nn.relu(c1_logit)

        s2_out  = tf.nn.max_pool(value=c1_out,
                                 ksize=model_shape['s2_shape'],
                                 strides=[1,2,2,1],
                                 padding='VALID',
                                 name='s2_pool')

        c3_logit  = conv_layer(s2_out,
                             kernel_shape=model_shape['c3_shape'],
                             kernel_stride=[1,1,1,1],
                             kernel_padding='VALID',
                             train_config=train_config,
                             scope='c3_conv')
        c3_out  = tf.nn.relu(c3_logit)

        s4_out  = tf.nn.max_pool(value=c3_out,
                                 ksize=model_shape['s4_shape'],
                                 strides=[1,2,2,1],
                                 padding='VALID',
                                 name='s4_pool')

        c5_logit  = conv_layer(s4_out,
                             kernel_shape=model_shape['c5_shape'],
                             kernel_stride=[1,1,1,1],
                             kernel_padding='VALID',
                             train_config=train_config,
                             scope='c5_conv')
        c5_out      = tf.nn.relu(c5_logit)

        f6_logit    = tf.layers.dense(c5_out,model_shape['f6_shape'][1])
        f6_logit    = tf.nn.dropout(x=f6_logit,
                                    keep_prob=dropout_keeprate_node,
                                    seed=train_config.random_seed)
        f6_out      = tf.nn.relu(f6_logit)

        out_logit   = tf.layers.dense(f6_out,model_shape['out_shape'][1])
        out_logit   = tf.nn.dropout(x=out_logit,
                                    keep_prob=dropout_keeprate_node,
                                    seed=train_config.random_seed)

        out_logit = tf.reshape(out_logit,
                               shape=[-1,
                                      model_shape['out_shape'][1]])

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

        model_out = get_model(model_in = lenet5_model_in,
                              dropout_keeprate_node=dropout_keeprate_node,
                              train_config = trainconfig_worker,
                              scope         = 'model')

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