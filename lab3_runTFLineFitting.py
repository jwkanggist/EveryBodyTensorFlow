#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
  filename: lab3_runTFLineFitting.py
  This is an example for linear regression in tensorflow
  Which is a line fitting example

  written by Jaewook Kang @ Aug 2017
#------------------------------------------------------------
'''
from os import getcwd

import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io



# from __future__ import print_function

# Preparing data set ================================================
from tensorflow.examples.tutorials.mnist import input_data

# generation of data set
total_size = 5000
training_size = 4000
validation_size = 1000

xsize = 20

x_data = np.zeros([xsize, total_size])

a_true = 2
b_true = 0.5

for i in range(total_size):
    x_data[:,i] =  np.linspace(0,10,xsize)

noise_var   = 1.0
noise       = np.sqrt(noise_var) * np.random.randn(xsize,total_size)
y_clean     = a_true * x_data + b_true
y_data      = y_clean + noise

x_training_data = x_data[:,0:training_size]
y_training_data = y_data[:,0:training_size]

x_validation_data = x_data[:,training_size:-1]
y_validation_data = y_data[:,training_size:-1]


# configure training parameters =====================================

learning_rate = 0.000001
training_epochs = 20
batch_size = 100
display_step = 1

# computational TF graph construction ================================
##---------------- Define graph nodes -------------------
# tf Graph data input holder
# (x,y) : input / output of prediction model
#         which will be feeded by training data  in the TF graph computation
# (a,b) : model parameters
#         which will be learned from training data in the TF graph computation

x = tf.placeholder(tf.float32, [xsize, None])
y = tf.placeholder(tf.float32, [xsize, None])


with tf.variable_scope(name_or_scope='model',
                        values=[x,y]):


    # Set model weights which is calculated in the TF graph
    # a = tf.Variable(0.,name='a') # initialization by 1
    # b = tf.Variable(tf.zeros([xsize]))
    # b = tf.Variable(0.,name='b')

    a = tf.get_variable(name='a',
                        shape=[1],
                        initializer=tf.random_normal_initializer)

    b = tf.get_variable(name='b',
                        shape=[1],
                        initializer=tf.random_normal_initializer)

    print ('TF graph nodes are defined')
    ##--------------------- Define function -----------------
    # define relationshitp btw instance data x and label data y
    # define optimizer used in the learning phase
    # define cost function for optimization

    # Construct model
    pred_y =  a * x + b

# Minimize error using MSE function
cost = tf.reduce_mean(tf.reduce_sum( tf.square(y - pred_y) , reduction_indices=1), name="mse")

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

print ('Functions in TF graph are ready')

## Performance evaluation model ========================_y===========
# y               : data output
# pred_y: prediction output by model,  a x + b
correct_prediction = cost

# Calculate error rate using data --------------
# where
# tf_reduce_mean(input_tensor, axis) : reduce dimension of tensor by computing the mean of elements
# # 'x' is [[1., 1.]
#         [2., 2.]]
# tf.reduce_mean(x) ==> 1.5
# tf.reduce_mean(x, 0) ==> [1.5, 1.5]
# tf.reduce_mean(x, 1) ==> [1.,  2.]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

error_rate_training     = np.zeros(training_epochs)
error_rate_validation   = np.zeros(training_epochs)

# tensorboard summary
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs/line_fitting'
subdir = "{}/run-{}/".format(root_logdir, now)

logdir = './pb_and_ckpt/' + subdir

if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)

summary_writer = tf.summary.FileWriter(logdir=logdir)
summary_writer.add_graph(graph=tf.get_default_graph())

# Launch the graph (execution) ========================================
# Initializing the variables
init = tf.global_variables_initializer()
summary_cost = tf.summary.scalar('cost',cost)

## -------------------- Learning iteration start --------------------
with tf.Session() as sess:
    sess.run(init) # this for variable use

    # Training cycle
    for epoch in range(training_epochs): # iteration loop
        avg_cost = 0.
        total_batch = int(training_size/batch_size) #
        # Loop over all batches
        for i in range(total_batch): # batch loop
            data_start_index = i * batch_size
            data_end_index   = (i + 1) * batch_size
            # feed traing data --------------------------
            batch_xs = x_training_data[:,data_start_index:data_end_index]
            batch_ys = y_training_data[:,data_start_index:data_end_index]

            #----------------------------------------------
            # Run optimization op (backprop) and cost op (to get loss value)
            # feedign training data
            _, local_batch_cost = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                                         y: batch_ys})

            # Compute average loss
            avg_cost += local_batch_cost / total_batch
            # print ("At %d-th batch in %d-epoch, avg_cost = %f" % (i,epoch,avg_cost) )

            summary_str = summary_cost.eval(feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, epoch)

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost/batch_size))

            batch_xs = x_training_data
            batch_ys = y_training_data

            error_rate_training[epoch]   = accuracy.eval({x: batch_xs, y: batch_ys},session=sess)/training_size
            error_rate_validation[epoch] = accuracy.eval({x: x_validation_data, y: y_validation_data},session=sess)/validation_size

            print("Training set MSE:", error_rate_training[epoch])
            print("Validation set MSE:", error_rate_validation[epoch])

        print("--------------------------------------------")
    print("Optimization Finished!")
    pred_a = sess.run(a)
    pred_b = sess.run(b)

##-------------------------------------------
    # # training Result display
    print("Validation set Err rate:", accuracy.eval({x: x_validation_data, y: y_validation_data},session=sess)/validation_size)

hfig1 = plt.figure(1,figsize=(10,10))
epoch_index = np.array([elem for elem in range(training_epochs)])
plt.plot(epoch_index,error_rate_training,label='Training data',color='r',marker='o')
plt.plot(epoch_index,error_rate_validation,label='Validation data',color='b',marker='x')
plt.legend()
plt.title('MSE of prediction:')
plt.xlabel('Iteration epoch')
plt.ylabel('MSE')

plt.show()

hfig2 = plt.figure(2,figsize=(10,10))
pred_y = pred_a * x_data[:,0] + pred_b
plt.plot(x_validation_data[:,0],y_validation_data[:,0],label='noisy data',color='b',marker='*')
plt.plot(x_validation_data[:,0], pred_y,label='prediction',color='r')
plt.legend()
plt.title('A line fitting example:')
plt.xlabel('X data')
plt.ylabel('Y data')

plt.show()
# FIG_SAVE_DIR = getcwd() + '/figs/'
# hfig1.savefig(FIG_SAVE_DIR + 'runExample_TFLogisticReg_aymeric_ErrRate.png')

# hfig1.clear()
