#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
  filename: lab4_runTFCurveFitting.py
  This is an example for linear regression in tensorflow
  Which is a curve fitting example

  written by Jaewook Kang @ Aug 2017
#------------------------------------------------------------
'''
from os import getcwd


import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io



# from __future__ import print_function

# Preparing data set ================================================
from tensorflow.examples.tutorials.mnist import input_data

# generation of sinusoid data set

total_size = 5000
training_size = 4000
validation_size = total_size - training_size

xsize = 50 # the size of single x_data

x_data = np.zeros([xsize, total_size])
cos_x  = np.zeros([xsize, total_size])

mag     = 1.0
phase_rad = np.pi/4
rad_freq = np.pi / 2.0

for i in range(total_size):
    x_data[:,i] =  np.linspace(-4,4,xsize)
cos_x =  np.cos(rad_freq*x_data + phase_rad)



noise_var   = 0.01
noise       = np.sqrt(noise_var) * np.random.randn(xsize,total_size)
y_clean     = cos_x
y_data      = y_clean + noise

x_training_data = x_data[:,0:training_size]
y_training_data = y_data[:,0:training_size]

x_validation_data = x_data[:,training_size:-1]
y_validation_data = y_data[:,training_size:-1]

# signal plot
# hfig1= plt.figure(1,figsize=[10,10])
# plt.plot(cos_x[:,1],color='b',label='clean')
# plt.plot(y_data[:,1],color='r',label='noisy')
# plt.legend()

# configure training parameters =====================================
learning_rate = 0.01
training_epochs = 20
batch_size = 100
display_step = 1

# computational TF graph construction ================================
##---------------- Define graph nodes -------------------
# tf Graph data input holder
# (x,y) : input / output of prediction model
#         which will be feeded by training data  in the TF graph computation
# (a,b,c,d) : model parameters
#         which will be learned from training data in the TF graph computation

x = tf.placeholder(tf.float32, [xsize,None])
y = tf.placeholder(tf.float32, [xsize,None])

# Set model weights which is calculated in the TF graph
a = tf.Variable(1.) # initialization by 1
b = tf.Variable(1.)
c = tf.Variable(1.)
d = tf.Variable(1.)


print ('TF graph nodes are defined')
##--------------------- Define function -----------------
# define relationshitp btw instance data x and label data y
# define optimizer used in the learning phase
# define cost function for optimization

# Construct model

pred_y = c*tf.cos(a*x+b)+d

# Minimize error using MSE function
cost = tf.reduce_mean(tf.reduce_sum( tf.square(y - pred_y) , reduction_indices=1), name="mse")

# Gradient Descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print ('Functions in TF graph are ready')

## Performance evaluation model ========================_y===========
# y               : data output
# pred_y: prediction output by model,
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

error_rate_training = np.zeros(training_epochs)
error_rate_validation = np.zeros(training_epochs)

# Launch the graph (execution) ========================================
# Initializing the variables
init = tf.global_variables_initializer()

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
    pred_c = sess.run(c)
    pred_d = sess.run(d)



hfig1 = plt.figure(1,figsize=(10,10))
epoch_index = np.array([elem for elem in range(training_epochs)])
plt.plot(epoch_index,error_rate_training,label='Training data',color='r',marker='o')
plt.plot(epoch_index,error_rate_validation,label='Validation data',color='b',marker='x')
plt.legend()
plt.title('MSE of prediction:')
plt.xlabel('Iteration epoch')
plt.ylabel('MSE')

hfig2 = plt.figure(2,figsize=(10,10))

pred_y = pred_c * np.cos(pred_a * x_data[:,0] + pred_b) +pred_d

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
