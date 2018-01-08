#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
  filename: lab5_runTFLogisticReg.py
  This is an example for logistic regression in tensorflow
  Which is a curve fitting example

  written by Jaewook Kang @ Sep 2017
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
from pandas import DataFrame
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
xsize  = 2
ysize  = 2
''' (X,Y) data generation '''
# data generation config
mu_class0 = [ 0.,0. ]
mu_class1 = [ 5.,5. ]

sigma_x1_class0 = 1.
sigma_x2_class0 = 1.

sigma_x1_class1 = 2.
sigma_x2_class1 = 2.

corr_coeff = -0.5
#---------------------

cov_class0 = [ [pow(sigma_x1_class0,2),corr_coeff*sigma_x2_class0*sigma_x2_class0],\
               [corr_coeff*sigma_x2_class0*sigma_x1_class0,pow(sigma_x2_class0,2)] ]

cov_class1 = [ [pow(sigma_x1_class1,2),corr_coeff*sigma_x2_class1*sigma_x1_class1],\
               [corr_coeff*sigma_x2_class1*sigma_x1_class1,pow(sigma_x2_class1,2)] ]

x_class0 = np.random.multivariate_normal(mu_class0,cov_class0,total_size)
x_class1 = np.random.multivariate_normal(mu_class1,cov_class1,total_size)

y_class0 = np.zeros([total_size,1])
y_class1 = np.ones([total_size,1])

''' Bernulli error generation '''
error_prob      = 0.01
error_class0    = np.random.binomial(1,error_prob,total_size).reshape(total_size,1)
error_class1    = np.random.binomial(1,error_prob,total_size).reshape(total_size,1)

t_class0 = np.logical_xor(y_class0,error_class0)
t_class1 = np.logical_xor(y_class1,error_class1)

t_class0 = np.float32(t_class0)
t_class1 = np.float32(t_class1)

t_data_class0       = np.zeros([total_size,ysize])
t_data_class0[:,0]  = np.ones(total_size)

t_data_class1       = np.zeros([total_size,ysize])
t_data_class1[:,1]  = np.ones(total_size)


x_data = DataFrame(data = np.concatenate((x_class0, x_class1),axis=0))
t_data = DataFrame(data = np.concatenate((t_data_class0, t_data_class1),axis=0))

# data permutation
permute_index = np.random.permutation(x_data.index)
x_data = x_data.reindex(permute_index)
t_data = t_data.reindex(permute_index)

x_data = x_data.values
t_data = t_data.values

# data plot
# hfig1= plt.figure(1,figsize=[10,10])
# plt.scatter(x_class0[:,0],x_class0[:,1], color='b',label='class0')
# plt.scatter(x_class1[:,0],x_class1[:,1], color='r',label='class1')
# plt.title('Data for Logistic Regression Example')
# plt.legend()

# data dividing
x_training_data = x_data[0:training_size,:]
t_training_data = t_data[0:training_size,:]

x_validation_data = x_data[training_size:-1,:]
t_validation_data = t_data[training_size:-1,:]


# configure training parameters =====================================
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

# computational TF graph construction ================================
##---------------- Define graph nodes -------------------
# tf Graph data input holder
# (x,y) : input / output of prediction model
#         which will be feeded by training data  in the TF graph computation
# (a,b,c,d) : model parameters
#         which will be learned from training data in the TF graph computation

x = tf.placeholder(tf.float32, [None,xsize])
y = tf.placeholder(tf.float32, [None,ysize])

# Set model weights which is calculated in the TF graph
W = tf.Variable(tf.zeros([xsize, ysize]))
b = tf.Variable(tf.zeros([ysize]))

print ('TF graph nodes are defined')
##--------------------- Define function -----------------
# define relationshitp btw instance data x and label data t
# define optimizer used in the learning phase
# define cost function for optimization


# Construct model
predModel = tf.nn.softmax(tf.matmul(x, W) + b)

# Minimize error using cross entropy function
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predModel), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print ('Functions in TF graph are ready')

## Performance evaluation model ========================_y===========
# t               : data output
# predModel: prediction output by model
correctPrediction = tf.equal(tf.argmax(predModel, 1), tf.argmax(y, 1))

# Calculate error rate using data --------------
# where
# tf_reduce_mean(input_tensor, axis) : reduce dimension of tensor by computing the mean of elements
# # 'x' is [[1., 1.]
#         [2., 2.]]
# tf.reduce_mean(x) ==> 1.5
# tf.reduce_mean(x, 0) ==> [1.5, 1.5]
# tf.reduce_mean(x, 1) ==> [1.,  2.]
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
errRateTraining     = np.zeros(training_epochs)
errRateValidation   = np.zeros(training_epochs)

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
            batch_xs = x_training_data[data_start_index:data_end_index,:]
            batch_ts = t_training_data[data_start_index:data_end_index,:]


            #----------------------------------------------
            # Run optimization op (backprop) and cost op (to get loss value)
            # feedign training data
            # print ('predModel = %s' % sess.run(predModel,feed_dict={x:batch_xs,y:batch_ts}))
            _, local_batch_cost = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ts})

            # Compute average loss
            avg_cost += local_batch_cost / total_batch
            # print ("At %d-th batch in %d-epoch, avg_cost = %f" % (i,epoch,avg_cost) )

            # Display logs per epoch step
        if display_step == 0:
            continue
        elif (epoch + 1) % display_step == 0:
            # print("Iteration:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            batch_train_xs = x_training_data
            batch_train_ys = t_training_data
            batch_valid_xs = x_validation_data
            batch_valid_ys = t_validation_data

            errRateTraining[epoch] = 1.0 - accuracy.eval({x: batch_train_xs, \
                                                          y: batch_train_ys}, session=sess)

            errRateValidation[epoch] = 1.0 - accuracy.eval({x: batch_valid_xs, \
                                                            y: batch_valid_ys}, session=sess)

            print("Training set Err rate: %s"   % errRateTraining[epoch])
            print("Validation set Err rate: %s" % errRateValidation[epoch])

        print("--------------------------------------------")
    print("Optimization Finished!")
    Wout = sess.run(W)
    bout = sess.run(b)

##-------------------------------------------
# # training Result display
print("Validation set Err rate:", accuracy.eval({x: x_validation_data, y: t_validation_data},session=sess)/validation_size)


hfig2 = plt.figure(2,figsize=(10,10))
epoch_index = np.array([elem for elem in range(training_epochs)])
plt.plot(epoch_index,errRateTraining,label='Training data',color='r',marker='o')
plt.plot(epoch_index,errRateValidation,label='Validation data',color='b',marker='x')
plt.legend()
plt.title('Classification Error Rate of prediction:')
plt.xlabel('Iteration epoch')
plt.ylabel('error Rate')

hfig3 = plt.figure(3,figsize=(10,10))

plt.scatter(x_class0[:,0], x_class0[:,1], color='b', label='class0')
plt.scatter(x_class1[:,0], x_class1[:,1], color='r', label='class1')
x1_classifier = np.linspace(min(x_data[:,0]),max(x_data[:,0]),50)
x2_classifier = - (Wout[0,0]*x1_classifier + bout[0]) / Wout[1,0]
plt.plot(x1_classifier,x2_classifier,color='k',label='Classifier')
plt.legend()
plt.title('Logistic Reg. example:')
plt.xlabel('X1 data')
plt.ylabel('X2 data')

