#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: lab12_runTCcheckBatchNorm_spiraldata.py

    To check Gradient Vanishing problem in
    A Multi-Hidden Layers Fully Connected Neural Network.
    This script aim to see how the "batch normalization"
    accelerate the training of
    A Multi-Hidden Layers Fully Connected Neural Network.

    Applying "batch normalization" to the lab10

    This example data set is using two class spiral data

    ref1:
    https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    ref2:
    http://ruishu.io/2016/12/27/batchnorm/

    written by Jaewook Kang @ Jan 2018
#------------------------------------------------------------
'''


from os import getcwd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io


# reading data set from csv file ==========================
xsize  = 2
ysize  = 2

data = pd.read_csv('./data/twospirals_N5000.csv')
data.columns=['xdata1','xdata2','tdata']
permutation_index = np.random.permutation(data.index)
permutated_data = data.reindex(permutation_index)
permutated_data.columns=['xdata1','xdata2','tdata']

x_data = np.zeros([permutated_data.xdata1.size,xsize])
x_data[:,0] = permutated_data.xdata1.values
x_data[:,1] = permutated_data.xdata2.values

t_data = np.zeros([permutated_data.tdata.size,ysize])
t_data[:,0] = permutated_data.tdata.values
t_data[:,1] = np.invert(permutated_data.tdata.values) + 2


total_size = permutated_data.xdata1.size
training_size = int(np.floor(permutated_data.xdata1.size * 0.8))
validation_size = total_size - training_size


# data dividing
x_training_data = x_data[0:training_size,:]
t_training_data = t_data[0:training_size,:]

x_validation_data = x_data[training_size:-1,:]
t_validation_data = t_data[training_size:-1,:]



# configure training parameters =====================================
learning_rate = 0.1
training_epochs = 100
batch_size = 30
display_step = 1
total_batch = int(training_size / batch_size)

# batch norm config
batch_norm_epsilon = 1E-5
batch_norm_decay = 0.99


# computational TF graph construction ================================
# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 7 # 2nd layer number of neurons
n_hidden_3 = 7 # 3rd layer number of neurons
n_hidden_4 = 4 # 4rd layer number of neurons
n_hidden_5 = 4 # 5rd layer number of neurons


num_input   = xsize   # two-dimensional input X = [1x2]
num_classes = ysize # 2 class
#-------------------------------


# tf Graph input
X           = tf.placeholder(tf.float32, [None, num_input],     name = 'x')
Y           = tf.placeholder(tf.float32, [None, num_classes],   name = 'y')
batchnorm_istraining_io = tf.placeholder(tf.bool,name='bn_phase')

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input,  n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'out':tf.Variable(tf.random_normal([n_hidden_5, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x,batchnorm_istraining_io,batch_norm_epsilon,batch_norm_decay):
    # Input fully connected layer with 10 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.layers.batch_normalization(inputs=layer_1, \
                                            momentum = batch_norm_decay,\
                                            center=True,\
                                            scale= True,\
                                            epsilon= batch_norm_epsilon,\
                                            training = batchnorm_istraining_io)
    layer_1 = tf.nn.relu(layer_1)


    # Hidden fully connected layer with 7 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.layers.batch_normalization(inputs=layer_2, \
                                            momentum=batch_norm_decay, \
                                            center=True,\
                                            scale= True,\
                                            epsilon= batch_norm_epsilon,\
                                            training = batchnorm_istraining_io)
    layer_2 = tf.nn.relu(layer_2)


    # Hidden fully connected layer with 7 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.layers.batch_normalization(inputs=layer_3, \
                                            momentum=batch_norm_decay, \
                                            center=True,\
                                            scale= True,\
                                            epsilon= batch_norm_epsilon,\
                                            training = batchnorm_istraining_io)
    layer_3 = tf.nn.relu(layer_3)

    # Hidden fully connected layer with 4 neurons
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.layers.batch_normalization(inputs=layer_4, \
                                            momentum=batch_norm_decay, \
                                            center=True,\
                                            scale= True,\
                                            epsilon= batch_norm_epsilon,\
                                            training = batchnorm_istraining_io)
    layer_4 = tf.nn.relu(layer_4)

    # Hidden fully connected layer with 4 neurons
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.layers.batch_normalization(inputs=layer_5, \
                                            momentum=batch_norm_decay, \
                                            center=True,\
                                            scale= True,\
                                            epsilon= batch_norm_epsilon,\
                                            training = batchnorm_istraining_io)
    layer_5 = tf.nn.relu(layer_5)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(x=X,\
                    batchnorm_istraining_io = batchnorm_istraining_io ,\
                    batch_norm_epsilon      = batch_norm_epsilon, \
                    batch_norm_decay        = batch_norm_decay)

prediction = tf.nn.softmax(logits)

# Define loss and optimizer
cost        = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

'''
from Tensorflow API doc
(https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm )
Note: when training, the moving_mean and moving_variance need to be updated.
By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
so they need to be added as a dependency to the train_op'''
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer   = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    ## when you use AdamOptimizer, instead of SGD, the error rate immediately becomes near zero.
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# ----------------------------------

# Evaluate model
correct_pred    = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy        = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

errRatebyTrainingSet     = np.zeros(training_epochs)
errRatebyValidationSet   = np.zeros(training_epochs)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training ===============================================
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    print("--------------------------------------------")

    for epoch in range(training_epochs):
        avg_cost = 0.

        for i in range(total_batch):
            data_start_index = i * batch_size
            data_end_index = (i + 1) * batch_size
            # feed traing data --------------------------
            batch_xs = x_training_data[data_start_index:data_end_index, :]
            batch_ts = t_training_data[data_start_index:data_end_index, :]

            #----------------------------------------------
            # Run optimization op (backprop) and cost op (to get loss value)
            # feeding training data
            _, local_batch_cost = sess.run([optimizer,cost], feed_dict={X: batch_xs,\
                                                                        Y: batch_ts, \
                                                                        batchnorm_istraining_io: True })

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

            # for error rate evaluation, the dropout rate must be 1.0, batch_norm is under validation mode
            errRatebyTrainingSet[epoch] = 1.0 - accuracy.eval(feed_dict={X: batch_train_xs, \
                                                                         Y: batch_train_ys, \
                                                                         batchnorm_istraining_io: False},\
                                                              session=sess)

            errRatebyValidationSet[epoch] = 1.0 - accuracy.eval(feed_dict={X: batch_valid_xs, \
                                                                           Y: batch_valid_ys, \
                                                                           batchnorm_istraining_io: False},\
                                                                session=sess)

            print("Training set Err rate: %s"   % errRatebyTrainingSet[epoch])
            print("Validation set Err rate: %s" % errRatebyValidationSet[epoch])
            print("--------------------------------------------")

    print("Optimization Finished!")

    # Calculate accuracy for test images
    ##-------------------------------------------

# Training result visualization ===============================================

hfig1= plt.figure(1,figsize=[10,10])
plt.scatter(data.xdata1.values[0:int(data.xdata1.size/2)],\
            data.xdata2.values[0:int(data.xdata1.size/2)], \
            color='b',label='class0')
plt.scatter(data.xdata1.values[int(data.xdata1.size/2)+2:-1],\
            data.xdata2.values[int(data.xdata1.size/2)+2:-1], \
            color='r',label='class1')
plt.title('Two Spiral data Example')
plt.legend()



hfig2 = plt.figure(2,figsize=(10,10))
epoch_index = np.array([elem for elem in range(training_epochs)])
plt.plot(epoch_index,errRatebyTrainingSet,label='Training data',color='r',marker='o')
plt.plot(epoch_index,errRatebyValidationSet,label='Validation data',color='b',marker='x')
plt.legend()
plt.title('Train/Valid Err with batch norm.' )
plt.xlabel('Iteration epoch')
plt.ylabel('error Rate')
plt.show()

