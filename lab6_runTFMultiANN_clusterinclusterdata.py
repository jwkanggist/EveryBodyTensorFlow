#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: lab6_runTFMultiANN_clusterinclusterdata.py

    A Multi-Hidden Layers Fully Connected Neural Network implementation with TensorFlow.
    This example is using two class cluster in cluster data

    written by Jaewook Kang @ Sep 2017
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
from datetime import datetime

# reading data set from csv file ==========================
xsize  = 2
ysize  = 2

data = pd.read_csv('./data/clusterincluster_N5000.csv')
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

# #data plot
hfig1= plt.figure(1,figsize=[10,10])
plt.scatter(data.xdata1.values[0:int(data.xdata1.size/2)],\
            data.xdata2.values[0:int(data.xdata1.size/2)], \
            color='b',label='class0')
plt.scatter(data.xdata1.values[int(data.xdata1.size/2)+2:-1],\
            data.xdata2.values[int(data.xdata1.size/2)+2:-1], \
            color='r',label='class1')
plt.title('Cluster in Cluster data Example')
plt.legend()



# configure training parameters =====================================
learning_rate = 1E-3
training_epochs = 100
batch_size = 100
display_step = 1


# computational TF graph construction ================================
# Network Parameters
n_hidden_1 = 7 # 1st layer number of neurons
n_hidden_2 = 5 # 2nd layer number of neurons
num_input = xsize   # two-dimensional input X = [1x2]
num_classes = ysize # 2 class

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 5 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden fully connected layer with 5 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

errRateTraining     = np.zeros(training_epochs)
errRateValidation   = np.zeros(training_epochs)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

now             = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir     = 'export/lab6_cluster/tf_logs'
logdir          = "{}/run-{}/".format(root_logdir,now)

summary_writer  = tf.summary.FileWriter(logdir=logdir)
summary_writer.add_graph(tf.get_default_graph())

loss_summary        = tf.summary.scalar('loss',cost)
accuracy_summary    = tf.summary.scalar('accuracy',accuracy)


# Start training ===============================================
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # save graph model
    tf.train.write_graph(sess.graph_def,getcwd() + '/export/lab6/pb','tfgraph_clusterincluster_ann_lab6.pbtxt')
    tf.train.write_graph(sess.graph_def,getcwd() + '/export/lab6/pb','tfgraph_clusterincluster_ann_lab6.pb',as_text =False)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(training_size/batch_size)

        for i in range(total_batch):
            data_start_index = i * batch_size
            data_end_index = (i + 1) * batch_size
            # feed traing data --------------------------
            batch_xs = x_training_data[data_start_index:data_end_index, :]
            batch_ts = t_training_data[data_start_index:data_end_index, :]

            #----------------------------------------------
            # Run optimization op (backprop) and cost op (to get loss value)
            # feedign training data
            _, local_batch_cost = sess.run([optimizer,cost], feed_dict={X: batch_xs,
                                                          Y: batch_ts})

            # Compute average loss
            avg_cost += local_batch_cost / total_batch
            # print ("At %d-th batch in %d-epoch, avg_cost = %f" % (i,epoch,avg_cost) )

            summary_str = accuracy_summary.eval(feed_dict={X: batch_xs, Y: batch_ts})
            summary_writer.add_summary(summary_str, epoch*training_epochs + i)

            # Display logs per epoch step
        if display_step == 0:
            continue
        elif (epoch + 1) % display_step == 0:
            # print("Iteration:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            batch_train_xs = x_training_data
            batch_train_ys = t_training_data
            batch_valid_xs = x_validation_data
            batch_valid_ys = t_validation_data

            errRateTraining[epoch] = 1.0 - accuracy.eval({X: batch_train_xs, \
                                                          Y: batch_train_ys}, session=sess)

            errRateValidation[epoch] = 1.0 - accuracy.eval({X: batch_valid_xs, \
                                                            Y: batch_valid_ys}, session=sess)

            print("Training set Err rate: %s"   % errRateTraining[epoch])
            print("Validation set Err rate: %s" % errRateValidation[epoch])

        print("--------------------------------------------")

    print("Optimization Finished!")

summary_writer.close()

# Training result visualization ===============================================
hfig2 = plt.figure(2,figsize=(10,10))
epoch_index = np.array([elem for elem in range(training_epochs)])
plt.plot(epoch_index,errRateTraining,label='Training data',color='r',marker='o')
plt.plot(epoch_index,errRateValidation,label='Validation data',color='b',marker='x')
plt.legend()
plt.title('Classification Error Rate of prediction:')
plt.xlabel('Iteration epoch')
plt.ylabel('error Rate')


