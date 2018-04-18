#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: lab7_runTCcheckGradientVanishing_spiraldata.py

    To check Gradient Vanishing problem in
    A Multi-Hidden Layers Fully Connected Neural Network.

    This example data set is using two class spiral data

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

# #data plot
hfig1= plt.figure(1,figsize=[10,10])
plt.scatter(data.xdata1.values[0:int(data.xdata1.size/2)],\
            data.xdata2.values[0:int(data.xdata1.size/2)], \
            color='b',label='class0')
plt.scatter(data.xdata1.values[int(data.xdata1.size/2)+2:-1],\
            data.xdata2.values[int(data.xdata1.size/2)+2:-1], \
            color='r',label='class1')
plt.title('Two Spiral data Example')
plt.legend()


# configure training parameters =====================================
learning_rate = 1E-5
training_epochs = 5
batch_size = 100
display_step = 1
total_batch = int(training_size / batch_size)


# computational TF graph construction ================================
# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 7 # 2nd layer number of neurons
n_hidden_3 = 7 # 3rd layer number of neurons
n_hidden_4 = 4 # 4rd layer number of neurons
n_hidden_5 = 4 # 5rd layer number of neurons


num_input = xsize   # two-dimensional input X = [1x2]
num_classes = ysize # 2 class

#-------------------------------

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

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
def neural_net(x):
    # Input fully connected layer with 10 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.softmax(layer_1)

    # Hidden fully connected layer with 7 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.softmax(layer_2)

    # Hidden fully connected layer with 7 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.softmax(layer_3)

    # Hidden fully connected layer with 4 neurons
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.softmax(layer_4)

    # Hidden fully connected layer with 4 neurons
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.softmax(layer_5)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
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

# for visualization of vanishing gradient problem
grad_wrt_weight_layer1_tensor = tf.gradients(cost,weights['h1'],\
                                             name='grad_wrt_weight_layer1')
grad_wrt_weight_layer2_tensor = tf.gradients(cost,weights['h2'],\
                                             name='grad_wrt_weight_layer2')
grad_wrt_weight_layer3_tensor = tf.gradients(cost,weights['h3'],\
                                             name='grad_wrt_weight_layer3')
grad_wrt_weight_layer4_tensor = tf.gradients(cost,weights['h4'],\
                                             name='grad_wrt_weight_layer4')
grad_wrt_weight_layer5_tensor = tf.gradients(cost,weights['h5'],\
                                             name='grad_wrt_weight_layer5')

grad_wrt_weight_layer1_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer2_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer3_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer4_iter = np.zeros([total_batch,1])
grad_wrt_weight_layer5_iter = np.zeros([total_batch,1])


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
            # feedign training data
            _, local_batch_cost = sess.run([optimizer,cost], feed_dict={X: batch_xs,
                                                          Y: batch_ts})

            if epoch == training_epochs - 1:
                # print ('Gradient calculation to see gradient vanishing problem')
                _, grad_wrt_weight_layer1 = sess.run([optimizer,grad_wrt_weight_layer1_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts})
                _, grad_wrt_weight_layer2 = sess.run([optimizer,grad_wrt_weight_layer2_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts})
                _, grad_wrt_weight_layer3 = sess.run([optimizer,grad_wrt_weight_layer3_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts})
                _, grad_wrt_weight_layer4 = sess.run([optimizer,grad_wrt_weight_layer4_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts})
                _, grad_wrt_weight_layer5 = sess.run([optimizer,grad_wrt_weight_layer5_tensor], feed_dict={X: batch_xs,
                                                          Y: batch_ts})
                grad_wrt_weight_layer1 = np.array(grad_wrt_weight_layer1)
                grad_wrt_weight_layer2 = np.array(grad_wrt_weight_layer2)
                grad_wrt_weight_layer3 = np.array(grad_wrt_weight_layer3)
                grad_wrt_weight_layer4 = np.array(grad_wrt_weight_layer4)
                grad_wrt_weight_layer5 = np.array(grad_wrt_weight_layer5)

                grad_wrt_weight_layer1 = grad_wrt_weight_layer1.reshape(grad_wrt_weight_layer1.shape[1],
                                                                    grad_wrt_weight_layer1.shape[2])
                grad_wrt_weight_layer2 = grad_wrt_weight_layer2.reshape(grad_wrt_weight_layer2.shape[1],
                                                                    grad_wrt_weight_layer2.shape[2])
                grad_wrt_weight_layer3 = grad_wrt_weight_layer3.reshape(grad_wrt_weight_layer3.shape[1],
                                                                    grad_wrt_weight_layer3.shape[2])
                grad_wrt_weight_layer4 = grad_wrt_weight_layer4.reshape(grad_wrt_weight_layer4.shape[1],
                                                                    grad_wrt_weight_layer4.shape[2])
                grad_wrt_weight_layer5 = grad_wrt_weight_layer5.reshape(grad_wrt_weight_layer5.shape[1],
                                                                    grad_wrt_weight_layer5.shape[2])

                grad_wrt_weight_layer1_iter[i] = grad_wrt_weight_layer1.mean()
                grad_wrt_weight_layer2_iter[i] = grad_wrt_weight_layer2.mean()
                grad_wrt_weight_layer3_iter[i] = grad_wrt_weight_layer3.mean()
                grad_wrt_weight_layer4_iter[i] = grad_wrt_weight_layer4.mean()
                grad_wrt_weight_layer5_iter[i] = grad_wrt_weight_layer5.mean()

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

            errRateTraining[epoch] = 1.0 - accuracy.eval({X: batch_train_xs, \
                                                          Y: batch_train_ys}, session=sess)

            errRateValidation[epoch] = 1.0 - accuracy.eval({X: batch_valid_xs, \
                                                            Y: batch_valid_ys}, session=sess)

            print("Training set Err rate: %s"   % errRateTraining[epoch])
            print("Validation set Err rate: %s" % errRateValidation[epoch])
            print("--------------------------------------------")

    print("Optimization Finished!")

# Training result visualization ===============================================

hfig2 = plt.figure(2,figsize=(10,10))
batch_index = np.array([elem for elem in range(total_batch)])
plt.plot(batch_index,grad_wrt_weight_layer1_iter,label='layer1',color='b',marker='o')
plt.plot(batch_index,grad_wrt_weight_layer4_iter,label='layer4',color='y',marker='o')
plt.plot(batch_index,grad_wrt_weight_layer5_iter,label='layer5',color='r',marker='o')
plt.legend()
plt.title('Weight Gradient over minibatch iter @ training epoch = %s' % training_epochs)
plt.xlabel('minibatch iter')
plt.ylabel('Weight Gradient')


plt.show()