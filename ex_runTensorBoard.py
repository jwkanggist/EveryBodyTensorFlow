#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: ex_runTensorBoard.py
    objectives: provide an example of Tensorboard using the
    "lab6_runTFMultiANN_MNIST.py" example.


    refs:
    - google doc: https ://www.tensorflow.org/get_started/summaries_and_tensorboard
    - Github: https     ://github.com/tensorflow/tensorboard
    - Tensorboard bugfix: https://github.com/dmlc/tensorboard/issues/36

    written by Jaewook Kang @ Dec 2017
#------------------------------------------------------------
'''


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# file setup for tensorboard record
from datetime import datetime
now  = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir="{}/run-{}/".format(root_logdir,now) #
#----------------

# training Parameters
learning_rate   = 0.01
num_steps       = 2000
batch_size      = 128*2
display_step    = 100


# Network Parameters
n_hidden_1  = 256 # 1st layer number of neurons
n_hidden_2  = 256 # 2nd layer number of neurons
num_input   = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input],name='input_data')
Y = tf.placeholder("float", [None, num_classes],name='output_data')

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]),name='h1_weight'),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='h2_weight'),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]),name='out_weight')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name='h1_bias'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]),name='h2_bias'),
    'out': tf.Variable(tf.random_normal([num_classes]),name='out_bias')
}


# Create model
def neural_net(x):
    # Hidden fully connected layer1 with 256 neurons
    with tf.name_scope('layer_1'):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.softmax(layer_1,name='layer_1_out')

    # Hidden fully connected layer2 with 256 neurons
    with tf.name_scope('layer_2'):
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.softmax(layer_2,name='layer_2_out')

    # Output fully connected layer with a neuron for each class
    with tf.name_scope('layer_out'):
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

# Construct model
logits = neural_net(X)

with tf.name_scope('model_output'):
    prediction = tf.nn.softmax(logits,name='pred_y')

# Define loss functions
with tf.name_scope("loss"):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y),name='loss')

# Define performance measure
with tf.name_scope("performance"):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy')

# Define optimizers
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name='Adamopt_op')
    train_op = optimizer.minimize(loss_op,name='train_op')



# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# summary for Tensorborard visualization
accuracy_summary    = tf.summary.scalar('accuracy',accuracy)
loss_summary        = tf.summary.scalar('loss',loss_op)


# file writing for Tensorboard
file_writer         = tf.summary.FileWriter(logdir=logdir)
file_writer.add_graph(tf.get_default_graph())


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # recording with tensorboard
        # 1) accuracy visualization
        summary_str = accuracy_summary.eval(feed_dict={X: batch_x, Y: batch_y})
        file_writer.add_summary(summary_str,step)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))

# close tensorboard recording
file_writer.close()