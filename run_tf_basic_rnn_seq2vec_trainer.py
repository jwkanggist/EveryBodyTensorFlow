#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_tf_basic_rnn_sqe2vec_trainer.py

    This script is for predicting mnist images

    author: Jaewook Kang @ 2018 Sep
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

model_config = \
{
    'batch_size': None,
    'n_input'   : 28,
    'num_steps': 28,
    'n_neurons': 150,
    'n_output'  : 10,
    'dtype'     : tf.float32
}


training_config = \
    {
        'learning_rate': 0.001,
        'n_epochs':10,
        'batch_size':150
    }


def get_rnn_dynamic_model(X,scope):

    with tf.name_scope(name=scope,values=[X]):
        # X : 28 x 1
        basic_cell  = tf.nn.rnn_cell.BasicRNNCell(num_units=model_config['n_neurons'],
                                                  name='basic_rnn_cell')
        # states : 150 x 1
        outputs, states = tf.nn.dynamic_rnn(cell=basic_cell,
                                      inputs=X,
                                      dtype=model_config['dtype'])

        # logits: 10 x 1
        logits = tf.layers.dense(states,model_config['n_output'])

    return logits




if __name__ == '__main__':

    # dataset preparation

    mnist = input_data.read_data_sets("/tmp/data/")


    input_shape     = [model_config['batch_size'],
                       model_config['num_steps'],
                       model_config['n_input']]

    output_shape    = [model_config['batch_size'],
                       model_config['num_steps'],
                       model_config['n_output']]


    X = tf.placeholder(dtype = model_config['dtype'],
                        shape = input_shape,
                        name  = 'X')


    Y =tf.placeholder(dtype=tf.int32,
                      shape=[None])

    # build model
    scope   = 'rnn_seq2vec_model'
    logits  = get_rnn_dynamic_model(X,scope)

    loss    = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,
                                                                            logits=logits))

    optimizer   = tf.train.AdamOptimizer(learning_rate=training_config['learning_rate'])
    training_op = optimizer.minimize(loss)

    correct     = tf.nn.in_top_k(logits,Y,1)
    accuracy    = tf.reduce_mean(tf.cast(correct,tf.float32))



    # tensorboard summary
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = 'tf_logs/rnn_basic_seq2vec_trainer'
    subdir = "{}/run-{}/".format(root_logdir, now)

    logdir = './pb_and_ckpt/' + subdir

    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)


    summary_writer = tf.summary.FileWriter(logdir=logdir)
    summary_writer.add_graph(graph=tf.get_default_graph())


    init = tf.global_variables_initializer()

    n_epochs    =   training_config['n_epochs']
    batch_size  =   training_config['batch_size']

    with tf.Session() as sess:

        sess.run(init)
        for epoch in range(n_epochs):

            x_test = mnist.test.images.reshape((-1,
                                                model_config['num_steps'],
                                                model_config['n_input']))
            y_test = mnist.test.labels

            for iteration in range(mnist.train.num_examples // batch_size):

                x_batch, y_batch = mnist.train.next_batch(batch_size)
                x_batch          = x_batch.reshape((-1,
                                                    model_config['num_steps'],
                                                    model_config['n_input']))
                sess.run(training_op,feed_dict={X:x_batch,
                                                Y:y_batch})

            acc_train = accuracy.eval(feed_dict={X:x_batch,
                                                 Y:y_batch})
            acc_test  = accuracy.eval(feed_dict={X:x_test,
                                                 Y:y_test})

            print(epoch,"Train accuracy:", acc_train, "Test accuracy:", acc_test)


        x_true,y_true = mnist.train.next_batch(1)
        x_true = x_true.reshape((-1,
                                 model_config['num_steps'],
                                  model_config['n_input']))

        y_pred = sess.run([logits], feed_dict={X:x_true})
        x_true = x_true.reshape((model_config['num_steps'],
                                  model_config['n_input']))

    plt.figure(1)
    imgplot = plt.imshow(x_true)
    plt.title('true = %s, pred = %s' %(y_true[0],np.argmax(y_pred)))
    plt.show()
    summary_writer.close()




