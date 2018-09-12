#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_tf_basic_matmul.py

    This script is for implementation of a basic rnn network
    using tf.matmul() and tf.tanh()

    author: Jaewook Kang @ 2018 Sep
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import numpy as np
import tensorflow as tf

model_config = \
{
    'batch_size': None,
    'n_input'   : 3,
    'n_output'  : 5,
    'num_steps' : 2,
    'dtype'     : tf.float32
}


def get_rnn_model(X0,X1,scope):

    Wx_shape = [model_config['n_input'],
                model_config['n_output']]

    Wh_shape = [model_config['n_output'],
                model_config['n_output']]

    bias_shape    = [1,model_config['n_output']]

    with tf.name_scope(name=scope,values=[X0,X1]):

        Wx = tf.get_variable(name='weight_x',
                             shape=Wx_shape,
                             initializer=tf.random_normal_initializer)

        Wh = tf.get_variable(name='weight_h',
                             shape=Wh_shape,
                             initializer=tf.random_normal_initializer)
        b  = tf.get_variable(name='bias',
                             shape=bias_shape,
                             initializer=tf.random_normal_initializer)

        Y0 = tf.tanh(tf.matmul(X0,Wx) + b)
        Y1 = tf.tanh(tf.matmul(X1,Wx) + tf.matmul(Y0,Wh) + b)



    return Y0,Y1



if __name__ == '__main__':

    input_shape     = [model_config['batch_size'],
                       model_config['n_input']]

    output_shape    = [model_config['batch_size'],
                       model_config['n_output']]


    X0 = tf.placeholder(dtype = model_config['dtype'],
                        shape = input_shape,
                        name  = 'X0')

    X1 = tf.placeholder(dtype= model_config['dtype'],
                        shape= input_shape,
                        name = 'X1')

    scope = 'basic_rnn_matmul_model'
    Y0,Y1 = get_rnn_model(X0,X1,scope)


    # tensorboard summary
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = 'tf_logs/rnn_basic_matmul'
    subdir = "{}/run-{}/".format(root_logdir, now)

    logdir = './pb_and_ckpt/' + subdir

    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)


    summary_writer = tf.summary.FileWriter(logdir=logdir)
    summary_writer.add_graph(graph=tf.get_default_graph())


    init = tf.global_variables_initializer()



    X0_batch = np.array([[0,1,2],
                         [3,4,5],
                         [6,7,8],
                         [9,0,1],
                         ])
    X1_batch = np.array([[9,8,7],
                         [6,5,4],
                         [3,2,1],
                         [0,9,8]])

    with tf.Session() as sess:

        sess.run(init)

        Y0_val,Y1_val = sess.run(fetches=[Y0,Y1],feed_dict={X0:X0_batch,
                                                            X1:X1_batch})


    print('Y0_val = %s' % Y0_val)
    print('Y1_val = %s' % Y1_val)
    summary_writer.close()







