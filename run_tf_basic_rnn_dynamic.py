#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_tf_basic_dynamic.py

    This script is for implementation of a basic rnn network
    using tf.nn.rnn_cell.BasicRNNCell(),tf.contrib.rnn.dynamic_rnn()

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


def get_rnn_dynamic_model(X,scope):


    with tf.name_scope(name=scope,values=[X]):
        basic_cell  = tf.nn.rnn_cell.BasicRNNCell(num_units=model_config['n_output'],
                                                  name='basic_rnn_cell')
        pred_y, states = tf.nn.dynamic_rnn(cell=basic_cell,
                                      inputs=X,
                                      dtype=model_config['dtype'])
    return pred_y



if __name__ == '__main__':

    input_shape     = [model_config['batch_size'],
                       model_config['num_steps'],
                       model_config['n_input']]

    output_shape    = [model_config['batch_size'],
                       model_config['num_steps'],
                       model_config['n_output']]


    X = tf.placeholder(dtype = model_config['dtype'],
                        shape = input_shape,
                        name  = 'X')



    scope = 'basic_rnn_dynamic_model'
    pred_y = get_rnn_dynamic_model(X,scope)


    # tensorboard summary
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = 'tf_logs/rnn_basic_dynamic'
    subdir = "{}/run-{}/".format(root_logdir, now)

    logdir = './pb_and_ckpt/' + subdir

    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)


    summary_writer = tf.summary.FileWriter(logdir=logdir)
    summary_writer.add_graph(graph=tf.get_default_graph())


    init = tf.global_variables_initializer()



    X_batch = np.array( [
                            [[0,1,2],[9,8,7]],
                            [[3,4,5],[6,5,4]],
                            [[6,7,8],[3,2,1]],
                            [[9,0,1],[0,0,0]]
                        ])

    with tf.Session() as sess:

        sess.run(init)

        Y_val = sess.run(fetches=[pred_y],feed_dict={X:X_batch})


    print('Y_val = %s' % Y_val)
    summary_writer.close()







