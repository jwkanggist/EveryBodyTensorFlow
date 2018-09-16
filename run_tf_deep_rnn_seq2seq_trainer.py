#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: run_tf_deep_rnn_seq2seq_trainer.py

    This script is for predicting time series

    author: Jaewook Kang @ 2018 Sep
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model_config = \
{
    'n_input'   : 1,
    'n_neurons' : 200,
    'n_layers'  : 3,
    'n_output'  : 1,
    'num_steps' : 30,
    'dtype'     : tf.float32
}


training_config = \
    {
        'learning_rate': 0.001,
        'n_iteration':2000
    }


def get_deep_rnn_seq2seq_model(X,scope):


    with tf.name_scope(name=scope,values=[X]):

        layers  = [tf.nn.rnn_cell.BasicRNNCell(num_units=model_config['n_neurons'],
                                        activation= tf.nn.relu,
                                        name='basic_rnn_cell')\
                                        for layer in range(model_config['n_layers'])]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

        rnn_outputs, states = tf.nn.dynamic_rnn(cell=multi_layer_cell,
                                                inputs=X,
                                                dtype=model_config['dtype'])


        stacked_rnn_outputs = tf.reshape(rnn_outputs,
                                         shape=[-1,
                                                model_config['n_neurons']])

        stacked_logits      = tf.layers.dense(stacked_rnn_outputs,
                                              model_config['n_output'])
        logits              = tf.reshape(stacked_logits,
                                         shape=[-1,
                                                model_config['num_steps'],
                                                model_config['n_output']])



    return logits




def gen_seq_data(shift_sample,sqe_sample_length):

    data_step = 0.1
    start_n = np.random.random_integers(low=0, high=30)
    tx = np.arange(start=start_n, stop=start_n + sqe_sample_length*data_step, step=data_step)
    ty = tx + shift_sample * data_step

    x_batch = tx * np.sin(tx) / 3 + 2 * np.sin(5 * tx)
    y_batch = ty * np.sin(ty) / 3 + 2 * np.sin(5 * ty)

    return x_batch, y_batch, tx, ty




if __name__ == '__main__':

    input_shape     = [1,
                       model_config['num_steps'],
                       model_config['n_input']]

    output_shape    = [1,
                       model_config['num_steps'],
                       model_config['n_output']]


    X = tf.placeholder(dtype = model_config['dtype'],
                        shape = input_shape,
                        name  = 'X')


    Y = tf.placeholder(dtype = model_config['dtype'],
                        shape = output_shape,
                        name  = 'Y')

    # build model
    scope   = 'deep_rnn_seq2seq_model'
    pred_y = get_deep_rnn_seq2seq_model(X,scope)

    loss    = tf.reduce_mean(tf.square(pred_y - Y))

    optimizer   = tf.train.AdamOptimizer(learning_rate=training_config['learning_rate'])
    training_op = optimizer.minimize(loss)



    # tensorboard summary
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = 'tf_logs/rnn_deep_seq2seq_trainer'
    subdir = "{}/run-{}/".format(root_logdir, now)

    logdir = './pb_and_ckpt/' + subdir

    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)


    summary_writer = tf.summary.FileWriter(logdir=logdir)
    summary_writer.add_graph(graph=tf.get_default_graph())


    init = tf.global_variables_initializer()

    n_iteration    =   training_config['n_iteration']

    with tf.Session() as sess:

        sess.run(init)
        shift_sample =2
        for iteration in range(n_iteration):

            x_batch,y_batch,tx_train,ty_train =gen_seq_data(shift_sample=shift_sample,
                                                            sqe_sample_length = model_config['num_steps'])
            x_test, y_test, tx_test, ty_test = gen_seq_data(shift_sample=shift_sample,
                                                            sqe_sample_length=model_config['num_steps'])

            x_batch          = x_batch.reshape((-1,\
                                              model_config['num_steps'],
                                              model_config['n_input']))
            y_batch          = y_batch.reshape((-1, \
                                              model_config['num_steps'],
                                              model_config['n_output']))

            x_test          = x_test.reshape((-1,\
                                              model_config['num_steps'],
                                              model_config['n_input']))
            y_test          = y_test.reshape((-1, \
                                              model_config['num_steps'],
                                              model_config['n_output']))


            _, mse_train,pred_y_train = sess.run([training_op,loss,pred_y],
                                              feed_dict={X:x_batch,
                                                         Y:y_batch})

            mse_test,pred_y_test = sess.run([loss,pred_y],
                                              feed_dict={X:x_test,
                                                         Y:y_test})

            print(iteration,"Train accuracy:", mse_train, "Test accuracy:", mse_test)


    x_batch         = x_batch.reshape((model_config['num_steps']))
    y_batch         = y_batch.reshape((model_config['num_steps']))
    pred_y_train    = pred_y_train.reshape((model_config['num_steps']))

    x_test          = x_test.reshape((model_config['num_steps']))
    y_test          = y_test.reshape((model_config['num_steps']))
    pred_y_test     = pred_y_test.reshape((model_config['num_steps']))

    plt.figure(1)
    plt.plot(tx_train,x_batch,color='b',marker='o',label='train_input')
    plt.plot(ty_train,y_batch,color='r',marker='x',label='train_output')
    plt.plot(ty_train,pred_y_train,color='m',marker='x',label='pred_output')
    plt.legend()

    plt.figure(2)
    plt.plot(tx_test,x_test,color='b',marker='o',label='test_input')
    plt.plot(ty_test,y_test,color='r',marker='x',label='test_output')
    plt.plot(ty_test,pred_y_test,color='m',marker='x',label='pred_output')
    plt.legend()
    plt.show()

    summary_writer.close()




