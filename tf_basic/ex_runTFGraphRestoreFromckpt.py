#-*- coding: utf-8 -*-
"""
#-----------------------------------------------------------------
  filename: ex_runTFGraphRestoreFromckpt.py
  objectives:
            - 1) Restoring training results to the an established
            TF computational graph from ckpt files,
            where the computational graph must have the same
            structure with the graph extracting the result.

  ref: http://solarisailab.com/archives/1422

  Written by Jaewook Kang @ 2017 Dec.
#-----------------------------------------------------------------
"""

from os import getcwd
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

import numpy as np
import pandas as pd

tf.reset_default_graph()

# .ckpt로 부터 모델 weight 복구하기
model_dir = getcwd() + '/pb_and_ckpt/ex/'

# construct naive three varible computational graph
v1 = tf.Variable(0, name='variable1')
v2 = tf.Variable(0, name='variable2')
v3 = tf.Variable(0, name='variable3')

init_op = tf.global_variables_initializer()

# Add ops to restore all the variables.
saver = tf.train.Saver()
# 1) max_to_keep : 들고 있는 saver.save()에서 의해서 저장된다.  checkpoint 개수
# 2) keep_checkpoint_every_n_hour: 해당 시점에서 가장 최근에 saver.save()에 의해서 저장된 파라미터를 저장하낟.
# saver = tf.train.Saver(max_to_keep= , keep_checkpoint_every_n_hours=)


with tf.Session() as sess:
    sess.run(init_op)
    print("---------------------------")
    print("Before model restored.")
    print("- v1 : %s" % v1.eval())
    print("- v2 : %s" % v2.eval())
    print("- v3 : %s" % v3.eval())
    print("---------------------------")
    # Later, launch the model, use the saver to restore variables from disk,
    # and do some work with the model.
    saver.restore(sess, model_dir + "model_variable.ckpt-5")

    # Check the values of the variables
    print("After model restored.")
    print("- v1 : %s" % v1.eval())
    print("- v2 : %s" % v2.eval())
    print("- v3 : %s" % v3.eval())
    print("---------------------------")
    print(" The end of graph restoring.")