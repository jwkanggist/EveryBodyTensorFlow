#-*- coding: utf-8 -*-
"""
#-----------------------------------------------------------------
  filename: ex_runTFGraphLoadFromckpt.py
  objectives:
            - 1) Load the TF graph structure from a ".ckpt.meta" file
            - 2) Restoring training result to the loaded graph from
                a ".ckpt.data" file

    ref: https://github.com/jonbruner/tensorflow-basics

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

filename = 'model_variable.ckpt-5'
model_dir = getcwd() + '/pb_and_ckpt/ex/'
model_filename = os.path.join(model_dir,filename)


# Load graph from ".ckpt.meta" file
saver = tf.train.import_meta_graph(model_filename+'.meta')
graph = tf.get_default_graph()


with tf.Session() as sess:

    # graph restoring
    saver.restore(sess, model_filename)

    #tf.get_default_graph().as_graph_def()

    # association of variables
    # - Therefore, it is important to label variables with name
    v1 = sess.graph.get_tensor_by_name(name="variable1:0")
    v2 = sess.graph.get_tensor_by_name(name="variable2:0")
    v3 = sess.graph.get_tensor_by_name(name="variable3:0")

    # Check the values of the variables
    print("After model restored.")
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())
    print("v3 : %s" % v3.eval())
