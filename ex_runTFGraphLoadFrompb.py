#-*- coding: utf-8 -*-
"""
#-----------------------------------------------------------------
  filename: ex_runTFGraphLoadFrompb.py
  objectives:
            - 1) Load the TF graph structure from a ".pb" file

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

filename = 'tf_graph_def.pb'
model_dir = getcwd() + '/model/ex/'
model_filename = os.path.join(model_dir,filename)

graph1 = tf.Graph()

with graph1.as_default():
    # pb 파일 읽어오기
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # Import the graph from "graph_def" into the current default graph
        _ = tf.import_graph_def(graph_def=graph_def,name='')


sess = tf.Session(graph=graph1)