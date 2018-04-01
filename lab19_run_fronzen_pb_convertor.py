#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: lab19_run_fronzen_pb_convertor.py

    description:

        - To convert frozen graph (.pb) from graphdef (.pb) and checkpoint (.ckpt)

    TF version dependency : Tensorflow >= 1.5

    references:
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/mobile/tflite/devguide.md#2-convert-the-model-format
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md#savedmodel

    author: Jaewook Kang
    date  : 2018 Apr
'''

import sys
from os import getcwd

input_model_path    = getcwd()+'/pb_and_ckpt/lenet5/'
output_model_path   = getcwd()+'/pb_and_ckpt/lenet5/frozen_pb_out/'

# The output node name is from Tensorboard
output_node_names   = 'model_out/Softmax'

sys.path.insert(0,  input_model_path)
sys.path.insert(0,  getcwd()+'/tf_lite/')

from tflite_convertor import TFliteConvertor


tflite_convertor = TFliteConvertor()

tflite_convertor.set_config_for_frozen_graph(input_dir_path=input_model_path+'runtrain-20180401140624/',
                                             input_pb_name='tf_graph_def_lenet5.pb',
                                             input_ckpt_name='lenet5_model_variable.ckpt',
                                             output_dir_path=output_model_path,
                                             output_node_names=output_node_names)

tflite_convertor.convert_to_frozen_graph()
