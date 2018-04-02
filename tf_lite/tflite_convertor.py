#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: tflite_convertor.py

    description:

        - To convert frozen graph (.pb) from graphdef (.pb) and checkpoint (.ckpt)
        - To convert tflite (.tflite) format from frozen graph (.pb)

    TF version dependency : Tensorflow >= 1.5

    references:
        - the newest ref:   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md#savedmodel
        - outdated ref:     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/mobile/tflite/devguide.md#2-convert-the-model-format

        - freeze_graph.py       : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
        - toco_from_protos.py   : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco

        - https://www.tensorflow.org/mobile/prepare_models
    author: Jaewook Kang
    date  : 2018 Apr
'''

from tensorflow.python.tools import freeze_graph
from os import chdir
from os import getcwd
from os import path
from subprocess import check_output



class TfliteConfig(object):

    def __init__(self):
        self.input_dir_path         = str()
        self.output_dir_path        = str()

        self.input_pb_file          = str()
        self.output_tflite_file     = str()
        self.inference_type         = str()
        self.input_shape            = str()
        self.input_array            = str()
        self.output_array           = str()

        self.tf_src_dir_path = str()


    def set_config(self, input_dir_path, output_dir_path,
                        input_pb_file, output_tflite_file,
                        inference_type, input_shape, input_array, output_array,
                        tf_src_dir_path):

        self.input_dir_path     = input_dir_path        # the pathname the input frozen graph stored
        self.output_dir_path    = output_dir_path       # the pathname the output tflite file will be saved

        self.input_pb_file      = input_pb_file         # input frozen graph filename
        self.output_tflite_file = output_tflite_file    # output tflite filename
        self.inference_type     = inference_type        # data type of weight in tflite
        self.input_shape        = input_shape           # getting from Tensorboard
        self.input_array        = input_array           # getting from Tensorboard
        self.output_array       = output_array          # getting from Tensorboard

        self.tf_src_dir_path    = tf_src_dir_path

        if not path.exists(self.output_dir_path):
            check_output('mkdir ' + output_dir_path, shell=True)

        self.show_config()


    def reset(self):
        self.input_dir_path         = str()
        self.output_dir_path        = str()

        self.input_pb_file      = str()
        self.output_tflite_file = str()
        self.inference_type     = str()
        self.input_shape        = str()
        self.input_array        = str()
        self.output_array       = str()
        
    def show_config(self):
        print ('# ------------------------------------------------ ')

        print ('# [TfliteConfig] input_dir_path: %s' % self.input_dir_path)
        print ('# [TfliteConfig] output_dir_path: %s' % self.output_dir_path)

        print ('# [TfliteConfig] input_pb_file: %s' % self.input_pb_file)
        print ('# [TfliteConfig] output_tflite_file: %s' % self.output_tflite_file)
        print ('# [TfliteConfig] inference_type: %s' % self.inference_type)
        print ('# ------------------------------------------------ ')




class FreezeGraphConfig (object):

    def __init__(self):

        self.input_dir_path     = str() # dir path where input pb/ckpt files are contained
        self.input_pb_name      = str() # input pb filename
        self.input_ckpt_name    = str() # input ckpt filename
        self.binary_opt         = True  # This is True always

        self.output_dir_path    = str() # dir path where output frozen pb file will be saved
        self.output_pb_name     = str() # output frozen pb filename
        self.output_node_names  = str() # the name of the output node of the target model




    def set_config(self,input_dir_path,input_pb_name,input_ckpt_name,
                 output_dir_path,output_node_names):

        self.input_dir_path     = input_dir_path
        self.input_pb_name      = input_pb_name
        self.input_ckpt_name    = input_ckpt_name
        self.binary_opt         = True

        self.output_dir_path    = output_dir_path
        self.output_pb_name     = 'frozen_' + self.input_pb_name
        self.output_node_names  = output_node_names

        if not path.exists(self.output_dir_path):
            check_output('mkdir ' + output_dir_path, shell=True)

        self.show_config()




    def reset(self):
        self.input_dir_path     = str()
        self.input_pb_name      = str()
        self.input_ckpt_name    = str()
        self.binary_opt         = True

        self.output_dir_path    = str()
        self.output_pb_name     = str()
        self.output_node_names  = str()




    def show_config(self):
        print ('# ------------------------------------------------ ')

        print ('# [FreezeGraphConfig] input_dir_path: %s' % self.input_dir_path)
        print ('# [FreezeGraphConfig] output_dir_path: %s' % self.output_dir_path)

        print ('# [FreezeGraphConfig] input_pb_name: %s' % self.input_pb_name)
        print ('# [FreezeGraphConfig] input_ckpt_name: %s' % self.input_ckpt_name)

        print ('# [FreezeGraphConfig] output_pb_name: %s' % self.output_pb_name)
        print ('# [FreezeGraphConfig] output_node_name: %s' % self.output_node_names)

        print ('# ------------------------------------------------ ')




class TFliteConvertor(object):

    def __init__(self):
        self.frozenpb_config_worker = FreezeGraphConfig()
        self.tflite_config_worker   = TfliteConfig()



    def set_config_for_frozen_graph(self,input_dir_path,
                                    input_pb_name,
                                    input_ckpt_name,
                                    output_dir_path,
                                    output_node_names):


        self.frozenpb_config_worker.set_config(input_dir_path=input_dir_path,
                                                input_pb_name=input_pb_name,
                                                input_ckpt_name=input_ckpt_name,
                                                output_dir_path=output_dir_path,
                                                output_node_names=output_node_names)




    def set_config_for_tflite(self,input_dir_path,
                                    output_dir_path,
                                    input_pb_file,
                                    output_tflite_file,
                                    inference_type,
                                    input_shape,
                                    input_array,
                                    output_array,
                                    tf_src_dir_path):



        self.tflite_config_worker.set_config(input_dir_path = input_dir_path,
                                             output_dir_path= output_dir_path,
                                             input_pb_file=input_pb_file,
                                             output_tflite_file=output_tflite_file,
                                             inference_type=inference_type,
                                             input_shape=input_shape,
                                             input_array=input_array,
                                             output_array=output_array,
                                             tf_src_dir_path=tf_src_dir_path)




    def convert_to_frozen_graph(self):

        input_pb_path           = self.frozenpb_config_worker.input_dir_path + self.frozenpb_config_worker.input_pb_name
        input_ckpt_path         = self.frozenpb_config_worker.input_dir_path + self.frozenpb_config_worker.input_ckpt_name
        output_frozen_pb_path   = self.frozenpb_config_worker.output_dir_path+ self.frozenpb_config_worker.output_pb_name

        freeze_graph.freeze_graph(\
            input_graph=input_pb_path,
            input_saver= "",                    # this argument is used with SavedModel
            input_binary=self.frozenpb_config_worker.binary_opt,
            input_checkpoint=input_ckpt_path,
            output_node_names=self.frozenpb_config_worker.output_node_names,
            restore_op_name="save/restore_all",  # unused in freeze_graph()
            filename_tensor_name="save/Const:0", # unused in freeze_graph()
            output_graph=output_frozen_pb_path,
            clear_devices=False,                # not clear how to use
            initializer_nodes="")
           # input_meta_graph=input_ckpt_path+".meta")




    def convert_to_tflite_from_frozen_graph(self):

        input_frozen_pb_path = self.tflite_config_worker.input_dir_path  + self.tflite_config_worker.input_pb_file
        output_tflite_path   = self.tflite_config_worker.output_dir_path + self.tflite_config_worker.output_tflite_file


        # toco arguments
        toco_cmd2 = ' --input_file=' + input_frozen_pb_path
        toco_cmd3 = ' --output_file='+ output_tflite_path
        toco_cmd4 = ' inference_type=' + self.tflite_config_worker.inference_type
        toco_cmd5 = ' input_shape='    + self.tflite_config_worker.input_shape
        toco_cmd6 = ' input_array='    + self.tflite_config_worker.input_array
        toco_cmd7 = ' output_array='   + self.tflite_config_worker.output_array

        # dir path change
        curr_path = getcwd()
        chdir(self.tflite_config_worker.tf_src_dir_path)

        # bazel clean
        bazel_clean_cmd = 'bazel clean --expunge'
        shell_out = check_output(bazel_clean_cmd,shell=True)
        print ('$ ' + bazel_clean_cmd)
        print ('> ' + shell_out)

        # main toco command
        toco_build_cmd1 = 'bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- '
        cmd = toco_build_cmd1 + toco_cmd2 \
                        + toco_cmd3 \
                        + toco_cmd4 \
                        + toco_cmd5 \
                        + toco_cmd6 \
                        + toco_cmd7



        print ('Dir path move to %s' % self.tflite_config_worker.tf_src_dir_path)
        print ('$ '+ cmd)
        shell_out = check_output(cmd, shell=True)


        # toco_cmd1 = 'toco '
        # cmd = toco_cmd1 + toco_cmd2 \
        #                 + toco_cmd3 \
        #                 + toco_cmd4 \
        #                 + toco_cmd5 \
        #                 + toco_cmd6 \l
        #                 + toco_cmd7

        #
        # print ('$ '+ toco_build_cmd1)
        # shell_out = check_output(toco_build_cmd1, shell=True)



        print ('> '+ shell_out)
        print ('Dir path return to %s' % curr_path)
        chdir(curr_path)