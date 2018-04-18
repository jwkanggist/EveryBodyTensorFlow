
#-*- coding: utf-8 -*-
"""
#-----------------------------------------------------------------
  filename: ex_runTFGraphSave.py
  objectives:
            - 1) save TF computational graph using
                tf.train.write_graph() operation
            - 2) save checkpoint of training result using
                tf.train.Savor() operation

  Written by Jaewook Kang @ 2017 Dec.
#-----------------------------------------------------------------
"""

from os import getcwd

import tensorflow as tf
import numpy as np
import pandas as pd

tf.reset_default_graph()

# construct naive three varible computational graph
# - Note that the variable naming is important to get association
# - between python and graph variables in graph restoring.

v1 = tf.Variable(0, name='variable1')
v2 = tf.Variable(2, name='variable2')
v3 = tf.Variable(3, name='variable3')

init_op = tf.global_variables_initializer()

# savor operation to save
saver = tf.train.Saver()  # including all variable
#saver = tf.train.Saver([v1, v2, v3])  # specifying variable to checkpoint
graph = tf.get_default_graph()

# period of checkpoint
ckpt_period = 5

with tf.Session(graph=graph) as sess:
    # 텍스트 파일 형식으로 저장
    savedir = getcwd() + '/pb_and_ckpt/ex'
    filename_pbtxt = 'tf_graph_def.pbtxt'
    filename_pb = 'tf_graph_def.pb'

    print ("---------------------------------------------------------")
    # STEP1) TF 계산 그래프를 정의하고  .pb파일로 저장
    tf.train.write_graph(sess.graph_def, savedir,filename_pbtxt )
    print ("TF graph_def is save in txt at %s." % savedir+filename_pbtxt)
    # 바이너리로 저장
    tf.train.write_graph(sess.graph_def, savedir, filename_pb , as_text=False)
    print ("TF graph_def is save in binary at %s." % savedir+filename_pb)
    print ("---------------------------------------------------------")

    sess.run(init_op)

    print("The init value of variables in the graph:")
    print("- v2 : %s" % v2.eval())
    print("- v3 : %s" % v3.eval())
    print ("---------------------------------------------------------")

    # 훈련 코드
    for step in range(10):
        # some training codes here
        if step % ckpt_period == 0:
            # STEP 2) 모델 훈련 결과값 (variable값) 를 checkpoint로 저장하기
            # step 별로 파일명 다르게 해서 Checkpoint 저장
            save_path = saver.save(sess, getcwd() + "/pb_and_ckpt/ex/model_variable.ckpt", global_step=step)
            #save_path = saver.save(sess, getcwd() + "/model/ex/model_variable.ckpt")

            print("step - %s: Model saved in file: %s" % (step, save_path))
    print ("---------------------------------------------------------")
    print ('Training finished.')