#-*- coding: utf-8 -*-

"""
#---------------------------------------------
  filename: ex_runTFconstant.py
  - tensorflow의 기본 계산 그래프를 생성하고 평가해본다
  - tf.constant()을 사용해 본다
  Written by Jaewook Kang
  2017 Aug.
#-------------------------------------------
"""

import tensorflow as tf
import numpy as np
import pandas as pd

g = tf.Graph()

with g.as_default():
    x = tf.constant(6,name="x_const")
    y = tf.constant(14,name="y_const")


    sum = tf.add(x,y,name="sum_xy")


with tf.Session(graph=g) as sess:
    print ('sum.eval()=%d'%sum.eval())
    print ('sess.run(sum)=%d'%sess.run(sum))

