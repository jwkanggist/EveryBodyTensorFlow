#-*- coding: utf-8 -*-

"""
#---------------------------------------------
  filename: ex_runTFfetch.py
  - Construct a computational graph consisting of
    multiple tf.constant() tensors in Tensorflow

  Written by Jaewook Kang
  2017 Aug.
#-------------------------------------------
"""


import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
temporal_add_result = tf.add(input2, input3)
mul = tf.multiply(input1, temporal_add_result)

with tf.Session() as sess:
  result = sess.run([mul, temporal_add_result])
  result_mul = mul.eval()
  result_intermed = temporal_add_result.eval()

  print('Result by sess.run(): %s' % result)
  print('Result by mul.eval(): %s' % result_mul)
  print('Result by temporal_add_result.eval(): %s' % result_intermed)

# 출력:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]