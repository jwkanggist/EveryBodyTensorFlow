#-*- coding: utf-8 -*-

"""
#-----------------------------------------------------------------
  filename: ex_runTFfeed.py
  - Feed training data into the computational graph in tf.Session
  - Use tf.placeholder()
  Written by Jaewook Kang
  2017 Aug.
#-----------------------------------------------------------------
"""

import tensorflow as tf



input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
mul_op = tf.multiply(input1, input2)

with tf.Session() as sess:

  # output 텐서를 그래프로 올리고 그 입력으로 아래를 계산그래프 내에서 할당
  # input1에 7.
  # input2에 2.
  print(sess.run([mul_op], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]