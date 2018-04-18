#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
  filename: lab2_runHelloTensorflow.py
  written by Jaewook Kang @ Sep 2017
#------------------------------------------------------------
'''


import tensorflow as tf

hello   = tf.constant('Hello, TensorFlow!')
sess    = tf.Session()
print '[Lab1] %s' % sess.run(hello)
# Hello, TensorFlow!
a = tf.constant(10)
b = tf.constant(32)
print '[Lab1] a + b = %s' % sess.run(a+b)
# 42
