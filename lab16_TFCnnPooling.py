#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
#------------------------------------------------------------
    filename: lab16_runCNNPooling.py

    Generation of exemplary feature of convolutional neural network
    using max-pooling operation

    This example is originally given by
    A. Geron "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
    O'REILLY 2017. page366

    written by Jaewook Kang @ Nov 2017
#------------------------------------------------------------
'''
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib.pyplot as plt

# load images
china   = load_sample_image("china.jpg")
flower  = load_sample_image("flower.jpg")
dataset = np.array([china,flower],dtype=np.float32)

# load images data set size
batch_size,height, width, channels = dataset.shape


X = tf.placeholder(tf.float32, shape=[None, height, width, channels],name='input')

stride = 4
tile_size = 4
# prediction CNN with two filters and input X
# X is the input mini-batch
# 4 X 4 tiny kernel is used for max pooling: ksize=[batch_size=1,height=4,width=4,channels=1]
# padding = 'VAILD', which means the conv layer does not use zero padding
pooling_output = tf.nn.max_pool(X,ksize=[1,tile_size,tile_size,1],strides=[1,stride,stride,1],padding='VALID')



with tf.Session() as sess:
    output = sess.run(pooling_output,feed_dict= {X:dataset})

hfig = plt.figure(1,figsize=(5,15))
plt.subplot(4,2,1)
plt.imshow(china)
plt.title('The original china')

plt.subplot(4,2,2)
plt.imshow(flower)
plt.title('The original flower')

plt.subplot(4,2,3)
plt.imshow(output[0,:,:,0], cmap='gray')
plt.title('R')

plt.subplot(4,2,4)
plt.imshow(output[1,:,:,0], cmap='gray')
plt.title('R')

plt.subplot(4,2,5)
plt.imshow(output[0,:,:,1], cmap='gray')
plt.title('G')

plt.subplot(4,2,6)
plt.imshow(output[1,:,:,1], cmap='gray')
plt.title('G')

plt.subplot(4,2,7)
plt.imshow(output[0,:,:,2], cmap='gray')
plt.title('B')

plt.subplot(4,2,8)
plt.imshow(output[1,:,:,2], cmap='gray')
plt.title('B')

plt.show()

