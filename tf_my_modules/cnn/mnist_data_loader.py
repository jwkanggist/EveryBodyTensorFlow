#-*- coding: utf-8 -*-
#! /usr/bin/env python
'''
    filename: mnist_data_loader.py

    decription:
    The MNIST database of handwritten digits, available from the LeCun's webpage,
    has a training set of 60,000 examples, and a test set of 10,000 examples.
    The digits have been size-normalized and centered in a fixed-size image, 28X28.

    author: Jaewook Kang
    date : 2018 Mar
'''

import os
import gzip
from os import getcwd
import numpy as np
import tensorflow as tf
from six.moves import urllib



# data filename =====================================================
'''
    we splits the LeCun's training data to training and validation data.
    we use the LeCun's test data as it is.
'''

class DataFilename(object):

    def __init__(self):
        self.trainingimages_filename = 'train-images-idx3-ubyte.gz'
        self.traininglabels_filename = 'train-labels-idx1-ubyte.gz'
        self.testimages_filename = 't10k-images-idx3-ubyte.gz'
        self.testlabels_filename = 't10k-labels-idx1-ubyte.gz'


class MnistLoader(object):

    def __init__(self):
        self.SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        self.WORK_DIRECTORY = getcwd() + '/data/mnist'
        # The size of mnist image
        # The image size of the first convolutional layer in LeNet5 is 28 X 28
        self.IMAGE_SIZE = 28
        # gray scsle image
        self.NUM_CHANNELS = 1
        self.PIXEL_DEPTH = 255
        # 0 to 9 char images
        self.NUM_LABELS = 10

    # function module for data set loading  ============================
    def download_mnist_dataset(self,filename):
        '''
            check whether we have the mnist dataset in the given WORK_DIRECTORY,
            otherwise, download the data from YANN's website,
        '''

        if not tf.gfile.Exists(self.WORK_DIRECTORY):
            tf.gfile.MakeDirs(self.WORK_DIRECTORY)
            print (" %s is not exist" % self.WORK_DIRECTORY)

        filepath = os.path.join(self.WORK_DIRECTORY,filename)

        print('filepath = %s' % filepath)

        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.SOURCE_URL+ filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
                print ('Successfully downloaded',filename,size,'bytes.')

            print('[download_mnist_dataset] filepath = %s' % filepath)
        return filepath




    def extract_data(self,filename, num_images):
        '''
        Extract the image into 4D tensor [image index, height,weight, channels]
        values are rescaled from [ 0, 255] down to [-0.5, 0.5]

        LeCun provides training set in a type of BMP format (No compression)
        One pixel value is from 0 to 255

        For representation, this needs 1byte = 8 bits ==> 2^8 =256

        TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
        '''
        print ('[extract_data] Extracting gzipped data from %s' % filename)

        with gzip.open(filename) as bytestream:
            # threw out the header which has 16 bytes
            bytestream.read(16)

            # extract image data
            buf     = bytestream.read(self.IMAGE_SIZE * self.IMAGE_SIZE * num_images * self.NUM_CHANNELS)

            # type cast from uint8 to np.float32 to work in tensorflow framework
            data    = np.frombuffer(buffer=buf,
                                    dtype =np.uint8).astype(np.float32)

            # rescaling data set over [-0.5 0.5]
            data    = (data - (self.PIXEL_DEPTH / 2.0) ) / self.PIXEL_DEPTH

            # reshaping to 4D tensors
            data    = data.reshape(num_images, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS)
            return data




    def extract_label(self,filename, num_images):
        '''
            Extract the lable into vector of int64 label IDs
        '''
        print ('[extract_label] Extracting gzipped data from %s' % filename)

        with gzip.open(filename=filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            # type cast from uint8 to np.int64 to work in tensorflow framework
            labels = np.frombuffer(buffer=buf,
                                   dtype=np.uint8).astype(np.int64)
        print('[extract_label] label= %s'% labels)
        return labels
