import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
import numpy as np
import math
import cv2
import sys
import os

IMG_PX_SIZE = 50
HM_SLICES = 20  # quantidade de fatias para todas as imagens .nii
x = tf.placeholder('float')
y = tf.placeholder('float')

n_classes = 3
keep_rate = 0.8

path = sys.argv[1]


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]


def mean(l):
    return sum(l)/len(l)


def process_data():
    img = nib.load(path + '/' + os.listdir(path)[0])
    slices = img.get_fdata()

    new_slices = []

    slices = [cv2.resize(np.array(each_slice), (IMG_PX_SIZE, IMG_PX_SIZE))
              for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / HM_SLICES)

    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == HM_SLICES - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES + 2:
        new_val = list(
            map(mean, zip(*new_slices[HM_SLICES-1], new_slices[HM_SLICES])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val

    if len(new_slices) == HM_SLICES + 1:
        new_val = list(
            map(mean, zip(*new_slices[HM_SLICES-1], new_slices[HM_SLICES])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val

    return np.array(new_slices)


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                               tamanho da janela      movimento da janela
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),  # Convolução 3x3x3 com 1 entrada e 32 saídas
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, IMG_PX_SIZE, IMG_PX_SIZE, HM_SLICES, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

X_new = process_data()

pred = convolutional_neural_network(x)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('modelo.meta')
    saver.restore(sess, 'modelo')
    print('Model loaded')

    sess.run(tf.initialize_all_variables())
    c = sess.run(pred, feed_dict={x: X_new})

    print(c)
    print(np.argmax(c[0]))