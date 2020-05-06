import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
import numpy as np
import shutil
import math
import glob
import gzip
import cv2
import sys
import os

tf.compat.v1.disable_eager_execution()


def extract(path):
    name = ''
    for filename in glob.iglob(path, recursive=False):
        with gzip.open(filename, 'rb') as f_in:
            name = filename.split('.')[0] + '.nii'
            with open(name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return name


def calc():
    img_px = math.ceil(50/4)
    slice_ct = math.ceil(30/4)

    return img_px * img_px * slice_ct * 64


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]


def mean(l):
    return sum(l)/len(l)


def apply_histogram(img_n):
    img_n = np.array(img_n)
    newImg = []

    for i in img_n:
        img = cv2.normalize(src=i, dst=None, alpha=0, beta=80,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        equ = cv2.equalizeHist(img)
        newImg.append(equ.tolist())

    return newImg


def process_data(path, apply_hist=True):
    img = nib.load(path)
    slices = img.get_fdata()

    new_slices = []

    slices = [cv2.resize(np.array(each_slice), (50, 50)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / 100)

    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    new_slices = new_slices[30:60]

    if apply_hist:
        new_slices = apply_histogram(new_slices)

    return np.array(new_slices)


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    number = calc()

    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([number, 1024])),
               'out': tf.Variable(tf.random_normal([1024, 3]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([3]))}

    x = tf.reshape(x, shape=[-1, 50, 50, 30, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, number])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, 0.8)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def classification(path):
    x = tf.placeholder('float')

    new_path = extract(path + '.gz')
    X_new = process_data(path=new_path, apply_hist=True)

    print(X_new.shape)

    pred = convolutional_neural_network(x)

    res = 0

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('modelo.meta')
        saver.restore(sess, 'modelo')

        sess.run(tf.initialize_all_variables())

        probabilities = tf.nn.softmax(pred)

        c = sess.run(probabilities, feed_dict={x: X_new})

        res = np.argmax(c)

    return res

# print(classification('/tmp/A65COG_I31102.nii'))