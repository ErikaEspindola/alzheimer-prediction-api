import tensorflow.compat.v1 as tf
import numpy as np
import math
from google.colab import drive

drive.mount('/content/drive/')
  
tf.disable_eager_execution()

IMG_SIZE_PX = 50
SLICE_COUNT = 30

n_classes = 3

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def calc():
    img_px = math.ceil(IMG_SIZE_PX/4)
    slice_ct = math.ceil(SLICE_COUNT/4)

    return img_px * img_px * slice_ct * 64


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                               tamanho da janela      movimento da janela
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    number = calc()

    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])), # Convolução 3x3x3 com 1 entrada e 32 saídas
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])), # Convolução 3x3x3 com 32 entrada e 64 saídas
               'W_conv3': tf.Variable(tf.random_normal([3, 3, 3, 64, 128])),
               'W_conv4': tf.Variable(tf.random_normal([3, 3, 3, 128, 256])),
               'W_conv5': tf.Variable(tf.random_normal([3, 3, 3, 256, 512])),
               'W_fc': tf.Variable(tf.random_normal([number, 1024])), # 1024 nós
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_conv3': tf.Variable(tf.random_normal([128])),
              'b_conv4': tf.Variable(tf.random_normal([256])),
              'b_conv5': tf.Variable(tf.random_normal([512])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    # Primeira camada de convolução
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    # Segunda camada de convolução
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    # Terceira camada de convolução
    conv3 = tf.nn.relu(conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool3d(conv3)

    # Quarta camada de convolução
    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool3d(conv4)

    # Quinta camada de convolução
    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv5']) + biases['b_conv5'])
    conv5 = maxpool3d(conv5)

    # Camada totalmente conectada
    fc = tf.reshape(conv2, [-1, number])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    # Camada de saída
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    much_data = np.load("/content/drive/My Drive/TCC/dataset-50-50-30-pre.npy", allow_pickle=True)
    train_data = much_data[400:]
    validation_data = much_data[:399]

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 25
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                X = data[0] # imagem
                Y = data[1] # label
                _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                epoch_loss += c

            print('Epoch', epoch + 1, '/', hm_epochs, '. Loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval(
            {x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

def gpu():
  with tf.device('/device:GPU:0'):
    train_neural_network(x)

gpu()