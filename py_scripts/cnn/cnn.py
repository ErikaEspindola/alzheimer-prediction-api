import tensorflow as tf
from tensorflow import keras
import numpy as np

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 3

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                               tamanho da janela      movimento da janela
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])), # Convolução 3x3x3 com 1 entrada e 32 saídas
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])), # Convolução 3x3x3 com 32 entrada e 64 saídas
               'W_fc': tf.Variable(tf.random_normal([54080, 1024])), # 1024 nós
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    # Primeira camada de convolução
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    # Segunda camada de convolução
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    # Camada totalmente conectada
    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    # Camada de saída
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    much_data = np.load('muchdata-50-50-20.npy', allow_pickle=True)
    train_data = much_data[0:10]
    validation_data = much_data[10:20]
    prediction = convolutional_neural_network(x)
    print(prediction)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 1
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                except Exception as e:
                    pass

            print('Epoch', epoch + 1, 'completed out of',
                  hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        saver = tf.train.Saver()

        saver.save(sess, '/home/erika/modelo')
        print('Accuracy:', accuracy.eval(
            {x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))


train_neural_network(x)
