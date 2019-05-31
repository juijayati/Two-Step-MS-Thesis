import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rn
import math


np.random.seed(42)
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

data_source_folder = 'DataSplit_700_50'

train_x = np.load(os.path.join(data_source_folder, 'tr_features.npy'))

train_x = np.delete(train_x, np.s_[0:193:1], 1)
#plt.scatter(train_x[:,15], train_x[0:,10]);
test_x = np.load(os.path.join(data_source_folder, 'ts_features.npy'))
test_x = np.delete(test_x, np.s_[0:193:1], 1)

train_y = np.load(os.path.join(data_source_folder, 'tr_super_labels.npy'))
test_y = np.load(os.path.join(data_source_folder, 'ts_super_labels.npy'))

#train_x = np.concatenate((train_x,test_x),axis=0)
#train_y = np.concatenate((train_y, test_y), axis = 0)

#train_x.tail()
training_epochs = 1
n_dim = train_x.shape[1]
n_classes = train_y.shape[1]
n_hidden_units_one = math.ceil(1.6 * n_dim)
n_hidden_units_two = math.ceil(1.4 * n_dim)
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev= (1 / np.sqrt(n_hidden_units_one))))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=(1 / np.sqrt(n_hidden_units_two))))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, cost = sess.run([optimizer, cost_function], feed_dict={X: train_x, Y: train_y})
        cost_history = np.append(cost_history, cost)

    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y, 1))
    print('Test accuracy: ', round(sess.run(accuracy, feed_dict={X: test_x, Y: test_y}), 3))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(cost_history)
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    plt.show()



