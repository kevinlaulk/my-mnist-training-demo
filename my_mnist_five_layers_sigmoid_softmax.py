# my own demo for base learning
# 2018-09-30

import tensorflow as tf
import numpy as np
import mnistdata
import os
import shutil
import math

print('tensorflow version: ', tf.__version__)
tf.set_random_seed(0)
mnist = mnistdata.read_data_sets('data', one_hot=True, reshape=False)
logdir1 = 'log'
if os.path.exists(logdir1):
    shutil.rmtree(logdir1)

print('~~~~~~~~~ design graph~~~~~~~~~~')
with tf.name_scope('input'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='X')
    Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)

with tf.name_scope('inference'):
    XX = tf.reshape(X, [-1, 784], name='X_reshape')
    w1 = tf.Variable(initial_value=tf.truncated_normal([784,500], stddev=0.1), name='w1')
    b1 = tf.Variable(initial_value=tf.ones([500])/10, name='b1')
    w2 = tf.Variable(initial_value=tf.truncated_normal([500,100], stddev=0.1), name='w2')
    b2 = tf.Variable(initial_value=tf.ones([100])/10, name='b2')
    w3 = tf.Variable(initial_value=tf.truncated_normal([100,50], stddev=0.1), name='w3')
    b3 = tf.Variable(initial_value=tf.ones([50])/10, name='b3')
    w4 = tf.Variable(initial_value=tf.truncated_normal([50,20], stddev=0.1), name='w4')
    b4 = tf.Variable(initial_value=tf.ones([20])/10, name='b4')
    w5 = tf.Variable(initial_value=tf.truncated_normal([20,10], stddev=0.1), name='w5')
    b5 = tf.Variable(initial_value=tf.ones([10])/10, name='b5')

    tf.summary.histogram('w1', w1) # yan zhong ying xiang su du
    tf.summary.histogram('b1', b1)
    tf.summary.histogram('w2', w2)
    tf.summary.histogram('b2', b2)
    # w = tf.Variable(initial_value=tf.zeros([784, 10]), name='w')
    # b = tf.Variable(initial_value=tf.zeros([10]), name='b')

with tf.name_scope('output'):
    Y1 = tf.nn.relu(tf.matmul(XX,w1)+b1)
    Y1d = tf.nn.dropout(Y1, pkeep)

    Y2 = tf.nn.relu(tf.matmul(Y1d,w2)+b2)
    Y2d = tf.nn.dropout(Y2, pkeep)

    Y3 = tf.nn.relu(tf.matmul(Y2d,w3)+b3)
    Y3d = tf.nn.dropout(Y3, pkeep)

    Y4 = tf.nn.relu(tf.matmul(Y3d,w4)+b4)
    Ylogits = tf.matmul(Y4,w5)+b5
    Y = tf.nn.softmax(Ylogits)

    # Y = tf.nn.softmax(tf.matmul(XX, w) + b)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('optimization'):
    lr = 0.0001 + tf.train.exponential_decay(0.003,step,2000,1/math.e)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
    tf.summary.scalar('learning rate', lr)

print('~~~init variables~~~')
init = tf.global_variables_initializer()
merge = tf.summary.merge_all()
saver = tf.train.Saver()

print('~~~ start training ~~~')
with tf.Session() as sess:
    sess.run(init)
    summary_writer1 = tf.summary.FileWriter(logdir = logdir1+'/train', graph = sess.graph)
    summary_writer2 = tf.summary.FileWriter(logdir = logdir1+'/test', graph = sess.graph)

    for i in range(10000+1):
        batch_x, batch_y = mnist.train.next_batch(100)

        if i%10 == 0:
            a, b, summary = sess.run([accuracy, cross_entropy, merge], feed_dict={X: batch_x, Y_: batch_y, step: i, pkeep: 1.0})
            # a, b = sess.run([accuracy, cross_entropy], feed_dict={X: batch_x, Y_: batch_y})
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(b))
            summary_writer1.add_summary(summary, global_step=i)

        if i%50 == 0:
            a, b, summary = sess.run([accuracy, cross_entropy, merge], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, step: i, pkeep: 1.0})
            # a, b = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
            print('****************epoch:', str(i*100 // mnist.train.images.shape[0]+1), 'accuracy:', str(a), 'loss:', str(b))
            saver.save(sess, 'model/model.ckpt', i)
            summary_writer2.add_summary(summary, global_step=i)
        _, summary = sess.run([train_step, merge], feed_dict={X: batch_x, Y_: batch_y, step: i, pkeep: 0.75})
        # sess.run([train_step], feed_dict={X: batch_x, Y_: batch_y})

        # summary_writer1.add_summary(summary, global_step= i)
summary_writer1.close()
summary_writer2.close()







