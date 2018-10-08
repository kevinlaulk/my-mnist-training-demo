# masaaki 2018-10-4

import tensorflow as tf
import numpy as np
import mnistdata
import shutil
import math
import os

print('tensorflow version : ', tf.__version__)
tf.set_random_seed(0)
logdir = 'log'
if os.path.exists(logdir):
    shutil.rmtree(logdir)
mnist = mnistdata.read_data_sets('data', one_hot=True, reshape=False)

print('~~~~design graph~~~~')

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
    Y_ = tf.placeholder(tf.float32, [None, 10], name='Y_')
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)

with tf.name_scope('weights'):
    w1 = tf.Variable(tf.truncated_normal([6,6,1,6],stddev=0.1), name='w1')
    b1 = tf.Variable(tf.constant(0.1, tf.float32, [6]), name='b1')
    w2 = tf.Variable(tf.truncated_normal([5,5,6,12],stddev=0.1), name='w2')
    b2 = tf.Variable(tf.constant(0.1, tf.float32, [12]), name='b2')
    w3 = tf.Variable(tf.truncated_normal([4,4,12,24],stddev=0.1), name='w3')
    b3 = tf.Variable(tf.constant(0.1, tf.float32, [24]), name='b3')
    w4 = tf.Variable(tf.truncated_normal([7*7*24,200],stddev=0.1), name='w4')
    b4 = tf.Variable(tf.constant(0.1, tf.float32, [200]), name='b4')
    w5 = tf.Variable(tf.truncated_normal([200,10],stddev=0.1), name='w5')
    b5 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name='b5')

    tf.summary.histogram('w1', w1)
    tf.summary.histogram('b1', b1)
    tf.summary.histogram('w2', w2)
    tf.summary.histogram('b2', b2)
    tf.summary.histogram('w3', w3)
    tf.summary.histogram('b3', b3)
    tf.summary.histogram('w4', w4)
    tf.summary.histogram('b4', b4)
    tf.summary.histogram('w5', w5)
    tf.summary.histogram('b5', b5)

with tf.name_scope('inference'):
    Y1 = tf.nn.relu(tf.nn.conv2d(X, w1, [1, 1, 1, 1], padding='SAME') + b1)
    # Y1d = tf.nn.dropout(Y1, pkeep)
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, w2, [1, 2, 2, 1], padding='SAME') + b2)
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, w3, [1, 2, 2, 1], padding='SAME') + b3)
    Y3r = tf.reshape(Y3, shape = [-1, 7*7*24])
    Y4 = tf.nn.relu(tf.matmul(Y3r, w4) + b4)
    Y4d = tf.nn.dropout(Y4, pkeep)
    Ylogist = tf.matmul(Y4d, w5) + b5
    Y = tf.nn.softmax(Ylogist)


    tf.summary.image('Y1',tf.reshape(Y1, [-1, 28,28,1]))
    tf.summary.image('Y2',tf.reshape(Y2, [-1, 14,14,1]))
    tf.summary.image('Y3',tf.reshape(Y3, [-1, 7,7,1]))



with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels= Y_, logits=Ylogist)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    correct_prediction = tf.equal(tf.arg_max(Y,1), tf.arg_max(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # 0.0001 + 0.003 * (1/e)^(step/2000))
    lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('lr', lr)

print('~~~init~~~')
init = tf.global_variables_initializer()
merge = tf.summary.merge_all()
saver = tf.train.Saver()

print('~~~training~~~')
test_log = np.empty(shape=[1], dtype=float)
with tf.Session() as sess:
    sess.run(init)
    train_summary = tf.summary.FileWriter(logdir+'/train', graph=sess.graph)
    test_summary = tf.summary.FileWriter(logdir+'/test', graph=sess.graph)

    for i in range(10000+1):
        batch_x, batch_y = mnist.train.next_batch(100)

        if i%10==0:
            a, c, summary = sess.run([accuracy, cross_entropy, merge], feed_dict={X: batch_x, Y_: batch_y, step: i, pkeep: 1.0})
            print(i, ': accuracy: ', a, ' loss: ', c)
            train_summary.add_summary(summary, global_step=i)
        if i%50==0:
            a, c, summary = sess.run([accuracy, cross_entropy, merge], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, step: i, pkeep:1.0})
            print('*********************epoch: ', i*100//mnist.train.images.shape[0]+1, 'accuracy: ', a, ' loss: ', c)
            saver.save(sess, 'model/model.ckpt', i)
            test_summary.add_summary(summary, global_step=i)
            test_log = np.append(test_log, a)
        sess.run(train_step, feed_dict={X: batch_x, Y_: batch_y, step: i, pkeep: 0.75} )

train_summary.close()
test_summary.close()
print('test accuracy max: ', test_log.max())