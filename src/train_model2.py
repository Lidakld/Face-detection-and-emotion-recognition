#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 08:14:42 2018

@author: lida kuang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from fer2013_data import get_data

import traceback
import glob

tf.app.flags.DEFINE_string('tf_data',
                           os.path.abspath('../data/fer2013/tfrecords'),
                           'Folder containing images.')

tf.app.flags.DEFINE_string('save_path',
                           os.path.abspath('../data/fer2013/train'),
                           'Folder containing model results')

tf.app.flags.DEFINE_string('inference_path',
                           os.path.abspath('../data/fer2013/inference'),
                           'Folder containing ifnerence results')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SIZE = 48
NUM_CHANNEL = 1

IMAGE_SIZE = 48
CLIPED_SIZE = 42
EMO_NUM = 7
TRAIN_SIZE = 4 * (35887 * 2 - 10000)
VALID_SIZE = 1500
TEST_SIZE = 5000
BATCH_SIZE = 50
NUM_CHANNEL = 1
EPOCHS = 50
SAVE_PATH = './saved_model'
emo_dict = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Suprise', 6: 'Neutral'
}


def model(data, keep_prob):
    # first layer IN: 42*42*1  OUT: 20*20*32
    kernel1 = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNEL, 32], stddev=5e-2))
    conv1 = tf.nn.conv2d(data, kernel1, [1, 1, 1, 1], padding='SAME')
    bias1 = tf.Variable(tf.zeros([32]))
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # second layer IN: 20*20*32  OUT: 10*10*32
    kernel2 = tf.Variable(tf.truncated_normal([4, 4, 32, 32], stddev=5e-2))
    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
    bias2 = tf.Variable(tf.zeros([32]))
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # third layer IN: 10*10*32  OUT: 5*5*64
    kernel3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=5e-2))
    conv3 = tf.nn.conv2d(pool2, kernel3, [1, 1, 1, 1], padding='SAME')
    bias3 = tf.Variable(tf.zeros([64]))
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, bias3))
    pool3 = tf.nn.max_pool(relu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully connected layers
    fc1_data = tf.reshape(pool3, shape=[-1, 5 * 5 * 64])
    fc1 = tf.contrib.layers.fully_connected(fc1_data, 1024, activation_fn=tf.nn.relu)
    fc1_out = tf.nn.dropout(fc1, keep_prob)

    fc2 = tf.contrib.layers.fully_connected(fc1_out, 512, activation_fn=tf.nn.relu)
    fc2_out = tf.nn.dropout(fc2, keep_prob)

    logits = tf.contrib.layers.fully_connected(fc2_out, 7, activation_fn=None)
    logits = tf.identity(logits, name='LOGITS')
    return logits


def train(batch_size=BATCH_SIZE):
    train_data_path = glob.glob(os.path.join(
        os.path.abspath(FLAGS.tf_data),
        "train" + "*"))

    test_data_path = glob.glob(os.path.join(
        os.path.abspath(FLAGS.tf_data),
        "validation" + "*"))

    [x_train, y_train], train_iter = get_data(train_data_path, batch_size=batch_size,
                                              max_angle=5)
    y_train = tf.one_hot(tf.squeeze(y_train), depth=7)
    [x_test, y_test], test_iter = get_data(test_data_path, batch_size=batch_size,
                                           max_angle=5)
    y_test = tf.one_hot(tf.squeeze(y_test), depth=7)

    keep_prob = tf.placeholder(tf.float32, name='KEEP')
    y_pred = model(x_train, keep_prob)
    # y_pred = mmodel(x_data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_train))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 300, 0.99, staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='ACCURACY')

    summary_op = tf.summary.merge([tf.summary.scalar('loss', cost),
                                   tf.summary.scalar('accuracy', accuracy)])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter.initializer)
        sess.run(test_iter.initializer)

        train_summary_writer = tf.summary.FileWriter(FLAGS.save_path, sess.graph)

        step = 0
        for epoch in range(EPOCHS):
            for batch_i in range(TRAIN_SIZE // BATCH_SIZE):
                step += 1

                x_feats, y_feats = sess.run([x_train, y_train])

                feed = {x_train: x_feats, y_train: y_feats, keep_prob: 0.6}
                sess.run(optimizer, feed_dict=feed)
                if step % 128 == 0:
                    (loss, acc, summary, y_pred_value) = sess.run([cost, accuracy, summary_op, y_pred], feed_dict=feed)
                    train_summary_writer.add_summary(summary, step)

                    x_val, y_val = sess.run([x_test, y_test])
                    feed_v = {x_train: x_val, y_train: y_val, keep_prob: 1.0}
                    acc_v = sess.run(accuracy, feed_dict=feed_v)
                    print("In epoch %d, batch %d, loss: %.3f, accuracy: %.3f, validation accuracy: %.3f" % (
                        epoch, batch_i, loss, acc, acc_v))
        saver = tf.train.Saver()
        saver.save(sess, FLAGS.save_path)
        print('Finished!')


def test():
    test_data_path = glob.glob(os.path.join(
        os.path.abspath(FLAGS.tf_data),
        "validation" + "*"))

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        [x_test, y_test], test_iter = get_data(test_data_path, batch_size=50,
                                               max_angle=5)
        # load the model
        loader = tf.train.import_meta_graph(SAVE_PATH + '.meta')
        loader.restore(sess, SAVE_PATH)
        load_x = loaded_graph.get_tensor_by_name('INPUT:0')
        load_y = loaded_graph.get_tensor_by_name('LABEL:0')
        load_acc = loaded_graph.get_tensor_by_name('ACCURACY:0')
        load_log = loaded_graph.get_tensor_by_name('LOGITS:0')
        load_keep = loaded_graph.get_tensor_by_name('KEEP:0')
        # record accuracy
        total_batch_acc = 0
        batch_count = TEST_SIZE // BATCH_SIZE
        for batch_i in range(batch_count):
            log = np.zeros((BATCH_SIZE, EMO_NUM))
            y_feats = y_test[batch_i * BATCH_SIZE: (batch_i + 1) * BATCH_SIZE]
            for k in range(4):
                x_feats, y_feats = sess.run([x_test, y_test])
                log1 = sess.run(load_log, feed_dict={
                    load_x: x_feats, load_y: y_feats, load_keep: 1.0
                })

                log += log1
            emos = sess.run(tf.argmax(log, 1))
            correct_emos = sess.run(tf.argmax(y_feats, 1))
            tmp = emos == correct_emos
            acc = tmp.sum() / tmp.shape[0]
            total_batch_acc += acc
            print('In test batch %d: the accuracy is %.3f' % (batch_i, acc))
        print('Total accuracy in test set is %.3f' % (total_batch_acc / batch_count))


if __name__ == '__main__':
    train()
    test()
