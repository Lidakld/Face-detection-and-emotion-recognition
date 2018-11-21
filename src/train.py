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
from fer2013_data import get_data
import matplotlib.pyplot as plt

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


def new_model(data):
    keep_prob = 1 #tf.placeholder(tf.float32, name='KEEP')
    data = tf.image.resize_images(data, tf.convert_to_tensor([42, 42]))
    print("Data: %s\n" % str(data))
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
    return logits, keep_prob


# Our application logic will be added here
def cnn_model_fn(images):
    """Model function for face detection"""
    # Input Layer
    input_layer = images

    # Convolutional Layer #1 (44, 44)
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        activation=tf.nn.relu)

    # Relu Layer
    bias1 = tf.Variable(tf.zeros([64]))
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))

    # Pooling Layer #1  (22, 22)
    pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=[3, 3], strides=2, padding='same')

    # Convolutional Layer and Pooling Layer #2 (18, 18)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=1, padding='same')

    # Convolutional Layer and Pooling Layer #3 (15, 15)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[4, 4],
        padding='valid',
        activation=tf.nn.relu)

    # Dense Layer (15 * 15)
    conv3_flat = tf.reshape(conv3, [-1, conv3.shape[1] * conv3.shape[2] * conv3.shape[3]])

    dense = tf.layers.dense(inputs=conv3_flat, units=3072, activation=tf.nn.softmax)

    # Logits Layer
    logits = tf.layers.dense(inputs=dense, units=7)

    return logits, None


def train_op(model, images, labels):
    """
    get operation of model

    :param label: gt
    :param model:  logits of final layers
    :return:
    """
    logits, keep_prob = model(images)

    # one_hot_labels =  tf.one_hot(tf.squeeze(labels), depth=7)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels))
    #
    # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='ACCURACY')

    loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)))
    accuracy = tf.contrib.metrics.accuracy(
        labels=labels, predictions=tf.argmax(input=logits, axis=1))

    metrics_op = {
        "accuracy": accuracy,
        "loss": loss
    }

    input_tensor = {
        'image': images,
        'label': labels
    }

    """summary"""
    train_loss_summary = tf.summary.scalar('t_loss', loss)
    train_metrics_summary = tf.summary.scalar('t_metrics', metrics_op['accuracy'])
    val_metrics_summary = tf.summary.scalar('v_metrics', metrics_op['accuracy'])

    summary_op = {
        'train': tf.summary.merge([train_loss_summary, train_metrics_summary]),
        'test': tf.summary.merge([val_metrics_summary])
    }
    return loss, metrics_op, input_tensor, summary_op


def augement_data(tfrecords_filenames, times=10):
    file_names = []
    for i in range(times):
        file_names.append(tfrecords_filenames)

    return file_names


def save_model(epoch_iteration, angle_range, sess):
    saver = tf.train.Saver()
    saver_path = saver.save(sess, os.path.join(FLAGS.save_path,
                                               "mode_ite_%d_angel_%d.ckpt" % (
                                                   epoch_iteration, angle_range)))
    print("[Save model] %s" % saver_path)


EPOCHS = 50
TRAIN_SIZE = 4 * (35887 * 2 - 10000)


def train(epoch_iteration=400, batch_size=8, angle_range=15):
    train_data_path = glob.glob(os.path.join(
        os.path.abspath(FLAGS.tf_data),
        "train" + "*"))

    test_data_path = glob.glob(os.path.join(
        os.path.abspath(FLAGS.tf_data),
        "validation" + "*"))

    [train_image, train_label], train_iter = get_data(train_data_path, batch_size=batch_size,
                                                      epoches_num=epoch_iteration,
                                                      max_angle=angle_range)

    [test_image, test_label], test_iter = get_data(test_data_path, batch_size=batch_size, epoches_num=epoch_iteration,
                                                   max_angle=angle_range)

    loss, metrics, input_tensor, summary_op = train_op(new_model, train_image, train_label)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 300, 0.99, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss, global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(train_iter.initializer)
        sess.run(test_iter.initializer)

        summ_writer = tf.summary.FileWriter(FLAGS.save_path, sess.graph)

        tf.logging.info("\tstart epoch")
        step = 0
        for epoch in range(epoch_iteration):
            for batch_id in range(int(TRAIN_SIZE / batch_size)):
                step += 1
                """train"""
                # get data
                train_image_v, train_label_v = sess.run([train_image, train_label])

                input_feed_train = {input_tensor['image']: train_image_v,
                                    input_tensor['label']: train_label_v}

                # backpropagation
                _, t_summary = sess.run([optimizer, summary_op['train']], feed_dict=input_feed_train)
                summ_writer.add_summary(t_summary, epoch)

                """test errors"""
                test_image_v, test_label_v = sess.run([test_image, test_label])
                input_feed_test = {input_tensor['image']: test_image_v,
                                   input_tensor['label']: test_label_v}

                test_summary = sess.run(summary_op['test'], feed_dict=input_feed_test)
                summ_writer.add_summary(test_summary, epoch)

                """print"""
                if step % 128 == 0:
                    # update metrics and loss
                    loss_v, tran_acc = sess.run([loss, metrics['accuracy']], feed_dict=input_feed_train)
                    test_acc = sess.run(metrics['accuracy'], feed_dict=input_feed_test)

                    print("In epoch %d, [step] %d,  loss: %.3f, Train: %.3f, Test: %.3f" % (
                        epoch, step, loss_v, tran_acc, test_acc))

            """save mode"""
            if epoch % 1 == 0:
                save_model(epoch, angle_range, sess)

        save_model(epoch, angle_range, sess)



if __name__ == "__main__":
    train(epoch_iteration=50,
          batch_size=64,
          angle_range=10)
