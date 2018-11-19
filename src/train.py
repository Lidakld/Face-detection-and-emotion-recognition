#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 08:14:42 2018

@author: lida kuang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import src.fer2013_data as dataset_fer

tf.app.flags.DEFINE_string('tf_data',
                           os.path.abspath('../data/fer2013/tfrecords'),
                           'Folder containing images.')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SIZE = 48


# Our application logic will be added here
def cnn_model_fn(images):
    """Model function for face detection"""
    images_t = images

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

    return logits, images_t


def train_op(model, labels):
    """
    get operation of model

    :param label: gt
    :param model:  logits of final layers
    :return:
    """
    logits, images_t = model

    model = tf.reshape(logits, [-1, 7])
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    input_tensor = {
        'image': images_t,
        'label': labels
    }
    return loss, eval_metric_ops, input_tensor


def train(epoch_iteration=10, batch_size = 30, angle_range = 15):
    tfrecords_filenames = os.path.join(FLAGS.tf_data, 'train-00000.tfrecord')
    images, labels = dataset_fer.read_and_decode(tfrecords_filenames, epoch_iteration,
                   batch_size=batch_size,
                   num_threads=8,
                   max_angle=angle_range)


    model = cnn_model_fn(images)
    loss, accuracy, input_tensor = train_op(model, labels)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 300, 0.99, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for epoch in range(epoch_iteration):
                image, label = sess.run([images, labels])
                _, loss, acc = sess.run([optimizer, loss, accuracy], feed_dict={input_tensor['image']:image,
                                                                                input_tensor['label']:label})
                if step % 1 == 0:
                    acc_v = 0
                    print("In epoch %d, loss: %.3f, accuracy: %.3f, validation accuracy: %.3f" % (
                    epoch, loss, acc, acc_v))
        saver = tf.train.Saver()
        saver_path = saver.save(sess, SAVE_PATH)
        print('Finished!')


if __name__ == "__main__":
    train()
