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
import fer2013_data as dataset_fer

tf.app.flags.DEFINE_string('tf_data',
                           os.path.abspath('../data/fer2013/tfrecords'),
                           'Folder containing images.')

tf.app.flags.DEFINE_string('save_path',
                           os.path.abspath('../data/fer2013/train'),
                           'Folder containing model results')


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

def augement_data(tfrecords_filenames, times = 10):
    file_names = []
    for i in range(times):
        file_names.append(tfrecords_filenames)

    return file_names

def train(epoch_iteration=100, batch_size = 8, angle_range = 15, time=100):
    train_data_path = os.path.join(FLAGS.tf_data, 'train-00000.tfrecord')
    val_data_path = os.path.join(FLAGS.tf_data, 'validation-00000.tfrecord')
    tf.logging.info("\t[train_data] in %s\n[val_data] in %s" % (train_data_path, val_data_path))

    train_data_files = augement_data(train_data_path, times=time)
    val_data_files = augement_data(val_data_path, times=time)

    train_images, train_labels = dataset_fer.read_and_decode(train_data_files, 1,
                   batch_size=batch_size,
                   num_threads=8,
                   max_angle=angle_range)

    val_images, val_labels = dataset_fer.read_and_decode(val_data_files, 1,
                                                             batch_size=batch_size,
                                                             num_threads=8,
                                                             max_angle=angle_range)

    model = cnn_model_fn(train_images)
    loss, metrics, input_tensor = train_op(model, train_labels)

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 300, 0.99, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    tf.logging.info("\tStart Triaining")
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tf.logging.info("\tstart epoch")
        for epoch in range(epoch_iteration):
                try:
                    train_image, train_label = sess.run([train_images, train_labels])

                    _, loss_v, acc = sess.run([optimizer, loss, metrics['accuracy']],
                                          feed_dict={input_tensor['image']:train_image,
                                                    input_tensor['label']:train_label})

                    val_image, val_label = sess.run([ val_images, val_labels ])
                    val_acc = sess.run(metrics['accuracy'],
                                   feed_dict={input_tensor['image']: val_image,
                                              input_tensor['label']: val_label})
                except:
                    tf.logging.warning("Traing data exthuastic")
                    break;

                if epoch % 100 == 0:
                    print("In epoch %d, loss: %.3f, accuracy: %.3f, validation accuracy: %.3f" % (
                    epoch, loss_v, acc[0], val_acc[0]))

        saver = tf.train.Saver()
        saver_path = saver.save(sess, os.path.join(FLAGS.save_path, "mode_ite_%d_angel_%d.ckpt" % (epoch_iteration, angle_range)))
        print('Finished! \t save files in %s' % saver_path)
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()


if __name__ == "__main__":
    train(epoch_iteration=10000,
            batch_size=64,
            angle_range=45,
            time=500)
