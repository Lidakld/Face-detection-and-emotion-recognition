#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 08:14:42 2018

@author: lida kuang
"""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd
#%%
IMAGE_SIZE = 48
EMO_NUM = 7
NUM_CHANNEL = 1
CLIPED_SIZE = 48
def GetSymmetric(pixel, size):
    '''
    pixel: np array with shape (count,size,size,1)
    '''
    count = pixel.shape[0]
    sym = np.zeros((count, size, size, NUM_CHANNEL))
    for i in range(count):
        for j in range(size):
            for k in range(size):
                sym[i,j,k,0] = pixel[i,j,size-k-1,0]
    return sym

def GetInput():
    path = '../data/fer2013.csv'
    all_data = pd.read_csv(path)
    label = np.array(all_data['emotion'])
    data = np.array(all_data['pixels'])
    sample_count = len(label)  # should be 35887

    pixel_data = np.zeros((sample_count, IMAGE_SIZE * IMAGE_SIZE))# 像素点数据
    label_data = np.zeros((sample_count, EMO_NUM), dtype = int)# 标签数据，独热
    for i in range(sample_count):
        x = np.fromstring(data[i], sep = ' ')
        max = x.max()
        x = x / (max + 0.001)  # 灰度归一化
        pixel_data[i] = x
        label_data[i, label[i]] = 1
    pixel_data = pixel_data.reshape(sample_count, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL)
    x_test = pixel_data[30000:35000]
    y_test = label_data[30000:35000]

    x_train = np.concatenate((pixel_data[0:30000],pixel_data[35000:]), axis = 0)
    symmetric_x_train = GetSymmetric(x_train, IMAGE_SIZE)
    x_train = np.concatenate((x_train, symmetric_x_train), axis = 0)
    y_train = np.concatenate((label_data[0:30000],label_data[35000:],label_data[0:30000],label_data[35000:]))
    return (x_train, y_train, x_test, y_test)

def GetClipedImage(pixel, start):
    '''
    pixel: raw 48*48 pixel data with shape (count, 48, 48, 1)
    start: a tuple such as (0,0),(2,3),(4,2), represents start point of clipped 42*42 image
    '''
    count = pixel.shape[0]
    out = np.zeros((count, CLIPED_SIZE, CLIPED_SIZE, NUM_CHANNEL))
    for i in range(count):
        for j in range(CLIPED_SIZE):
            out[i,j,:,0] = pixel[i,start[0]+j,start[1]:start[1]+CLIPED_SIZE,0]
    return out

def DataPreprocess(pixel, label = []):
    '''
    pixel: pixel data with shape (count,48,48,1)
    label: optical, corresponding label of pixel
    '''
    a = np.random.randint(0,2)
    b = np.random.randint(3,5)
    c = np.random.randint(0,2)
    d = np.random.randint(3,5)
    pixel1 = GetClipedImage(pixel, (a,c))
    pixel2 = GetClipedImage(pixel, (a,d))
    pixel3 = GetClipedImage(pixel, (b,c))
    pixel4 = GetClipedImage(pixel, (b,d))
    out_p = np.concatenate((pixel1, pixel2, pixel3, pixel4), axis = 0)
    if len(label) == 0:
        return out_p
    else:
        out_l = np.concatenate((label, label, label, label), axis = 0)
        return (out_p, out_l)
    
#%%
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  print("\ninput shape: "+ str(input_layer.shape))
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters = 64,
      kernel_size=[5, 5],
      activation=tf.nn.relu)
  print("\nconv1 shape: "+ str(conv1.shape))
# Relu Layer
  bias1 = tf.Variable(tf.zeros([64]))
  relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
#     pool1 = tf.nn.max_pool(relu1, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
  print("\nrelu1 shape: "+ str(relu1.shape))

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs = relu1, pool_size=[3, 3], strides=2, padding = 'same')
  print("\npool1 shape: "+ str(pool1.shape))
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      activation=tf.nn.relu)
  print("\nconv2 shape: "+ str(conv2.shape))
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=1, padding = 'same')
  print("pool2 shape: "+ str(pool2.shape))
# Convolutional Layer #2 and Pooling Layer #2
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[4, 4],
      padding = 'valid',
      activation=tf.nn.relu)
  print("\nconv3 shape: "+ str(conv3.shape))
  # Dense Layer
  conv3_flat = tf.reshape(conv3, [-1, conv3.shape[1] * conv3.shape[2] * conv3.shape[3]])
  print("\nconv3_flat shape: "+ str(conv3_flat.shape))
  
  dense = tf.layers.dense(inputs=conv3_flat, units=3072, activation=tf.nn.softmax)
  print("\ndense shape: " + str(dense.shape))
  # Logits Layer
  logits = tf.layers.dense(inputs=dense, units=10)
  print("\nlogist shape: " + str(logits.shape))

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  
def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
  model_fn=cnn_model_fn)
  
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
  
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=5,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

#%%
# Imports
# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()

    
