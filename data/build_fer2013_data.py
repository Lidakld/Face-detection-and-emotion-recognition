"""Converts fer2013 images into tf-records

    fer2013
    ├── eval
    ├── images
    │   ├── train
    │   │   ├── angry
    │   │   ├── disgust
    │   │   ├── fear
    │   │   ├── happy
    │   │   ├── neutral
    │   │   ├── sad
    │   │   └── surprise
    │   └── validation
    │       ├── angry
    │       ├── disgust
    │       ├── fear
    │       ├── happy
    │       ├── neutral
    │       ├── sad
    │       └── surprise
    ├── inference
    ├── tfrecords
    └── train

Image folder:
  ./fer2013/images

Semantic segmentation annotations:
  ./fer2013/images

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:
"""
# %%
import numpy as np
import os.path
import sys
import tensorflow as tf
import cv2 as cv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           os.path.abspath('./fer2013/images'),
                           'Folder containing images.')
tf.app.flags.DEFINE_string(
    'output_dir',
    os.path.abspath('./fer2013/tfrecords'),
    'Path to save converted tfrecords.')

_NUM_SHARDS = 4


def load_image(addr, shape=(48, 48)):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv.imread(addr, cv.IMREAD_GRAYSCALE)
    # img = cv.resize(img, shape, interpolation=cv.INTER_CUBIC)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_dataset(dataset_split):
    """
    Convert split data into tfrecord
    :param dataset_split:
    :return:
    """

    dic = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }

    dataset = os.path.join(FLAGS.image_folder, dataset_split)
    labels = os.listdir(dataset)
    num_per_shard = 10000
    shard_id = 0

    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d.tfrecord' % (dataset_split, shard_id))
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
    print(FLAGS.output_dir)
    idx = 0
    for label in labels:
        file_path = os.path.join(dataset, label)
        filenames = os.listdir(file_path)
        label = dic[label]

        for filename in filenames:
            image_filename = os.path.join(
                file_path, filename)
            image_data = load_image(image_filename)

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data.tostring())),
                'image/label': _int64_feature(label)
            }))

            tfrecord_writer.write(example.SerializeToString())

            idx += 1
            if idx % num_per_shard == 0:
                shard_id += 1
                output_filename = os.path.join(
                    FLAGS.output_dir,
                    '%s-%05d.tfrecord' % (dataset_split, shard_id))

                tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)

    sys.stdout.write('\n')
    sys.stdout.flush()


def main():
    for dataset_split in os.listdir(FLAGS.image_folder):
        _convert_dataset(dataset_split)


if __name__ == '__main__':
    main()
    # tf.app.run()
