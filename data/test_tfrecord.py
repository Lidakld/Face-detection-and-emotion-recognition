import tensorflow as tf
import skimage.io as io

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

tfrecords_filename = '/home/zhou/workspace/Face-detection-and-emotion-recognition/data/fer2013/tfrecords/train-00000.tfrecord'

#%%
def read_and_decode(tfrecords_filename, batch_size=1, num_threads=8):
    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/label': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    annotation = tf.decode_raw(features['image/label'], tf.int64)
    image = tf.reshape(image, [48, 48, 1])
    # annotation = tf.reshape(annotation, [1])

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=48,
                                                           target_width=48)

    annotations = tf.train.shuffle_batch([resized_image],
                                                 batch_size=batch_size,
                                                 capacity=30,
                                                 num_threads=num_threads,
                                                 min_after_dequeue=10)

    return annotations


# Even when reading in multiple threads, share the filename
# queue.
annotation = read_and_decode(tfrecords_filename)

# %%
# The op for initializing the variables.
with tf.Session()  as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    anno = sess.run([annotation])

    coord.request_stop()
    coord.join(threads)
    #
    # # Let's read off 3 batches just for example
    # for i in range(3):
    #
    #     print(img[0, :, :, :].shape)
    #
    #     print('current batch')
    #
    #     # We selected the batch size of two
    #     # So we should get two image pairs in each batch
    #     # Let's make sure it is random
    #
    #     io.imshow(img[0, :, :, :])
    #     io.show()
    #
    #     io.imshow(anno[0, :, :, 0])
    #     io.show()
    #
    #     io.imshow(img[1, :, :, :])
    #     io.show()
    #
    #     io.imshow(anno[1, :, :, 0])
    #     io.show()
    #
    # coord.request_stop()
    # coord.join(threads)


#%%
import tensorflow as tf
import numpy as np

tfrecords_filename = '/home/zhou/workspace/Face-detection-and-emotion-recognition/data/fer2013/tfrecords/train-00000.tfrecord'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    img_string = (example.features.feature['image/encoded']
        .bytes_list
        .value[0])

    # Convert to a numpy array (change dtype to the datatype you stored)
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    # Print the image shape; does it match your expectations?
    print(img_1d.shape)

#%%
for example in tf.python_io.tf_record_iterator(tfrecords_filename):
    result = tf.train.Example.FromString(example)

#%%
import collections
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder


data_dir = '/home/zhou/workspace/Face-detection-and-emotion-recognition/data/fer2013/tfrecords'
file_pattern = '%s-*'
file_pattern = os.path.join(data_dir, file_pattern % 'train')


# Specify how the TF-Examples are decoded.
keys_to_features = {
  'image/encoded': tf.FixedLenFeature(
      (), tf.string, default_value=''),
  'image/filename': tf.FixedLenFeature(
      (), tf.string, default_value=''),
  'image/label': tf.FixedLenFeature(
      (), tf.int64, default_value=0)
}

items_to_handlers = {
      'image': tfexample_decoder.Image(
          image_key='image/encoded',
          format_key='jpg',
          channels=1),
      'label': tfexample_decoder.Tensor('image/label')
}


decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

ignore_label = 8
num_classes = 7
dataset_name = 'fer2013_train'
dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=45,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      ignore_label=ignore_label,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)
