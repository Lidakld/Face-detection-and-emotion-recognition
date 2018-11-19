import tensorflow as tf
import matplotlib.pyplot as plt
import math

IMG_SIZE = 48


def read_and_decode(tfrecords_filenames, epoch_iteration=1, batch_size=30, num_threads=8, max_angle=15):
    filename_queue = tf.train.string_input_producer([tfrecords_filenames], num_epochs=epoch_iteration)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/label': tf.FixedLenFeature([], tf.int64)}

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image/encoded'], tf.float32)
    annotation = features['image/label']
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 1])

    annotation = tf.reshape(annotation, [1])
    image, annotation = tf.train.shuffle_batch([image, annotation],
                                               batch_size=batch_size,
                                               capacity=30,
                                               num_threads=num_threads,
                                               min_after_dequeue=10)
    angle = tf.random_uniform([batch_size, 1], -1 * max_angle / 2, max_angle / 2) / 180 * math.pi
    image = tf.contrib.image.rotate(image, angle[0])

    # annotation = tf.one_hot(annotation, depth=7)
    # return image, annotation
    return (image, annotation).make_one_shot_iterator()


def test_read(tfrecords_filename, epoch_iteration):
    images, annotations = read_and_decode(tfrecords_filename, epoch_iteration)

    vdic = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise'
    }

    with tf.Session()  as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image, annotation = sess.run([images, annotations])

        print("annotation", annotation)
        # Let's read off 3 batches just for example
        try:
            for i in range(3):
                plt.imshow(image[i, :, :, 0])
                plt.title(annotation[i, 0])
                plt.show()

            plt.close()
        except:
            pass

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()


def inspect_content(tfrecords_filename):
    # %%
    import tensorflow as tf
    import numpy as np

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        # Parse the next example
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features.feature['image/encoded']
            .bytes_list
            .value[0])

        # Convert to a numpy array (change dtype to the datatype you stored)
        img_1d = np.fromstring(img_string, dtype=np.float32)
        # Print the image shape; does it match your expectations?
        print(img_1d.shape)

        # label
        label_v = (example.features.feature['image/label'])
        print(label_v)


if __name__ == "__main__":
    tfrecords_filename = '/home/zhou/workspace/Face-detection-and-emotion-recognition/data/fer2013/tfrecords/train-00000.tfrecord'

    test_read(tfrecords_filename, 2)
    # inspect_content(tfrecords_filename)
    # for example in tf.python_io.tf_record_iterator(tfrecords_filename):
    #     result = tf.train.Example.FromString(example)
    #
    #     print(result)

# %%
# tfrecords_filename = '/home/zhou/workspace/Face-detection-and-emotion-recognition/data/fer2013/tfrecords/train-00000.tfrecord'
# dataset = tf.data.TFRecordDataset([tfrecords_filename], buffer_size=1024, num_parallel_reads=8)
# iterator = dataset.make_one_shot_iterator()
# next_batch = iterator.get_next()
#
# sess = tf.Session()
# print("epoch 1")
# for _ in range(4):
#     print(sess.run(next_batch))
