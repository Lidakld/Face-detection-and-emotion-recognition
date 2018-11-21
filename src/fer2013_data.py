import tensorflow as tf
import matplotlib.pyplot as plt
import math

IMG_SIZE = 48
CLIPED_SIZE = 42
NUM_CHANNEL = 1

EPOCHS = 50
TRAIN_SIZE = 4 * (35887 * 2 - 10000)
BATCH_SIZE = 50

vdic = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}


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


def parser(serialized_example, max_angle=15):
    """Parses a single tf.Example into image and label tensors."""
    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/label': tf.FixedLenFeature([], tf.int64)}

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image/encoded'], tf.float32)
    annotation = features['image/label']
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 1])

    annotation = tf.reshape(annotation, [1])
    angle = tf.random_uniform([1], -1 * max_angle / 2, max_angle / 2) / 180 * math.pi
    image = tf.contrib.image.rotate(image, angle[0])
    image = tf.image.random_crop(image, (CLIPED_SIZE, CLIPED_SIZE, 1))
    # image = tf.image.resize_images(image, tf.convert_to_tensor([IMG_SIZE, IMG_SIZE]))

    return image, annotation


def get_data(tfrecords_filename, batch_size=50, epoches_num=40000, max_angle=15):
    """
    Get data operation

    :param tfrecords_filename: list of file pathes
    :param batch_size:
    :param epoches_num:
    :param max_angle:
    :return:
    """
    # get dataset base
    dataset = tf.data.TFRecordDataset(tfrecords_filename,
                                      buffer_size=2 * batch_size,
                                      num_parallel_reads=8).repeat()
    # parse image and label
    dataset = dataset.map(
        parser, num_parallel_calls=batch_size)

    # Potentially shuffle records.
    min_queue_examples = int(epoches_num * 0.6)
    dataset = dataset.shuffle(buffer_size=50 * 5)

    # shuffle data
    dataset = dataset.batch(batch_size)

    # iterater
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return next_element, iterator


if __name__ == "__main__":
    tfrecords_filename = '/home/zhou/workspace/Face-detection-and-emotion-recognition/data/fer2013/tfrecords/train-00000.tfrecord'

    [image, label], iterator = get_data([tfrecords_filename], batch_size=10, epoches_num=40000, max_angle=15)
    new_annotation = tf.one_hot(tf.squeeze(label), depth=7)
    print(new_annotation)
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        image, annotation, one_hot = sess.run([image, label, new_annotation])

        print(image.shape)
        print("annotation", annotation)
        print("one_hot", one_hot)
        for i in range(10):
            plt.imshow(image[i, :, :, 0])
            plt.title('%d, %s, %s' % (annotation[i, 0], vdic[annotation[i, 0]],
                                      str(one_hot[i])))
            plt.show()

        plt.close()
