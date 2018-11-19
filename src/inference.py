# Author 'fxiong'
"""
Created on Sun Nov 18 08:14:42 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

tf.app.flags.DEFINE_string('tf_model',
                           os.path.abspath('../data/fer2013/train'),
                           'Folder containing images.')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_SIZE = 48

