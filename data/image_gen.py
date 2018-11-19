#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:39:45 2018

@author: lidakuang
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_cvs_to_images(args):
    logger.info('[Converting] start converting images')

    path = args.package_path
    data = pd.read_csv(path)[['emotion', 'pixels']]
    # %%
    emotion = data['emotion'].values
    pixels = data['pixels']
    # %%
    images = list(pixels.apply(lambda x: x.split(' ')))
    images = np.asfarray(images)
    emotions = {0: 'angry',
                1: 'disgust',
                2: 'fear',
                3: 'happy',
                4: 'sad',
                5: 'surprise',
                6: 'neutral'}

    outpath = args.image_path
    for i in range(images.shape[1]):
        if i % 200 == 0:
            logger.info("converting percentage %22f" % (i / images.shape[1]))
        img = images[i].reshape(48, 48)
        im = Image.fromarray(img).convert('RGB')
        im_name = '{:04d}'.format(i) + '.jpg'
        if i <= images.shape[1] * 0.8:
            usage = 'train'
        else:
            usage = 'validation'
        im_path = Path(outpath, usage, emotions[emotion[i]])
        im_path.mkdir(parents=True, exist_ok=True)
        im.save(Path(im_path, im_name))

    logger.info('[Converted] csv converted to images in args.image_path')

    return True


def set_up_log(args):
    msg = "User Input:\t{0}".format(" ".join([x for x in sys.argv]))
    logger.info(msg)
    logger.info("----------------")
    for k, v in vars(args).items():
        msg = " {0}:\t{1}".format(k, v)
        logger.info(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert download [data] to tf-record, [data]('
                    'https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition'
                    '-challenge/data)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("package_path",
                        help="path to csv data package")
    parser.add_argument("image_path",
                        help="path to converted image")
    parser.add_argument("tf_record_path",
                        help="path to tf record files")

    return parser.parse_args()


def main():
    args = parse_args()
    set_up_log(args)
    if convert_cvs_to_images(args) is not True:
        logger.error("converting failed")
    else:
        logger.error("converiting success")


if __name__ == "__main__":
    main()
