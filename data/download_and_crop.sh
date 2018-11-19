#!/usr/bin/env bash

# Donwload data from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
# and covnerted it to tf-record

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./fer2013"
mkdir -p "${WORK_DIR}"

# Download the csv
BASE_URL="https://s3-us-west-2.amazonaws.com/cs522.com/fer2013.tar.gz"
FILENAME="fer2013.tar.gz"

if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget "${BASE_URL}"
    tar xvf ${FILENAME}
fi

# Convert csv to image and tf record
mkdir -p ${WORK_DIR}/images
mkdir -p ${WORK_DIR}/tfrecords
mkdir -p ${WORK_DIR}/train
mkdir -p ${WORK_DIR}/eval
mkdir -p ${WORK_DIR}/inference

# to images
python3 ./image_gen.py ${WORK_DIR}/fer2013.csv \
	 ${WORK_DIR}/images \
	 ${WORK_DIR}/tfrecords 

# to tf records
python3 ./build_fer2013_data.py  --image_folder ${WORK_DIR}/images \
                --output_dir ${WORK_DIR}/tfrecords

echo "[Finished] Data is ready!"