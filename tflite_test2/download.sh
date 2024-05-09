#!/usr/bin/sh

curl -L -o lite0-detection-default.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/efficientdet/tfLite/lite0-detection-default/1/download
tar zxvf lite0-detection-default.tar.gz
rm lite0-detection-default.tar.gz
mv 1.tflite lite0-detection-default.tflite

curl -L -o lite0-detection-metadata.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/efficientdet/tfLite/lite0-detection-metadata/1/download
tar zxvf lite0-detection-metadata.tar.gz
rm lite0-detection-metadata.tar.gz
mv 1.tflite lite0-detection-metadata.tflite

unzip lite0-detection-metadata.tflite

curl -L -o lite0-int8.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/efficientdet/tfLite/lite0-int8/1/download
tar zxvf lite0-int8.tar.gz
rm lite0-int8.tar.gz
mv 1.tflite lite0-int8.tflite

curl -L -o ssd_mobilenet_default.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/ssd-mobilenet-v1/tfLite/default/1/download
tar zxvf ssd_mobilenet_default.tar.gz
rm ssd_mobilenet_default.tar.gz
mv 1.tflite ssd_mobilenet_default.tflite
