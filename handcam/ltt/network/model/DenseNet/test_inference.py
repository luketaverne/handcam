"""Test ImageNet pretrained DenseNet"""
import cv2
import numpy as np

import tensorflow as tf
import tensorflow

from tensorflow.python.keras.optimizers import SGD

# We only test DenseNet-121 in this script for demo purpose
from handcam.ltt.network.model.DenseNet.densenet161 import DenseNet

im = cv2.resize(
    cv2.imread(
        "/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/cat.jpg"
    ),
    (224, 224),
).astype(np.float32)
# im = cv2.resize(cv2.imread('/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/shark.jpg'), (224, 224)).astype(np.float32)

# Subtract mean pixel and multiple by scaling constant
# Reference: https://github.com/shicai/DenseNet-Caffe
im[:, :, 0] = (im[:, :, 0] - 103.94) * 0.017
im[:, :, 1] = (im[:, :, 1] - 116.78) * 0.017
im[:, :, 2] = (im[:, :, 2] - 123.68) * 0.017


# Use pre-trained weights for Tensorflow backend
weights_path = "/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/imagenet_models/densenet161_weights_tf.h5"

# Insert a new dimension for the batch_size
im = np.expand_dims(im, axis=0)

# Test pretrained model
model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

print(model.get_layer(name="conv2_2_x1_scale").input_shape)
print(model.get_layer(name="conv2_2_x1_scale").output_shape)

out = model.predict(im)

# Load ImageNet classes file
classes = []
with open(
    "/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/classes.txt",
    "r",
) as list_:
    for line in list_:
        classes.append(line.rstrip("\n"))

print("Prediction: " + str(classes[np.argmax(out)]))
