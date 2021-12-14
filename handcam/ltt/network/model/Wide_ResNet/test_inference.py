"""Test ImageNet pretrained DenseNet"""
import cv2
import numpy as np

import tensorflow as tf
import tensorflow

from tensorflow.python.keras.optimizers import SGD

# We only test DenseNet-121 in this script for demo purpose
from handcam.ltt.network.model.Wide_ResNet import wrn_keras_luke as wrn

# im = cv2.resize(cv2.imread('/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/cat.jpg'), (224, 224)).astype(np.float32)
im = cv2.resize(
    cv2.imread(
        "/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/shark.jpg"
    ),
    (224, 224),
).astype(np.float32)


# Use pre-trained weights for Tensorflow backend
weights_path = "/local/home/luke/wrn-50-2-keras.h5"

# Insert a new dimension for the batch_size
im = np.expand_dims(im, axis=0)

# Test pretrained model
model = wrn.define_keras_model((224, 224, 3), nb_classes=1000)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

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
