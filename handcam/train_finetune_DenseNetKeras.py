"""Test ImageNet pretrained DenseNet"""
from datetime import datetime

from tensorflow.python.keras.callbacks import Callback

from handcam.ltt.network.Tools import write_log
from handcam.ltt.util.Preprocessing import simple_crop


import cv2
import numpy as np

import tensorflow as tf
import tensorflow

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, ProgbarLogger
from tensorflow.python.keras.callbacks import TensorBoard


from handcam.ltt.datasets.uw_rgbd import UWDataHandler

data_handler = UWDataHandler.Handler()
"""
Configuration
"""
log_dir = "/local/home/luke/programming/master-thesis/python/logs/DenseNetKeras/"
config = {}

config["loss"] = "categorical_crossentropy"
config["num_epochs"] = 100000
config["batch_size"] = 6
config["train_names"] = ["train_loss", "train_accuracy"]
config["val_names"] = ["val_loss", "val_accuracy"]
config["log-dir"] = log_dir + datetime.now().strftime("%Y-%m-%d-%H:%M")


"""
Begin setting up DenseNet
"""
from handcam.ltt.network.model.DenseNet.densenet161_depth import DenseNet

# im = cv2.resize(cv2.imread('/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/cat.jpg'), (224, 224)).astype(np.float32)
# im = cv2.resize(cv2.imread('/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/shark.jpg'), (224, 224)).astype(np.float32)

# Subtract mean pixel and multiple by scaling constant
# Reference: https://github.com/shicai/DenseNet-Caffe
# im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
# im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
# im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

# Use pre-trained weights from ImageNet
weights_path = "/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/imagenet_models/densenet161_weights_tf.h5"

# add a temp dimension for depth
# print(im.shape)
# im = np.concatenate((im,np.zeros((224,224,1))),axis=2)
# print(im.shape)

# Insert a new dimension for the batch_size
# im = np.expand_dims(im, axis=0)
# print(im.shape)

# Test pretrained model

# After the next line, `model` will have the pre-trained weights loaded
model = DenseNet(
    reduction=0.5, classes=data_handler.num_classes, weights_path=weights_path
)

# sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam()
model.compile(optimizer=adam, loss=config["loss"], metrics=["accuracy"])

checkpointer = ModelCheckpoint(
    filepath="/tmp/DenseNetKeras-test-weights.hdf5",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)
checkpointer.set_model(model)
progbar = ProgbarLogger(count_mode="steps")
progbar.set_model(model)
tb = TensorBoard(config["log-dir"], batch_size=config["batch_size"])
tb.set_model(model)


class printbatch(Callback):
    def on_batch_end(self, epoch, logs={}):
        print(logs)


pb = printbatch()

"""
Begin training loop
"""
batch_no = 1
epoch = 0

train_generator = data_handler.trainGenerator(batch_size=config["batch_size"])
validation_generator = data_handler.validationGenerator(batch_size=config["batch_size"])

steps_per_epoch = int(np.floor(data_handler.num_examples / config["batch_size"]))
fake_steps_per_epoch = 1000


callback_list = [tb, checkpointer, progbar]


model.fit_generator(
    train_generator,
    steps_per_epoch=fake_steps_per_epoch,
    epochs=config["num_epochs"],
    verbose=1,
    callbacks=callback_list,
    validation_data=validation_generator,
    validation_steps=10,
    class_weight=data_handler.object_percentage,
    workers=1,
)
