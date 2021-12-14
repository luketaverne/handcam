from datetime import datetime

from tensorflow.python.keras.callbacks import Callback

from handcam.ltt.datasets.handcam.HandCamDataHandler import HandGenerator
from handcam.ltt.datasets.uw_rgbd.UWDataHandler import (
    TrainGenerator,
    ValidationGenerator,
)
from handcam.ltt.network.Tools import write_log
from handcam.ltt.util.Preprocessing import DataAugmentation


import cv2
import numpy as np

import tensorflow as tf
import tensorflow

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, ProgbarLogger
from tensorflow.python.keras.callbacks import TensorBoard


from handcam.ltt.datasets.uw_rgbd import UWDataHandler
from handcam.ltt.datasets.handcam import HandCamDataHandler

uw_handler = UWDataHandler.Handler()
data_handler_handcam = HandCamDataHandler.Handler()
"""
Configuration
"""
log_dir = "/local/home/luke/programming/master-thesis/python/logs/WRNKeras/"
config = {}

config["loss"] = "categorical_crossentropy"
config["num_epochs"] = 100000
config["batch_size"] = 32
config["train_names"] = ["train_loss", "train_accuracy"]
config["val_names"] = ["val_loss", "val_accuracy"]
config["log-dir"] = log_dir + datetime.now().strftime("%Y-%m-%d-%H:%M")


"""
Begin setting up DenseNet
"""
from handcam.ltt.network.model.Wide_ResNet import wrn_keras_luke

# im = cv2.resize(cv2.imread('/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/cat.jpg'), (224, 224)).astype(np.float32)
# im = cv2.resize(cv2.imread('/local/home/luke/programming/master-thesis/python/ltt/network/model/DenseNet/resources/shark.jpg'), (224, 224)).astype(np.float32)

# Subtract mean pixel and multiple by scaling constant
# Reference: https://github.com/shicai/DenseNet-Caffe
# im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
# im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
# im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

# Use pre-trained weights from ImageNet
weights_path = "/local/home/luke/programming/master-thesis/python/ltt/network/model/Wide_ResNet/weights/WRNKeras-test-weights.hdf5"

# add a temp dimension for depth
# print(im.shape)
# im = np.concatenate((im,np.zeros((224,224,1))),axis=2)
# print(im.shape)

# Insert a new dimension for the batch_size
# im = np.expand_dims(im, axis=0)
# print(im.shape)

# Test pretrained model

# After the next line, `model` will have the pre-trained weights loaded
model = wrn_keras_luke.define_keras_model((224, 224, 4), classes=uw_handler.num_classes)

# sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam()
model.compile(optimizer=adam, loss=config["loss"], metrics=["accuracy"])
# model.load_weights(weights_path)

checkpointer = ModelCheckpoint(
    filepath="/tmp/WRNKeras-test-weights.hdf5",
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
batch_no = 0
epoch = 1

# Setup the data_augmentation
# TODO: Make this less of a mess???
hand_generator = HandGenerator(
    f=data_handler_handcam.f, batch_size=config["batch_size"]
)

train_generator = TrainGenerator(
    f=uw_handler.f,
    train_indicies=uw_handler.train_indicies,
    num_classes=uw_handler.num_classes,
    batch_size=config["batch_size"],
)

data_augmentation = DataAugmentation(
    image_generator=train_generator,
    hand_generator=hand_generator,
    rotations=True,
    center_crop=(224, 224),
    batch_size=config["batch_size"],
)

validation_generator = ValidationGenerator(
    f=uw_handler.f,
    validation_indicies=uw_handler.validation_indicies,
    num_classes=uw_handler.num_classes,
    batch_size=config["batch_size"],
)

validation_augmentation = DataAugmentation(
    image_generator=validation_generator,
    hand_generator=hand_generator,
    center_crop=(224, 224),
    rotations=True,
    batch_size=config["batch_size"],
)

# print(next(validation_augmentation)[0].shape)

# while True:
#     next(train_generator)
#
# steps_per_epoch = int(np.floor(data_handler.num_examples/config['batch_size']))
fake_steps_per_epoch = 100


callback_list = [tb, checkpointer, progbar]

# while True:
#     im_batch, _ = next(data_augmentation)
#     cv2.imshow('frame', im_batch[0,:,:,0:3])
#     cv2.waitKey(34)
#     print('im')


model.fit_generator(
    data_augmentation,
    steps_per_epoch=fake_steps_per_epoch,
    epochs=config["num_epochs"],
    verbose=1,
    callbacks=callback_list,
    validation_data=validation_augmentation,
    validation_steps=10,
    class_weight=uw_handler.object_percentage,
    workers=1,
    use_multiprocessing=False,
    max_queue_size=100,
)
