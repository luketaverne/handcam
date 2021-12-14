import threading
from datetime import datetime
from typing import List
from handcam.ltt.datasets.handcam.HandCamDataHandler import HandGenerator
from handcam.ltt.datasets.uw_rgbd.UWDataHandler import (
    TrainGenerator,
    ValidationGenerator,
)
from handcam.ltt.network.Tools import write_log
from handcam.ltt.util.Preprocessing import DataAugmentation
from dask import delayed, threaded, compute
from tensorflow.python.keras.utils import to_categorical
import h5py

import cv2
import numpy as np

from handcam.ltt.datasets.uw_rgbd import UWDataHandler
from handcam.ltt.datasets.handcam import HandCamDataHandler

import multiprocessing as mp


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

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

# Make new class for faster generation


class SpeedyUWGenerator(object):
    def __init__(
        self, handler: UWDataHandler, batch_size: int = 32, gen_type: str = "train"
    ) -> None:
        if gen_type not in ["train", "val"]:
            raise ValueError("gen_type should be one of 'train' or 'test'")

        self.handler = handler
        self.batch_size = batch_size
        self.indices = (
            self.handler.train_indicies
            if gen_type == "train"
            else self.handler.validation_indicies
        )
        self.indices = np.random.permutation(self.indices)
        self.num_batches_in_RAM = 5
        self.load_new_RAM_after = 3
        self.num_samples_in_RAM = self.num_batches_in_RAM * self.batch_size
        self.loading_to_RAM = False
        self.ready_to_switch_RAM = True
        self.temp_RAM_loaded = False

        self.x_rgb = self.handler.f["rgb"]
        self.x_depth = self.handler.f["depth"]
        self.y = self.handler.f["labels"]

        self.x_rgb_RAM = None
        self.x_depth_RAM = None
        self.y_RAM = None

        self.num_classes = self.handler.num_classes  # for UW dataset
        self.num_examples = len(self.indices)
        self.num_batches_per_epoch = int((self.num_examples - 1) / self.batch_size) + 1

        self.epoch = 0
        self.epoch_RAM = 0
        self.batch_num = 0
        self.batch_num_RAM = 0
        # self.lock = threading.Lock()

        self.ready_to_switch_RAM = True

        self.__start_next_preproc()
        self.__wait_for_RAM_job()

        print("Init done")

    def __wait_for_RAM_job(self):
        for job in jobs:
            job.get()

    def __start_next_preproc(self):
        jobs.append(pool.apply_async(self.__prepare_RAM_batch, self))

    def __prepare_RAM_batch(self):
        self.__load_to_RAM()
        self.__switch_RAM()

    def __load_to_RAM(self):
        if self.loading_to_RAM:
            # Already reading, don't want to do it more
            return
        print("Loading to RAM")
        self.loading_to_RAM = True

        if self.batch_num_RAM == self.num_batches_per_epoch:
            self.indices = np.random.permutation(self.indices)
            self.epoch_RAM += 1
            self.batch_num_RAM = 0
        else:
            self.batch_num_RAM += 1

        start_index_RAM = self.batch_num_RAM * self.num_samples_in_RAM
        end_index_RAM = min(
            (self.batch_num_RAM + 1) * self.num_samples_in_RAM, self.num_examples
        )

        batch_indicies = list(self.indices[start_index_RAM:end_index_RAM])

        self.temp_x_rgb, self.temp_x_depth, self.temp_y = (
            np.asarray(
                compute(
                    [delayed(self.x_rgb.__getitem__)(i) for i in batch_indicies],
                    get=threaded.get,
                ),
                dtype=np.uint8,
            )[0],
            np.asarray(
                compute(
                    [delayed(self.x_depth.__getitem__)(i) for i in batch_indicies],
                    get=threaded.get,
                ),
                dtype=np.uint16,
            )[0],
            to_categorical(
                compute(
                    [delayed(self.y.__getitem__)(i) for i in batch_indicies],
                    get=threaded.get,
                ),
                num_classes=self.num_classes,
            ),
        )

        self.temp_RAM_loaded = True
        print("Loaded to RAM")

    def __switch_RAM(self):
        while (self.ready_to_switch_RAM == False) or (self.temp_RAM_loaded == False):
            pass
        print("Switching RAM")
        self.x_rgb_RAM, self.x_depth_RAM, self.y_RAM = (
            self.temp_x_rgb,
            self.temp_x_depth,
            self.temp_y,
        )
        self.ready_to_switch_RAM = False
        self.loading_to_RAM = False

        del self.temp_y, self.temp_x_depth, self.temp_x_rgb

        print("RAM switched")

    # def augment_im(self, x_rgb, x_depth):
    #     x_rgb

    def __iter__(self):
        return self

    def __next__(self):
        # with self.lock:
        reset_once = True
        while self.__wait_for_RAM_job():
            # Do nothing until the stuff is in ram
            if reset_once:
                self.indices = np.random.permutation(self.indices)
                self.epoch += 1
                self.batch_num = 0
                reset_once = False
            pass

        if not self.ready_to_switch_RAM and self.batch_num > self.load_new_RAM_after:
            self.__start_next_preproc()  # start getting next batch ready.

        start_index = self.batch_num * self.batch_size
        end_index = (self.batch_num + 1) * self.batch_size

        # batch_indicies = sorted(list(self.indices[start_index:end_index]))
        batch_indicies = range(start_index, end_index)

        if self.batch_num == self.num_batches_in_RAM:
            # last part of the loop
            self.indices = np.random.permutation(self.indices)
            self.epoch += 1
            self.batch_num = 0
            self.ready_to_switch_RAM = True

        else:
            self.batch_num += 1
        print("gen")

        return (
            np.asarray(
                compute(
                    [delayed(self.x_rgb_RAM.__getitem__)(i) for i in batch_indicies],
                    get=threaded.get,
                ),
                dtype=np.uint8,
            )[0],
            np.asarray(
                compute(
                    [delayed(self.x_depth_RAM.__getitem__)(i) for i in batch_indicies],
                    get=threaded.get,
                ),
                dtype=np.uint16,
            )[0],
            to_categorical(
                compute(
                    [delayed(self.y_RAM.__getitem__)(i) for i in batch_indicies],
                    get=threaded.get,
                ),
                num_classes=self.num_classes,
            ),
        )


#
# def process_wrapper(start_index, end_index):
#


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
    simple_crop=(224, 224),
    batch_size=config["batch_size"],
)

pool = mp.Pool(processes=4)
jobs = []

datagen = SpeedyUWGenerator(handler=uw_handler, batch_size=32, gen_type="train")

while True:
    next(datagen)
