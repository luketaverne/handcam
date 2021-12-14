import os
import re
import h5py
import numpy as np
import cv2

import threading

from dask import delayed, threaded, compute
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from handcam.ltt.util.Preprocessing import simple_crop_im, simple_crop_batch
from typing import List


class Handler:
    """
    For handling the UW RGB-D dataset in a python environment.


    """

    top_level_datasets_dir = "/local/home/luke/datasets"
    train_dir = os.path.join(top_level_datasets_dir, "rgbd-dataset")
    labels_file = os.path.join(train_dir, "labels.txt")
    num_examples = 208410  # find /local/home/luke/datasets/rgbd-dataset -type f -name "*_depth.png" | wc -l

    def __init__(self, load_from_file=False, validation_split=0.10):
        print("Loading UW RGB-D dataset...")

        self.validation_split = validation_split

        self.__load_labels__()
        self.__load_object_instances__(load_from_file)
        self.__load_object_poses__(load_from_file)
        self.__calculate_object_counts__(load_from_file)

        self.__handle_validation_split__(validation_split)
        self.__load_hdf5_database__(create_new=False)

        print("Finished loading UW RGB-D dataset.")

    def trainGenerator(self, batch_size):
        # shuffling source : <https://www.kaggle.com/arseni/h5py-dataset-caching-with-shuffled-batch-generator>
        x_train = self.f["data"]
        y_train = self.f["labels"]

        num_train_examples = len(self.train_indicies)
        num_batches_per_epoch = int((num_train_examples - 1) / batch_size) + 1

        epoch = 0

        while 1:
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, num_train_examples)

                batch_indicies = sorted(
                    list(self.train_indicies[start_index:end_index])
                )

                if batch_num == num_batches_per_epoch:
                    # last part of the loop
                    self.train_indicies = np.random.permutation(self.train_indicies)
                    epoch += 1
                # print('gen')

                yield simple_crop_batch(
                    np.asarray(
                        compute(
                            [delayed(x_train.__getitem__)(i) for i in batch_indicies],
                            get=threaded.get,
                        ),
                        dtype=np.uint8,
                    )[0],
                    crop_size=(224, 224),
                ), to_categorical(
                    compute(
                        [delayed(y_train.__getitem__)(i) for i in batch_indicies],
                        get=threaded.get,
                    ),
                    num_classes=self.num_classes,
                )

    def validationGenerator(self, batch_size):
        # shuffling source : <https://www.kaggle.com/arseni/h5py-dataset-caching-with-shuffled-batch-generator>
        x_data = self.f["data"]
        y_data = self.f["labels"]

        num_val_examples = len(self.validation_indicies)
        num_batches_per_epoch = int((num_val_examples - 1) / batch_size) + 1

        epoch = 0

        while 1:
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, num_val_examples)

                batch_indicies = sorted(
                    list(self.validation_indicies[start_index:end_index])
                )

                if batch_num == num_batches_per_epoch:
                    # last part of the loop
                    self.validation_indicies = np.random.permutation(
                        self.validation_indicies
                    )
                    epoch += 1

                yield simple_crop_batch(
                    np.asarray(
                        compute(
                            [delayed(x_data.__getitem__)(i) for i in batch_indicies],
                            get=threaded.get,
                        ),
                        dtype=np.float32,
                    )[0],
                    crop_size=(224, 224),
                ), to_categorical(
                    compute(
                        [delayed(y_data.__getitem__)(i) for i in batch_indicies],
                        get=threaded.get,
                    ),
                    num_classes=self.num_classes,
                )

    def get_random_minibatch(self, batch_size=100, validation=False):
        im_batch = np.zeros(
            (batch_size, 480, 640, 4), dtype=np.uint8
        )  # resize later on using the preprocessing kit
        batch_labels = [""] * batch_size
        object_choices = np.random.choice(
            a=range(len(self.labels)),
            size=batch_size,
            replace=True,
            p=self.object_percentage,
        )
        num_errors = 0
        error_indicies = []

        for i in range(len(object_choices)):
            # Object class is chosen, get the label string
            chosen_label_string = self.labels[object_choices[i]]

            # For the chosen label string, get the object instances and choose an index
            possible_instances_for_chosen_object_class = self.train_label_instances[
                chosen_label_string
            ]
            instance_choice_index = np.random.choice(
                a=range(len(possible_instances_for_chosen_object_class))
            )
            chosen_instance_string = possible_instances_for_chosen_object_class[
                instance_choice_index
            ]

            # For the chosen instance string, get the possible poses and choose one
            poses_for_chosen_instance = self.train_instances_pose_dict[
                chosen_instance_string
            ]
            pose_choice_index = np.random.choice(
                a=range(len(poses_for_chosen_instance))
            )
            chosen_pose_string = poses_for_chosen_instance[pose_choice_index]

            rgb_image_path = os.path.join(
                self.train_dir,
                chosen_label_string,
                chosen_instance_string,
                chosen_pose_string,
            )
            depth_image_path = os.path.join(
                self.train_dir,
                chosen_label_string,
                chosen_instance_string,
                self.__convert_rgb_image_name_to_depth__(chosen_pose_string),
            )

            try:
                rgb_image = self.__load_image_from_file__(image_path=rgb_image_path)
                depth_image = self.__load_image_from_file__(image_path=depth_image_path)
            except AttributeError as e:
                num_errors += 1
                error_indicies.append(i)
                continue

            out_image = np.concatenate((rgb_image, depth_image), axis=2)

            im_batch[i, :, :, :] = out_image
            batch_labels[i] = object_choices[i]  # output the labels as integers

        if num_errors > 0:
            new_mini_mini_batch, new_batch_labels = self.get_random_minibatch(
                batch_size=num_errors
            )

            for i in range(num_errors):
                im_batch[error_indicies[i], :, :, :] = new_mini_mini_batch[i, :, :, :]
                batch_labels[i] = new_batch_labels[i]

        return np.asarray(im_batch, dtype=np.uint8), np.asarray(
            batch_labels, dtype=np.int
        )

    def __load_labels__(self):
        labels = []
        # setup labels for dataset. Train and eval are the same.
        with open(self.labels_file, "r") as file:
            labels = [line.strip() for line in file]
        self.labels = labels
        self.num_classes = len(labels)
        print("Labels loaded.")

    def __load_object_instances__(self, load_from_file):
        if load_from_file:
            # TODO: load from file
            pass
        else:
            train_label_instances = {}

            for i in range(len(self.labels)):
                temp_train_list = []
                temp_eval_list = []
                for x in os.listdir(os.path.join(self.train_dir, self.labels[i])):
                    if os.path.isdir(
                        os.path.join(os.path.join(self.train_dir, self.labels[i]), x)
                    ):
                        temp_train_list.append(x)

                train_label_instances[self.labels[i]] = temp_train_list

            self.train_label_instances = train_label_instances
        print("Instances loaded.")

    def __load_object_poses__(self, load_from_file):
        if load_from_file:
            # TODO: load from file
            pass
        else:
            train_instances_pose_dict = {}

            pose_image_re = r"[a-zA-Z_]+(_\d{1,3}){3}\.png"

            for label, object_instances in self.train_label_instances.items():
                for instance in object_instances:
                    # use regex to filter out the *_depth and *_mask files
                    instance_pose_list = [
                        fn
                        for fn in os.listdir(
                            os.path.join(self.train_dir, label, instance)
                        )
                        if re.match(pose_image_re, fn)
                    ]

                    train_instances_pose_dict[instance] = instance_pose_list

            self.train_instances_pose_dict = train_instances_pose_dict

        print("Instance poses loaded.")

    def __calculate_object_counts__(self, load_from_file):
        """Counts the number of occurances of each object, so that we can balance the training."""
        total_pose_count = 0  # total number of images
        per_object_counts = [0] * len(self.labels)  # counts per label class
        object_percentage = [0] * len(self.labels)
        total_image_count = 0

        for i in range(len(self.labels)):
            num_instances_for_label = 0
            for n in range(len(self.train_label_instances[self.labels[i]])):
                for pose in self.train_instances_pose_dict.keys():
                    num_instances_for_label += 1
                    total_pose_count += 1
                    for image in self.train_instances_pose_dict[pose]:
                        total_image_count += 1

            per_object_counts[i] = num_instances_for_label

            # total_image_count = sum(per_object_counts)

        for i in range(len(object_percentage)):
            object_percentage[i] = per_object_counts[i] / total_pose_count

        self.per_object_counts = per_object_counts
        self.total_pose_count = total_pose_count
        self.object_percentage = object_percentage
        self.total_image_count = total_image_count

        print("Object percentages loaded.")

    def __load_image_from_file__(self, image_path):
        depth = "depth" == image_path.split("_")[-1].split(".")[0]

        imread_flag = cv2.IMREAD_ANYDEPTH if depth else cv2.IMREAD_ANYCOLOR

        out_im = cv2.imread(image_path, imread_flag)

        # try:
        #     out_im = out_im.astype(np.float32)
        # except AttributeError as e:
        #     print("Error with file: " + image_path )
        #     print(e)
        #     raise AttributeError()

        if depth:
            out_im = np.expand_dims(out_im, axis=2)

        return out_im

    def __load_4d_image_from_file__(self, rgb_image_path):
        out_image = np.empty((480, 640, 3), dtype=np.uint8)
        out_depth = np.empty((480, 640, 1), dtype=np.uint16)

        try:
            out_image = self.__load_image_from_file__(image_path=rgb_image_path)
            out_depth = self.__load_image_from_file__(
                image_path=self.__convert_rgb_image_name_to_depth__(rgb_image_path)
            )
        except AttributeError as e:
            print("Problem with image ", rgb_image_path, " %s: ", e)

        return out_image, out_depth

    def __convert_rgb_image_name_to_depth__(self, image_name):
        """Expects image_name as a string ending with .png"""
        return image_name.split(".")[0] + "_depth.png"

    def __convert_rgb_image_name_to_mask__(self, image_name):
        """Expects image_name as a string ending with .png"""
        return image_name.split(".")[0] + "_mask.png"

    def __handle_validation_split__(self, validation_split):
        num_train_examples = int(np.floor(self.num_examples * self.validation_split))
        indicies = np.arange(self.num_examples)
        indicies = np.random.permutation(indicies)

        self.train_indicies = indicies[0:num_train_examples]
        self.validation_indicies = indicies[num_train_examples : self.num_examples]

    def __load_hdf5_database__(self, create_new=False):
        if create_new:
            self.__create_hdf5_database__()
        else:
            self.f = h5py.File(
                os.path.join(self.top_level_datasets_dir, "uw-rgbd.hdf5"), "r"
            )

    def __create_hdf5_database__(self):
        # Assume that we create the whole thing from scratch
        while True:
            response = raw_input(
                "I'm going to overwrite the database. Is that really what you want? (Y/n): "
            )
            if response == "Y":
                break
            elif response in ["n", "N"]:
                print("Okay, quitting")
                return False
            else:
                print("Does not compute. Try again.")

        self.f = h5py.File(
            os.path.join(self.top_level_datasets_dir, "uw-rgbd.hdf5"), "w"
        )
        # uncomment if you want to delete the dataset
        # if 'data' in self.f:
        #     del self.f['data']
        # if 'labels' in self.f:
        #     del self.f['labels']
        rgb = self.f.create_dataset(
            "rgb",
            (self.num_examples, 480, 640, 3),
            chunks=(1, 480, 640, 3),
            dtype=np.uint8,
            compression="gzip",
            shuffle=True,
        )
        depth = self.f.create_dataset(
            "depth",
            (self.num_examples, 480, 640, 1),
            chunks=(1, 480, 640, 1),
            dtype=np.uint16,
            compression="gzip",
            shuffle=True,
        )
        labels = self.f.create_dataset(
            "labels",
            (self.num_examples,),
            dtype=np.uint,
            compression="gzip",
            shuffle=True,
        )

        im_num = 0
        for label_index in range(len(self.labels)):
            label = self.labels[label_index]
            for instance in self.train_label_instances[label]:
                for pose in self.train_instances_pose_dict[instance]:
                    rgb_image_path = os.path.join(self.train_dir, label, instance, pose)
                    (
                        rgb[im_num, :, :, :],
                        depth[im_num, :, :],
                    ) = self.__load_4d_image_from_file__(rgb_image_path=rgb_image_path)
                    labels[im_num] = label_index
                    im_num += 1

                    if im_num % 1000 == 0:
                        print(im_num)

        self.f.flush()
        self.f.close()


class TrainGenerator(object):
    # shuffling source : <https://www.kaggle.com/arseni/h5py-dataset-caching-with-shuffled-batch-generator>
    def __init__(
        self, f: h5py.File, train_indicies: List, num_classes: int, batch_size: int = 32
    ) -> None:

        self.f = f
        self.train_indicies = np.random.permutation(train_indicies)
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.x_train_rgb = self.f["rgb"]
        self.x_train_depth = self.f["depth"]
        self.y_train = self.f["labels"]

        self.image_shape = self.x_train_rgb.shape[1:3]

        self.num_train_examples = len(self.train_indicies)
        self.num_batches_per_epoch = (
            int((self.num_train_examples - 1) / self.batch_size) + 1
        )

        self.epoch = 0
        self.batch_num = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            start_index = self.batch_num * self.batch_size
            end_index = min(
                (self.batch_num + 1) * self.batch_size, self.num_train_examples
            )

            batch_indicies = sorted(list(self.train_indicies[start_index:end_index]))

            if self.batch_num == self.num_batches_per_epoch:
                # last part of the loop
                self.train_indicies = np.random.permutation(self.train_indicies)
                self.epoch += 1
                self.batch_num = 0
            # print('gen')
            else:
                self.batch_num += 1

            return (
                np.asarray(
                    compute(
                        [
                            delayed(self.x_train_rgb.__getitem__)(i)
                            for i in batch_indicies
                        ],
                        get=threaded.get,
                    ),
                    dtype=np.uint8,
                )[0],
                np.asarray(
                    compute(
                        [
                            delayed(self.x_train_depth.__getitem__)(i)
                            for i in batch_indicies
                        ],
                        get=threaded.get,
                    ),
                    dtype=np.uint16,
                )[0],
                to_categorical(
                    compute(
                        [delayed(self.y_train.__getitem__)(i) for i in batch_indicies],
                        get=threaded.get,
                    ),
                    num_classes=self.num_classes,
                ),
            )


class ValidationGenerator(object):
    def __init__(
        self,
        f: h5py.File,
        validation_indicies: List,
        num_classes: int,
        batch_size: int = 32,
    ) -> None:

        self.f = f
        self.validation_indicies = np.random.permutation(validation_indicies)
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.x_val_rgb = self.f["rgb"]
        self.x_val_depth = self.f["depth"]
        self.y_val = self.f["labels"]

        self.image_shape = self.x_val_rgb.shape[1:3]

        self.num_val_examples = len(self.validation_indicies)
        self.num_batches_per_epoch = (
            int((self.num_val_examples - 1) / self.batch_size) + 1
        )

        self.epoch = 0
        self.batch_num = 0
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            start_index = self.batch_num * self.batch_size
            end_index = min(
                (self.batch_num + 1) * self.batch_size, self.num_val_examples
            )

            batch_indicies = sorted(
                list(self.validation_indicies[start_index:end_index])
            )

            if self.batch_num == self.num_batches_per_epoch:
                # last part of the loop
                self.validation_indicies = np.random.permutation(
                    self.validation_indicies
                )
                self.epoch += 1
                self.batch_num = 0
            else:
                self.batch_num += 1
            # print('gen')

            return (
                np.asarray(
                    compute(
                        [
                            delayed(self.x_val_rgb.__getitem__)(i)
                            for i in batch_indicies
                        ],
                        get=threaded.get,
                    ),
                    dtype=np.uint8,
                )[0],
                np.asarray(
                    compute(
                        [
                            delayed(self.x_val_depth.__getitem__)(i)
                            for i in batch_indicies
                        ],
                        get=threaded.get,
                    ),
                    dtype=np.uint16,
                )[0],
                to_categorical(
                    compute(
                        [delayed(self.y_val.__getitem__)(i) for i in batch_indicies],
                        get=threaded.get,
                    ),
                    num_classes=self.num_classes,
                ),
            )
