import subprocess

from handcam.ltt.util.TFTools import (
    _DatasetInitializerHook,
    shuffle_dataset,
    per_sequence_standardization,
    per_sequence_standardization_rgbd,
)

import tensorflow as tf
import glob
import sys
import numpy as np
import six
import os
import pickle
import datetime

# from handcam.ltt.network.model.Wide_ResNet import wide_resnet_tf_depth as resnet_model_rgbd
from handcam.ltt.network.model.Wide_ResNet import wide_resnet_tf as resnet_model
from handcam.ltt.network.model.RNNModel import LSTMModel as lstm_model
from handcam.ltt.util.Utils import AttrDict, handcam_gesture_spotting_acc

models_to_eval = {
    #     # 'split0/single_frames_resnet-18/rgb/train/2018-08-14/11:37': None, #best rgb
    #     # 'split0/single_frames_resnet-18/rgbd/train/2018-08-14/10:52': None, #best rgbd
    #     # 'split0/single_frames_resnet-18/depth/train/2018-08-14/15:09': None #best depth
    #     # 'split0/sequence_resnet-18/rgbd/frozen_train/2018-08-15/01:55': None,
    #     # 'split0/sequence_resnet-18/rgb/frozen_train/2018-08-14/21:35': None,
    #     # 'split0/sequence_resnet-18/depth/frozen_train/2018-08-14/17:18': None,
    #     # 'split0/sequence_resnet-18/rgb/train/2018-08-15/20:45': None,
    #     # 'split0/sequence_resnet-18/depth/train/2018-08-15/16:23': None,
    #     # 'split0/sequence_resnet-18/rgbd/train/2018-08-16/00:25': None,
    #     '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/depth/frozen_train/2018-08-20/19:40': None,
    "split0/sequence_resnet-18/depth/train/2018-08-20/22:25": None,
    #     '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgbd/frozen_train/2018-08-20/21:32': None,
    #     '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgbd/train/2018-08-21/00:01': None,
    #     '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgb/frozen_train/2018-08-20/20:22': None,
    "split0/sequence_resnet-18/rgb/train/2018-08-20/23:09": None,
    #     '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/depth/frozen_train/2018-08-21/21:19': None,
    "split1/sequence_resnet-18/depth/train/2018-08-21/23:38": None,
    #     '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgbd/frozen_train/2018-08-21/22:40': None,
    #     '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgbd/train/2018-08-22/01:17': None,
    #     '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgb/frozen_train/2018-08-21/21:57': None,
    "split1/sequence_resnet-18/rgb/train/2018-08-22/00:17": None,
    #     '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-50/depth/frozen_train/2018-08-23/19:12': None,
    #     '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-50/rgbd/frozen_train/2018-08-23/23:04': None,
    #     '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-50/rgb/frozen_train/2018-08-23/20:16': None,
    #     '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/depth/frozen_train/2018-08-22/04:11': None,
    "split2/sequence_resnet-18/depth/train/2018-08-22/08:31": None,
    #     '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgbd/frozen_train/2018-08-22/06:55': None,
    #     '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgbd/train/2018-08-22/09:59': None,
    #     '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgb/frozen_train/2018-08-22/04:55': None,
    "split2/sequence_resnet-18/rgb/train/2018-08-22/08:55": None,
    #     '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-50/depth/frozen_train/2018-08-24/04:35': None,
    #     '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-50/rgbd/frozen_train/2018-08-24/07:02': None,
    #     '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-50/rgb/frozen_train/2018-08-24/05:26': None,
    #     '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/depth/frozen_train/2018-08-22/12:23': None,
    "split3/sequence_resnet-18/depth/train/2018-08-22/15:58": None,
    #     '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgbd/frozen_train/2018-08-22/15:08': None,
    #     '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgbd/train/2018-08-22/18:08': None,
    #     '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgb/frozen_train/2018-08-22/13:59': None,
    "split3/sequence_resnet-18/rgb/train/2018-08-22/17:17": None,
    #     '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-50/depth/frozen_train/2018-08-24/12:50': None,
    #     '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-50/rgbd/frozen_train/2018-08-24/16:53': None,
    #     '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-50/rgb/frozen_train/2018-08-24/14:32': None,
    #     '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/depth/frozen_train/2018-08-22/20:36': None,
    "split4/sequence_resnet-18/depth/train/2018-08-22/23:04": None,
    #     '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgbd/frozen_train/2018-08-22/22:14': None,
    #     '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgbd/train/2018-08-23/01:16': None,
    #     '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgb/frozen_train/2018-08-22/21:36': None,
    "split4/sequence_resnet-18/rgb/train/2018-08-23/00:10": None,
    #     '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-50/depth/frozen_train/2018-08-24/23:00': None,
    #     '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-50/rgbd/frozen_train/2018-08-25/04:29': None,
    #     '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-50/rgb/frozen_train/2018-08-25/01:49': None,
    #     '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/depth/frozen_train/2018-08-23/04:06': None,
    "split5/sequence_resnet-18/depth/train/2018-08-23/06:34": None,
    #     '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgbd/frozen_train/2018-08-23/05:55': None,
    #     '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgbd/train/2018-08-23/08:19': None,
    #     '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgb/frozen_train/2018-08-23/05:13': None,
    "split5/sequence_resnet-18/rgb/train/2018-08-23/07:38": None,
    #     '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-50/depth/frozen_train/2018-08-25/11:27': None,
    #     '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-50/rgbd/frozen_train/2018-08-25/15:06': None,
    #     '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-50/rgb/frozen_train/2018-08-25/13:11': None,
    #     '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/depth/frozen_train/2018-08-23/12:15': None,
    "split6/sequence_resnet-18/depth/train/2018-08-23/15:13": None,
    #     '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgbd/frozen_train/2018-08-23/13:58': None,
    #     '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgbd/train/2018-08-23/17:53': None,
    #     '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgb/frozen_train/2018-08-23/12:50': None,
    "split6/sequence_resnet-18/rgb/train/2018-08-23/16:47": None,
    #     '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-50/depth/frozen_train/2018-08-26/00:18': None,
    #     '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-50/rgbd/frozen_train/2018-08-26/03:24': None,
    #     '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-50/rgb/frozen_train/2018-08-26/01:13': None,
    #     '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/depth/frozen_train/2018-08-22/16:30': None,
    "split7/sequence_resnet-18/depth/train/2018-08-22/21:24": None,
    #     '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgb/frozen_train/2018-08-22/18:23': None,
    "split7/sequence_resnet-18/rgb/train/2018-08-22/22:36": None,
    #     '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgbd/frozen_train/2018-08-28/10:04': None,
    #     '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgbd/train/2018-08-28/10:38': None,
    #     '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-50/depth/frozen_train/2018-08-26/11:24': None,
    #     '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-50/rgbd/frozen_train/2018-08-26/14:05': None,
    #     '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-50/rgb/frozen_train/2018-08-26/12:13': None,
    #     '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/depth/frozen_train/2018-08-23/00:54': None,
    "split8/sequence_resnet-18/depth/train/2018-08-23/03:20": None,
    #     '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgbd/frozen_train/2018-08-23/02:27': None,
    #     '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgbd/train/2018-08-23/04:51': None,
    #     '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgb/frozen_train/2018-08-23/01:27': None,
    "split8/sequence_resnet-18/rgb/train/2018-08-23/04:08": None,
    #     '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-50/depth/frozen_train/2018-08-26/22:18': None,
    #     '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-50/rgbd/frozen_train/2018-08-27/02:58': None,
    #     '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-50/rgb/frozen_train/2018-08-26/23:59': None,
    #     '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/depth/frozen_train/2018-08-23/07:21': None,
    "split9/sequence_resnet-18/depth/train/2018-08-23/09:15": None,
    #     '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgbd/frozen_train/2018-08-23/08:35': None,
    #     '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgbd/train/2018-08-23/10:38': None,
    #     '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgb/frozen_train/2018-08-23/07:58': None,
    "split9/sequence_resnet-18/rgb/train/2018-08-23/09:49": None,
    #     '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-50/depth/frozen_train/2018-08-27/07:32': None,
    #     '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-50/rgbd/frozen_train/2018-08-27/12:23': None,
    #     '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-50/rgb/frozen_train/2018-08-27/09:29': None,
    "split0/sequence_resnet-18/rgbd/train/2018-08-21/00:01": 8400,
    "split1/sequence_resnet-18/rgbd/train/2018-08-22/01:17": 6001,
    "split2/sequence_resnet-18/rgbd/train/2018-08-22/09:59": 6101,
    "split3/sequence_resnet-18/rgbd/train/2018-08-22/18:08": 6251,
    "split4/sequence_resnet-18/rgbd/train/2018-08-23/01:16": 10151,
    "split5/sequence_resnet-18/rgbd/train/2018-08-23/08:19": 13601,
    "split6/sequence_resnet-18/rgbd/train/2018-08-23/17:53": 4200,
    "split7/sequence_resnet-18/rgbd/train/2018-08-28/10:38": 6300,
    "split8/sequence_resnet-18/rgbd/train/2018-08-23/04:51": 4200,
    "split9/sequence_resnet-18/rgbd/train/2018-08-23/10:38": 8000,
}

# dict[resnet_size][seq_or_single][modality][gesture_or_accuracy]
compiled_results_dict = {
    # 'resnet-50': {
    #     'single_frames': {},
    #     'sequence_frozen': {}
    # },
    "resnet-18": {
        # 'single_frames': {},
        # 'sequence_frozen': {},
        "sequence_end2end": {}
    }
}

for resnet_type in compiled_results_dict.keys():
    for model_type in compiled_results_dict[resnet_type].keys():
        for split_id in range(0, 10):
            compiled_results_dict[resnet_type][model_type]["split%d" % split_id] = {}
            if split_id == 0 and resnet_type == "resnet-50":
                if model_type == "single_frames":
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgb"] = {"accuracy": 0.9818, "gesture_spotting": 0.9945}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["depth"] = {"accuracy": 0.8604, "gesture_spotting": 0.9394}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgbd"] = {"accuracy": 0.9813, "gesture_spotting": 0.9924}
                else:
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgb"] = {"accuracy": 0.9540, "gesture_spotting": 0.9922}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["depth"] = {"accuracy": 0.9461, "gesture_spotting": 0.9864}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgbd"] = {"accuracy": 0.9602, "gesture_spotting": 0.9919}
                # need to put the numbers from the paper in here, models are gone.
                pass
            else:
                for modality in ["depth", "rgb", "rgbd"]:
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ][modality] = {"accuracy": None, "gesture_spotting": None}


class_names = [
    "grasp_1",
    "grasp_2",
    "grasp_3",
    "grasp_4",
    "grasp_5",
    "grasp_6",
    "grasp_7",
]
class_names_to_index = {
    "grasp_1": 0,
    "grasp_2": 1,
    "grasp_3": 2,
    "grasp_4": 3,
    "grasp_5": 4,
    "grasp_6": 5,
    "grasp_7": 6,
}

features_single_frame = {
    "image/img": tf.FixedLenFeature((), tf.string, default_value=""),
    "sample_name": tf.FixedLenFeature((), tf.string, default_value=""),
    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
    "image/frame_num": tf.FixedLenFeature((), tf.int64, default_value=0),
}

context_features = {
    "vid_length": tf.FixedLenFeature((), tf.int64),
    "first_grasp_frame": tf.FixedLenFeature((), tf.int64),
    "last_grasp_frame": tf.FixedLenFeature((), tf.int64),
    "sample_name": tf.FixedLenFeature((), tf.string),
}

sequence_features = {
    "vid": tf.FixedLenSequenceFeature([], dtype=tf.string),
    # 'accel': tf.FixedLenSequenceFeature([], dtype=tf.string),
    # 'gyro': tf.FixedLenSequenceFeature([], dtype=tf.string),
    # 'pose': tf.FixedLenSequenceFeature([], dtype=tf.string),
    "frame_labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
}


def get_latest_checkpoint_id(path):
    path = os.path.join(path, "model-*.index")
    base_command = "ls %s | head -n 1" % path
    result = subprocess.run(
        [base_command], stdout=subprocess.PIPE, shell=True
    ).stdout.decode("utf-8")
    latest_id = result.split("/")[-1].split("model-")[-1].split(".index")[0]

    return latest_id


# run for each model. Load the FLAGS.pckl file to set everything up the same way as for training
for model_path in models_to_eval.keys():
    checkpoint_id = models_to_eval[model_path]

    if (
        "/tmp/luke/handcam" not in model_path
        and "/media/luke/hdd-3tb" not in model_path
    ):
        model_path = os.path.join("/tmp/luke/handcam/", model_path)

    with open(os.path.join(model_path, "FLAGS.pckl"), "rb") as f:
        flags_dict = pickle.load(f)
        FLAGS = AttrDict(flags_dict)  # allow attribute access to FLAGS.

    # need to modify FLAGS a bit
    FLAGS.batch_size = 1

    for key, val in FLAGS.items():
        print("%s, %s" % (key, val))

    def _parse_function_single_frame(example_proto):
        # parsed_features = tf.parse_single_example(example_proto, features)
        print("next")
        features_parsed = tf.parse_single_example(example_proto, features_single_frame)

        label = features_parsed["image/class/label"]
        sample_name = tf.decode_raw(features_parsed["sample_name"], tf.uint8)

        # frame_labels = tf.to_int32(sequence_parsed["frame_labels"])
        img = tf.decode_raw(features_parsed["image/img"], tf.uint16)
        img = tf.reshape(img, [240, 320, 4])
        img.set_shape([240, 320, 4])
        if FLAGS.input_modality == "rgb":
            img = img[..., 0:3]
            img.set_shape([240, 320, 3])
        elif FLAGS.input_modality == "depth":
            img = img[..., 3:]
            img.set_shape([240, 320, 1])
        elif FLAGS.input_modality == "rgbd":
            img.set_shape([240, 320, 4])

        return img, label, sample_name

    def _parse_function_sequence(example_proto):
        # parsed_features = tf.parse_single_example(example_proto, features)
        print("next")
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            example_proto, context_features, sequence_features
        )
        seq_len = tf.to_int32(context_parsed["vid_length"])
        first_grasp_frame = tf.to_int32(context_parsed["first_grasp_frame"])
        last_grasp_frame = tf.to_int32(context_parsed["last_grasp_frame"])
        sample_name = tf.decode_raw(context_parsed["sample_name"], tf.uint8)

        img = tf.decode_raw(sequence_parsed["vid"], tf.uint16)
        img = tf.reshape(img, [-1, 240, 320, 4])

        img = tf.cast(img, tf.float32)
        one_hot = tf.one_hot(
            sequence_parsed["frame_labels"], FLAGS.num_classes, dtype=tf.int64
        )
        # imu_data = [0.1]

        try:
            if FLAGS.with_naive_IMU:
                accel = tf.decode_raw(sequence_parsed["accel"], tf.float32)
                accel = tf.reshape(
                    accel, [-1, 3]
                )  # [-1,3] because I stripped the timestamps in the tfrecords
                gyro = tf.decode_raw(sequence_parsed["gyro"], tf.float32)
                gyro = tf.reshape(gyro, [-1, 3])

                imu_data = tf.concat([accel, gyro], axis=1)
                print(imu_data.shape)
        except KeyError as e:
            pass

        return img, one_hot, seq_len, first_grasp_frame, last_grasp_frame, sample_name

    def _center_crop_single_frame(img, label, sample_name):
        img = img[8:232, 48:272, :]
        return img, label, sample_name

    def _preprocessing_op_single_frame(img, label, sample_name):
        if FLAGS.input_modality == "rgb":
            img = tf.image.resize_images(
                img,
                size=[FLAGS.image_size, FLAGS.image_size],
                method=tf.image.ResizeMethod.BILINEAR,
            )
            img = tf.cast(img, tf.float32)
            img.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
            img = tf.image.per_image_standardization(img)
        elif FLAGS.input_modality == "depth":
            img = tf.image.resize_images(
                img,
                size=[FLAGS.image_size, FLAGS.image_size],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            img = tf.cast(img, tf.float32)
            img.set_shape([FLAGS.image_size, FLAGS.image_size, 1])
            img = img - 4000  # depth offset
        elif FLAGS.input_modality == "rgbd":
            rgb = img[..., 0:3]
            depth = img[..., 3:] - 4000

            rgb = tf.image.resize_images(
                rgb,
                size=[FLAGS.image_size, FLAGS.image_size],
                method=tf.image.ResizeMethod.BILINEAR,
            )
            rgb = tf.cast(rgb, tf.float32)
            rgb.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
            rgb = tf.image.per_image_standardization(rgb)

            depth = tf.image.resize_images(
                depth,
                size=[FLAGS.image_size, FLAGS.image_size],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            depth = tf.cast(depth, tf.float32)
            depth.set_shape([FLAGS.image_size, FLAGS.image_size, 1])

            img = tf.concat([rgb, depth], axis=2)

        one_hot = tf.one_hot(label, FLAGS.num_classes, dtype=tf.int64)

        return img, one_hot, sample_name

    def preprocessing_op_sequence(image_op):
        """
        Creates preprocessing operations that are going to be applied on a single frame.

        TODO: Customize for your needs.
        You can do any preprocessing (masking, normalization/scaling of inputs, augmentation, etc.) by using tensorflow operations.
        Built-in image operations: https://www.tensorflow.org/api_docs/python/tf/image
        """
        with tf.name_scope("preprocessing"):
            # crop
            image_op = image_op[8:232, 48:272, :]

            if FLAGS.input_modality == "rgb":
                # rgb standardize
                image_op = image_op[..., 0:3]
                image_op = tf.image.per_image_standardization(image_op)

                # resizing
                image_op = tf.image.resize_images(
                    image_op,
                    size=[FLAGS.image_size, FLAGS.image_size],
                    method=tf.image.ResizeMethod.BILINEAR,
                )
                image_op.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

            elif FLAGS.input_modality == "depth":
                # zero the depth
                image_op = image_op[..., 3:]
                image_op = image_op - 4000

                # resizing
                image_op = tf.image.resize_images(
                    image_op,
                    size=[FLAGS.image_size, FLAGS.image_size],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
                image_op.set_shape([FLAGS.image_size, FLAGS.image_size, 1])

            elif FLAGS.input_modality == "rgbd":
                # rgb standardize
                rgb = image_op[..., 0:3]
                depth = image_op[..., 3:] - 4000
                rgb = tf.image.per_image_standardization(rgb)

                # resizing
                rgb = tf.image.resize_images(
                    rgb,
                    size=[FLAGS.image_size, FLAGS.image_size],
                    method=tf.image.ResizeMethod.BILINEAR,
                )
                depth = tf.image.resize_images(
                    depth,
                    size=[FLAGS.image_size, FLAGS.image_size],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )

                # recombine
                image_op = tf.concat([rgb, depth], axis=2)
                image_op.set_shape([FLAGS.image_size, FLAGS.image_size, 4])

            return image_op

    def read_and_decode_sequence(filename_queue):
        # reader = tf.TFRecordReader()
        # _, serialized_example = reader.read(filename_queue)

        with tf.name_scope("TFRecordDecoding"):
            # parse sequence
            (
                seq_img,
                seq_labels,
                seq_len,
                first_grasp_frame,
                last_grasp_frame,
                sample_name,
            ) = _parse_function_sequence(filename_queue)

            # preprocessing each frame
            seq_img = tf.map_fn(
                lambda x: preprocessing_op_sequence(x),
                elems=seq_img,
                dtype=tf.float32,
                back_prop=False,
            )

            print("seq_img shape: " + str(seq_img.shape))
            print("seq_len shape: " + str(seq_len.shape))
            print("seq_labels shape: " + str(seq_labels.shape))

            return [seq_img, seq_labels, seq_len, sample_name]

    def input_pipeline_sequence(filenames, name="input_pipeline", shuffle=True):
        with tf.name_scope(name):
            # Create a queue of TFRecord input files.
            filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=FLAGS.num_epochs, shuffle=shuffle
            )
            # Read the data from TFRecord files, decode and create a list of data samples by using threads.
            sample_list = [
                read_and_decode_sequence(filename_queue)
                for _ in range(FLAGS.ip_num_read_threads)
            ]
            # Create batches.
            # Since the data consists of variable-length sequences, allow padding by setting dynamic_pad parameter.
            # "batch_join" creates batches of samples and pads the sequences w.r.t the max-length sequence in the batch.
            # Hence, the padded sequence length can be different for different batches.
            batch_rgb, batch_labels, batch_lens, sample_names = tf.train.batch_join(
                sample_list,
                batch_size=FLAGS.batch_size,
                capacity=FLAGS.ip_queue_capacity,
                enqueue_many=False,
                dynamic_pad=True,
                allow_smaller_final_batch=False,
                name="batch_join_and_pad",
            )

            return batch_rgb, batch_labels, batch_lens, sample_names

    # Sanity check FLAGS
    if FLAGS.input_modality not in ["rgb", "rgbd", "depth"]:
        raise (ValueError("input_modality must be one of: rgb, rgbd, depth."))
    else:
        print(FLAGS.input_modality)

    if FLAGS.model_type not in ["single_frames", "sequence"]:
        raise (ValueError("model_type must be one of: single_frames, sequence."))

    if FLAGS.mode not in ["train", "eval", "frozen_train"]:
        raise (ValueError("mode must be one of: train, eval, frozen_train"))

    if FLAGS.resnet_size not in [18, 50]:
        raise (ValueError("resnet size must be one of: 18, 50"))

    if FLAGS.model_type == "sequence":
        # TODO: Set everything up for sequence training
        dataset_dir = FLAGS.dataset_root
        all_filenames = glob.glob(
            os.path.join(dataset_dir, "tfrecords", "*", "*.tfrecord")
        )
        with open(
            os.path.join(
                dataset_dir,
                "validation_split" + str(FLAGS.validation_split_num) + ".pckl",
            ),
            "rb",
        ) as f:
            validation_samples = pickle.load(f)

        validation_tfrecord_filenames = []

        for filename in all_filenames:
            for validation_sample in validation_samples:
                if validation_sample in filename:
                    # filename is a train sample
                    validation_tfrecord_filenames.append(filename)
                    break

        # np.random.shuffle(validation_tfrecord_filenames)
        # validation_tfrecord_filenames = ['/home/luke/datasets/handcam/20180907/204904-grasp_7.tfrecord']
        # validation_tfrecord_filenames = ['/home/luke/datasets/handcam/20180907/214111-grasp_7.tfrecord']

    elif FLAGS.model_type == "single_frames":
        # TODO: Set everything up for single frames
        dataset_dir = os.path.join(FLAGS.dataset_root, "single_frames")
        validation_tfrecord_filenames = glob.glob(
            os.path.join(
                dataset_dir,
                "tfrecords",
                "split" + str(FLAGS.validation_split_num),
                "validation*.tfrecord",
            )
        )
    # ex: /tmp/luke/handcam/split0/sequence/rgbd/frozen_train/2018-08-13
    log_root = model_path

    print("Evaluating %s" % model_path)
    # seq_params = seq_nets_to_eval[seq_name]['config']
    # is_sequence = True
    # max_seq_len = seq_params['seq_len']
    # loss_type = seq_params['loss_type']
    # img_type = seq_params['img_type']
    # current_resnet_model = seq_params['resnet_model']
    # model_dir = seq_params['model_dir']
    # filenames = seq_test_tfrecord_filenames
    # checkpoint_id = tf.train.latest_checkpoint(model_dir)

    rnn_config = {}
    # RNN model parameters
    rnn_config[
        "num_hidden_units"
    ] = FLAGS.rnn_hidden_units  # Number of units in an LSTM cell.
    rnn_config["num_layers"] = FLAGS.rnn_layers  # Number of LSTM stack.
    rnn_config["num_class_labels"] = FLAGS.num_classes
    rnn_config["batch_size"] = FLAGS.batch_size
    rnn_config[
        "loss_type"
    ] = "average"  # or 'last_step' # In the case of 'average', average of all time-steps is used instead of the last time-step.

    tf.reset_default_graph()  # graph needs to be cleared for reuse in loop
    if FLAGS.model_type == "single_frames":
        with tf.variable_scope("preprocessing"):
            filenames_placeholder = tf.placeholder(tf.string, shape=[None])

            dataset = tf.data.TFRecordDataset(
                filenames_placeholder, compression_type="GZIP"
            )
            dataset = dataset.map(_parse_function_single_frame, num_parallel_calls=4)
            dataset = dataset.repeat(1)
            # dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.map(_center_crop_single_frame, num_parallel_calls=4)
            dataset = dataset.map(_preprocessing_op_single_frame, num_parallel_calls=4)
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size)
            )
            dataset = dataset.prefetch(1)

            iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes
            )
            initializer = iterator.make_initializer(dataset)
            (
                test_batch_samples_op,
                test_batch_labels_op,
                sample_names,
            ) = iterator.get_next()
            test_feed_dict = {filenames_placeholder: validation_tfrecord_filenames}
    else:
        with tf.variable_scope("preprocessing"):
            filenames_placeholder = tf.placeholder(tf.string, shape=[None])

            dataset = tf.data.TFRecordDataset(
                filenames_placeholder, compression_type="GZIP"
            )
            dataset = dataset.map(
                lambda x: read_and_decode_sequence(x), num_parallel_calls=4
            )
            dataset = dataset.repeat(1)
            # dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size)
            )
            dataset = dataset.prefetch(1)

            iterator = tf.data.Iterator.from_structure(
                dataset.output_types, dataset.output_shapes
            )
            initializer = iterator.make_initializer(dataset)
            (
                test_batch_samples_op,
                test_batch_labels_op,
                test_batch_seq_len_op,
                sample_names,
            ) = iterator.get_next()

            test_feed_dict = {filenames_placeholder: validation_tfrecord_filenames}

    sess = tf.Session()
    # init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    hps = resnet_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=FLAGS.num_classes,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
    )

    with tf.name_scope("Inference"):
        # Create model
        if FLAGS.model_type == "single_frames":
            inferModel = resnet_model.ResNet(
                hps,
                test_batch_samples_op,
                test_batch_labels_op,
                "inference",
                batch_size=FLAGS.batch_size,
                model_type=FLAGS.model_type,
                input_modality=FLAGS.input_modality,
                resnet_size=FLAGS.resnet_size,
                imu_data=None,
            )
            inferModel.build_graph()
        else:
            cnnModel = resnet_model.ResNet(
                hps,
                test_batch_samples_op,
                test_batch_labels_op,
                "inference",
                batch_size=FLAGS.batch_size,
                model_type=FLAGS.model_type,
                input_modality=FLAGS.input_modality,
                resnet_size=FLAGS.resnet_size,
                imu_data=None,
            )
            cnnModel.build_graph()
            cnn_representations = cnnModel.cnn_representations

            inferModel = lstm_model(
                config=rnn_config,
                input_op=cnn_representations,
                labels_op=test_batch_labels_op,
                seq_len_op=test_batch_seq_len_op,
                mode="inference",
            )

            inferModel.build_graph()
            print("\n# of parameters: %s" % inferModel.num_parameters)

    # Restore computation graph.
    saver = tf.train.Saver()
    if checkpoint_id is None:
        print("No checkpoint ID specified, using latest checkpoint.")
        if True or "/media/luke/hdd-3tb" in model_path:
            # checkpoints were moved, need to get latest checkpoint manually.
            checkpoint_id = get_latest_checkpoint_id(model_path)
        else:
            checkpoint_id = tf.train.latest_checkpoint(model_path)
    if type(checkpoint_id) is not str:
        checkpoint_id = str(checkpoint_id)
    if "model-" not in checkpoint_id:
        checkpoint_id = "model-" + checkpoint_id
    checkpoint_path = os.path.join(model_path, checkpoint_id)
    print("Checkpoint path: " + checkpoint_path)
    saver.restore(sess, checkpoint_path)

    ######
    # Eval loop
    ######
    step = 0
    test_predictions = []
    test_pred_probs = []
    test_correct_labels = []
    test_sample_names = []
    test_accuracy = 0
    batch_counter = 0

    # init
    sess.run(initializer, feed_dict=test_feed_dict)

    try:
        while not coord.should_stop():
            # Get predicted labels and sample ids for submission csv.
            [pred_probs, predictions, sample_ids, out_sample_name, acc] = sess.run(
                [
                    inferModel.logits,
                    inferModel.predictions,
                    test_batch_labels_op,
                    sample_names,
                    inferModel.batch_accuracy,
                ],
                feed_dict={},
            )
            test_predictions.extend(predictions)
            test_pred_probs.extend(pred_probs)
            # print(sample_ids)
            test_correct_labels.extend(sample_ids)
            test_accuracy += acc
            batch_counter += 1
            # print(sample_ids.shape)
            for name in out_sample_name:
                test_sample_names.append(str(name.flatten(), "ascii"))
                # print(str(name.flatten(), 'ascii'))

    except tf.errors.OutOfRangeError:
        print("Done.")
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    test_accuracy = test_accuracy / batch_counter

    out_dict = {
        "preds": test_predictions,
        "pred_probs": test_pred_probs,
        "sample_names": test_sample_names,
        "labels": test_correct_labels,
        "val_accuracy": test_accuracy,
    }

    spotting_accuracy = handcam_gesture_spotting_acc(out_dict)
    out_dict["gesture_spotting_accuracy"] = spotting_accuracy

    with open(os.path.join(model_path, "results.pckl"), "wb") as f:
        pickle.dump(out_dict, f)

    # print("per frame: %.2f\tgesture spotting: %.2f" % (100 * out_dict['val_accuracy'], 100 * spotting_accuracy))

    dict_resnet_type = "resnet-%d" % FLAGS.resnet_size
    dict_model_type = FLAGS.model_type
    if FLAGS.model_type != "single_frames":
        dict_model_type = (
            "sequence_frozen" if FLAGS.mode == "frozen_train" else "sequence_end2end"
        )

    dict_split_name = "split%d" % FLAGS.validation_split_num
    dict_modality = FLAGS.input_modality

    compiled_results_dict[dict_resnet_type][dict_model_type][dict_split_name][
        dict_modality
    ] = {
        "accuracy": out_dict["val_accuracy"],
        "gesture_spotting": spotting_accuracy,
        "full_results_dict": out_dict,
    }

    sess.close()

with open("/local/home/luke/all_validations.pckl", "wb") as f:
    pickle.dump(compiled_results_dict, f)
