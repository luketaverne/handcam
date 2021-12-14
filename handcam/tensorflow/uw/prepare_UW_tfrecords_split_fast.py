# Import SlackerGPU to set env variables, import tf (tf 1.4)
import data_aug_uw

from handcam.ltt.util import SlackerGPU

slackerGPU = SlackerGPU.SlackerGPU(
    username="ltaverne", desired_server="ait-server-03", num_gpus=0
)
import tensorflow as tf

flags = tf.app.flags

# State your dataset directory
flags.DEFINE_string(
    "dataset_dir",
    "/local/home/luke/datasets/rgbd-dataset/",
    "String: Your dataset directory",
)

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float(
    "validation_size",
    0.1,
    "Float: The proportion of examples in the dataset to be used for validation",
)

# The number of shards per dataset split.
flags.DEFINE_integer("num_records_per_shard", 100, "Int: Number of images per shard")
# Seed for repeatability.
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability.")

# Output filename for the naming the TFRecord file
flags.DEFINE_string(
    "tfrecord_filename",
    "uw-rgbd",
    "String: The output filename to name your TFRecord file",
)


FLAGS = flags.FLAGS

# other imports
import random
from random import shuffle

random.seed(FLAGS.random_seed)
import glob
import os
import math
import sys
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(FLAGS.random_seed)
from scipy.ndimage import rotate
from handcam.ltt.util.Utils import apply_alpha_matting_tf

from handcam.ltt.datasets.handcam import HandCamDataHandler

data_handler_handcam = HandCamDataHandler.Handler()
from handcam.ltt.datasets.handcam.HandCamDataHandler import HandGenerator

hand_generator = HandGenerator(f=data_handler_handcam.f, batch_size=1)

# Get an iterator for all of the images
shuffle_data = True
dataset_root = FLAGS.dataset_dir
depth_image_pattern = dataset_root + "*/*/*_depth.png"
hand_mask_pattern = "/local/home/luke/datasets/handcam/greenscreens/*-mask.png"
class_names = sorted([i for i in next(os.walk(dataset_root))[1]])
class_names_to_index = dict(zip(class_names, range(len(class_names))))
object_instances = glob.glob(os.path.join(dataset_root, "*/*/"))
#'/local/home/luke/datasets/rgbd-dataset/hand_towel/hand_towel_2/hand_towel_2_4_184_depth.png'
depth_filenames = glob.glob(depth_image_pattern)
hand_mask_filenames = glob.glob(hand_mask_pattern)


def get_hand_root_from_mask_path(hand_mask_path):
    return hand_mask_path.split("-mask.png")[0]


def get_class_name_from_path(depth_im_path):
    return depth_im_path.split(dataset_root)[1].split("/")[0]


def get_rgb_path(depth_im_path):
    return depth_im_path.split("_depth.png")[0] + ".png"


def shuffle_dataset(rgb_filenames, depth_filenames, labels):
    c = list(zip(rgb_filenames, depth_filenames, labels))
    shuffle(c)

    return zip(*c)  # rgb_filenames, depth_filenames, labels


hand_root_filenames = [get_hand_root_from_mask_path(i) for i in hand_mask_filenames]
num_hands = len(hand_root_filenames)

NUM_CLASSES = 51


def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
    values: A scalar or list of values.
    Returns:
    a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
    values: A string.
    Returns:
    a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _parse_function(rgb_filenames, depth_filenames, ids, i):
    depthFilename = depth_filenames[i]
    rgbFilename = rgb_filenames[i]
    hand_index = np.random.random_integers(0, num_hands - 1)  # closed interval
    handFilename = hand_root_filenames[hand_index]
    rotAngle = (
        -10.0 - 95.0
    ) * np.random.rand() + 95.0  # x shift  # degrees, pos is CCW
    h_shift_im = np.random.randint(-45, 45)  # x shift
    v_shift_im = np.random.randint(0, 90)  # y shift
    h_shift_hand = np.random.randint(0, 200)  # x shift
    v_shift_hand = np.random.randint(-100, 0)  # x shift
    flip_lr = round(np.random.rand())

    aug_im = data_aug_uw.load_and_augment(
        depthFilename,
        rgbFilename,
        handFilename,
        rotAngle,
        h_shift_im,
        v_shift_im,
        h_shift_hand,
        v_shift_hand,
        flip_lr,
    )

    # Have to scale, can't do it in opencv as CV_32F has a range of 0.0-1.0
    # print(aug_im.dtype)
    # print(np.max(aug_im))
    aug_im[..., 0:3] = np.float32(np.uint8(255.0 * aug_im[..., 0:3]))
    aug_im[..., 3] = np.float32(np.uint16(65535.0 * aug_im[..., 3]))

    img_str = tf.compat.as_bytes(aug_im.tostring())

    return img_str, ids[i]


# Borrowing work from <https://github.com/kwotsin/create_tfrecords/blob/python-3.0/dataset_utils.py>


def _get_dataset_filename(dataset_dir, shard_id, instance_path, _NUM_SHARDS):
    instance_path = instance_path.split("/")[-2]
    output_filename = "%s_%05d-of-%05d.tfrecord" % (
        instance_path,
        shard_id,
        _NUM_SHARDS,
    )
    return os.path.join(
        "/local/home/luke/datasets/rgbd-dataset-tfrecords", output_filename
    )


def image_to_tfexample(rgb_filenames, depth_filenames, ids, i):
    img, class_id = _parse_function(rgb_filenames, depth_filenames, ids, i)

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/img": bytes_feature(img),
                "image/class/label": int64_feature(class_id),
            }
        )
    )


def _convert_dataset(object_instances, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.
    Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
    """
    # assert split_name in ['train', 'validation']

    with tf.Graph().as_default():
        with tf.Session("") as sess:
            for instance in object_instances:
                depth_filenames = glob.glob(os.path.join(instance, "*_depth.png"))
                ids = [
                    class_names_to_index[get_class_name_from_path(i)]
                    for i in depth_filenames
                ]
                rgb_filenames = [get_rgb_path(i) for i in depth_filenames]

                rgb_filenames, depth_filenames, ids = shuffle_dataset(
                    rgb_filenames, depth_filenames, ids
                )

                total_shards = int(
                    np.ceil(len(depth_filenames) / float(FLAGS.num_records_per_shard))
                )

                for shard_id in range(total_shards):
                    output_filename = _get_dataset_filename(
                        dataset_dir,
                        shard_id,
                        instance_path=instance,
                        _NUM_SHARDS=total_shards,
                    )

                    with tf.python_io.TFRecordWriter(
                        output_filename
                    ) as tfrecord_writer:
                        start_ndx = shard_id * FLAGS.num_records_per_shard
                        end_ndx = min(
                            (shard_id + 1) * FLAGS.num_records_per_shard,
                            len(rgb_filenames),
                        )
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write(
                                "\r>> Converting image %d/%d shard %d"
                                % (i + 1, len(rgb_filenames), shard_id)
                            )
                            sys.stdout.flush()

                            example = image_to_tfexample(
                                rgb_filenames, depth_filenames, ids, i
                            )
                            tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write("\n")
    sys.stdout.flush()


_convert_dataset(object_instances, dataset_dir=FLAGS.dataset_dir)
