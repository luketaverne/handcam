# Import SlackerGPU to set env variables, import tf (tf 1.4)
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
flags.DEFINE_integer(
    "num_shards", 1000, "Int: Number of shards to split the TFRecord files"
)

flags.DEFINE_integer(
    "start_shard", 0, "Int: Start number of shards to split the TFRecord files"
)
flags.DEFINE_integer(
    "end_shard", 1000, "Int: Start number of shards to split the TFRecord files"
)


flags.DEFINE_bool("make_train", True, "Bool: True for train, False for eval")

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
class_names = sorted([i for i in next(os.walk(dataset_root))[1]])
class_names_to_index = dict(zip(class_names, range(len(class_names))))
#'/local/home/luke/datasets/rgbd-dataset/hand_towel/hand_towel_2/hand_towel_2_4_184_depth.png'
# depth_iterator = glob.iglob(depth_image_pattern)
depth_filenames = glob.glob(depth_image_pattern)
# print(class_names_to_index)
# print(len(depth_filenames))


def get_class_name_from_path(depth_im_path):
    return depth_im_path.split(dataset_root)[1].split("/")[0]


def get_rgb_path(depth_im_path):
    return depth_im_path.split("_depth.png")[0] + ".png"


def shuffle_dataset(rgb_filenames, depth_filenames, labels):
    c = list(zip(rgb_filenames, depth_filenames, labels))
    shuffle(c)

    return zip(*c)  # rgb_filenames, depth_filenames, labels


# Get label for every filename
ids = [class_names_to_index[get_class_name_from_path(i)] for i in depth_filenames]
# Get corresponding rgb filename for each depth filename
rgb_filenames = [get_rgb_path(i) for i in depth_filenames]

if shuffle_data:
    rgb_filenames, depth_filenames, ids = shuffle_dataset(
        rgb_filenames, depth_filenames, ids
    )

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


def _crop(img, label):
    # im = tf.concat([tf.cast(rgb,dtype=tf.float32), tf.cast(depth,dtype=tf.float32)], axis=2)

    # im = tf.transpose(im,[2,0,1])
    img = tf.image.resize_image_with_crop_or_pad(img, 224, 224)

    return img, label


def _merge_rgb_depth(rgb, depth, label):
    im = tf.concat(
        [tf.cast(rgb, dtype=tf.float32), tf.cast(depth, dtype=tf.float32)], axis=2
    )

    # im = tf.transpose(im,[2,0,1])
    # im = tf.image.resize_image_with_crop_or_pad(im, 224, 224)

    return im, label


def _parse_function(rgb_img, depth_img, label, sess):
    # Decode images
    rgb_img = tf.image.decode_image(rgb_img, channels=3)
    depth_img = tf.image.decode_image(depth_img, channels=1)

    rgb_img.set_shape([480, 640, 3])
    depth_img.set_shape([480, 640, 1])

    # depth_img = tf.expand_dims(depth_img, axis=2)

    # Flip
    rgb_img, depth_img, label = _random_flip_lr(rgb_img, depth_img, label)

    # Translate
    rgb_img, depth_img, label = _random_translate(rgb_img, depth_img, label)

    # Rotate
    rgb_img, depth_img, label = _random_rotations(rgb_img, depth_img, label)

    # Hand overlay
    rgb_img, depth_img, label = _hand_overlay(rgb_img, depth_img, label)

    # merge
    img, label = _merge_rgb_depth(rgb_img, depth_img, label)

    # crop
    img, label = _crop(img, label)

    # Encode again
    img.set_shape((224, 224, 4))
    img = tf.expand_dims(img, axis=0)
    # out_im = tf.compat.as_bytes(img.eval()[0].tostring())

    img_arr = sess.run(img)

    img_arr = np.asarray(img_arr, dtype=np.uint16)

    img_str = tf.compat.as_bytes(img_arr[0].tostring())

    return img_str, label


def _random_rotations(rgb, depth, label):
    rotate = np.random.rand()

    if rotate > 0.5:
        angle = (-0.174 - 1.66) * np.random.rand() + 1.66  # Between -95 and 10 degrees
        rgb = tf.contrib.image.rotate(rgb, angle)
        depth = tf.contrib.image.rotate(depth, angle)

    return rgb, depth, label


def _random_flip_lr(rgb, depth, label):
    flip = np.random.rand()

    if flip > 0.5:
        rgb = tf.image.flip_left_right(rgb)
        depth = tf.image.flip_left_right(depth)

    return rgb, depth, label


def _random_translate(rgb, depth, label):
    translate = np.random.rand()

    if translate > 0.5:
        transforms = [1, 0, 0, 0, 1, 0, 0, 0]

        transforms[2] = np.random.randint(-45, 45)  # x shift
        transforms[5] = np.random.randint(-45, 45)  # y shift

        rgb = tf.contrib.image.transform(rgb, transforms)
        depth = tf.contrib.image.transform(depth, transforms)

    return rgb, depth, label


def _hand_overlay(rgb, depth, label):
    # Get a random hand and mask
    next_hand = next(hand_generator)[0]  # index to remove batch size of 1
    hand, mask = next_hand[..., 0:3], np.expand_dims(next_hand[..., 3], axis=2)
    hand = hand[..., ::-1]

    rgb = apply_alpha_matting_tf(hand, rgb, mask)
    depth = apply_alpha_matting_tf(tf.zeros(shape=depth.shape), depth, mask)

    return rgb, depth, label


def input_parser(rgb_img_path, depth_img_path):
    # convert label to one-hot
    #     one_hot = tf.one_hot(label, NUM_CLASSES)

    # Read the images from file
    rgb_img = tf.read_file(rgb_img_path)
    depth_img = tf.read_file(depth_img_path)

    return rgb_img, depth_img


# Borrowing work from <https://github.com/kwotsin/create_tfrecords/blob/python-3.0/dataset_utils.py>


def _get_dataset_filename(
    dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS
):
    output_filename = "%s_%s_%05d-of-%05d.tfrecord" % (
        tfrecord_filename,
        split_name,
        shard_id,
        _NUM_SHARDS,
    )
    return os.path.join(dataset_dir, output_filename)


def image_to_tfexample(rgb_img, depth_img, class_id, sess):
    img, class_id = _parse_function(rgb_img, depth_img, class_id, sess)

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/img": bytes_feature(img),
                "image/class/label": int64_feature(class_id),
            }
        )
    )


def _convert_dataset(
    split_name,
    rgb_filenames,
    depth_filenames,
    ids,
    dataset_dir,
    tfrecord_filename,
    _NUM_SHARDS,
):
    """Converts the given filenames to a TFRecord dataset.
    Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ["train", "validation"]

    num_per_shard = int(math.ceil(len(rgb_filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        with tf.Session("") as sess:
            for shard_id in range(FLAGS.start_shard, FLAGS.end_shard):
                output_filename = _get_dataset_filename(
                    dataset_dir,
                    split_name,
                    shard_id,
                    tfrecord_filename=FLAGS.tfrecord_filename,
                    _NUM_SHARDS=_NUM_SHARDS,
                )

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(rgb_filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write(
                            "\r>> Converting image %d/%d shard %d"
                            % (i + 1, len(rgb_filenames), shard_id)
                        )
                        sys.stdout.flush()

                        # Read the filename:
                        rgb_img = tf.gfile.FastGFile(rgb_filenames[i], "rb").read()
                        depth_img = tf.gfile.FastGFile(depth_filenames[i], "rb").read()

                        example = image_to_tfexample(rgb_img, depth_img, ids[i], sess)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write("\n")
    sys.stdout.flush()


# Do the creation
num_validation = int(FLAGS.validation_size * len(rgb_filenames))

rgb_training_filenames = rgb_filenames[num_validation:]
rgb_validation_filenames = rgb_filenames[:num_validation]
depth_training_filenames = depth_filenames[num_validation:]
depth_validation_filenames = depth_filenames[:num_validation]
training_ids = ids[num_validation:]
validation_ids = ids[:num_validation]

training_shards = int(FLAGS.num_shards * (1 - FLAGS.validation_size))
validation_shards = int(FLAGS.num_shards * FLAGS.validation_size)

if FLAGS.make_train:
    _convert_dataset(
        "train",
        rgb_training_filenames,
        depth_training_filenames,
        training_ids,
        dataset_dir=FLAGS.dataset_dir,
        tfrecord_filename=FLAGS.tfrecord_filename,
        _NUM_SHARDS=training_shards,
    )
else:
    _convert_dataset(
        "validation",
        rgb_validation_filenames,
        depth_validation_filenames,
        validation_ids,
        dataset_dir=FLAGS.dataset_dir,
        tfrecord_filename=FLAGS.tfrecord_filename,
        _NUM_SHARDS=validation_shards,
    )
