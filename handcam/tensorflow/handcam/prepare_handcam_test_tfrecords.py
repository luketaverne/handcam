import random
import sys
import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import scipy.misc
import matplotlib.pyplot as plt  # noqa: ignore=E402
from handcam.ltt.datasets.handcam.OniProcessingCpp import OniSampleReader
import glob

# import cPickle as pickle
import pickle

# from primesense.utils import OpenNIError

flags = tf.app.flags

# State your dataset directory
flags.DEFINE_string(
    "dataset_dir",
    "/local/home/luke/datasets/handcam/",
    "String: Your dataset directory",
)
flags.DEFINE_bool(
    "redo_tfrecords",
    False,
    "Bool: True to remake all samples, False to only process new ones.",
)
# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float(
    "validation_size",
    0.1,
    "Float: The proportion of examples in the dataset to be used for validation",
)
flags.DEFINE_bool(
    "shuffle", True, "Bool: True for shuffle samples, False to leave as arbitrary list"
)
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability.")
flags.DEFINE_integer("samples_per_shard", 1, "Number of samples per shard")
# flags.DEFINE_integer('start_shard', 0, 'Int: Start number of shards to split the TFRecord files')
# flags.DEFINE_integer('end_shard', 1000, 'Int: Start number of shards to split the TFRecord files')

# Output filename for the naming the TFRecord file
flags.DEFINE_string(
    "tfrecord_filename",
    "handcam",
    "String: The output filename to name your TFRecord file",
)

FLAGS = flags.FLAGS

random.seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)

# processed_samples = []
#
# new_samples = [i for i in all_samples if i not in processed_samples] # might be slow after finding lots of samples

# tf_records = glob.glob(FLAGS.dataset_dir + "tfrecords/*.tfrecord")
# for i in range(len(tf_records)):
#     # Extract the id: 20180312/154902_grasp0
#     tf_records[i] = int(tf_records[i].split("_")[-1].split(".tfrecord")[0])


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


def _bytes_feature_list(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
    e.g, sentence in list of ints
    """
    return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def _parse_function(sample_name, test_type):
    oni = OniSampleReader(
        os.path.join(FLAGS.dataset_dir, "test_set", test_type, sample_name)
    )
    vid_arr = oni.vid
    # ids = [tf.compat.as_bytes(idd.tostring()) for idd in oni.frame_labels]
    ids = oni.frame_labels
    first_grasp_frame = 0
    last_grasp_frame = ids.shape[0]
    grasp_frames = np.where(ids != 6)
    try:
        first_grasp_frame = np.min(grasp_frames)
        last_grasp_frame = np.max(grasp_frames)
    except ValueError:
        # Must be a nonsense grasp
        pass
    # print(len(ids))
    vid_length = vid_arr.shape[0]
    # accel = oni.accel
    # gyro = oni.gyro
    # pose = oni.pose

    vid_str = [tf.compat.as_bytes(frame.tostring()) for frame in vid_arr]

    return vid_str, ids, vid_length, first_grasp_frame, last_grasp_frame


def _get_dataset_filename(sample_name, test_type):
    output_filename = "%s.tfrecord" % (sample_name)
    return os.path.join(
        FLAGS.dataset_dir, "test_set", "tfrecords", test_type, output_filename
    )


def _convert_dataset(samples, test_type):
    i = 0
    for sample in samples:
        # readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        with tf.Graph().as_default():
            with tf.Session("") as sess:
                # for shard_id in range(start_tfrecord_id, num_shards_needed):
                output_filename = _get_dataset_filename(sample, test_type)
                # print(output_filename)

                # with tf.python_io.TFRecordWriter(output_filename, options=readerOptions) as tfrecord_writer:
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    # start_ndx = (shard_id - start_tfrecord_id) * FLAGS.samples_per_shard
                    # end_ndx = min((shard_id + 1 - start_tfrecord_id) * FLAGS.samples_per_shard, len(sample_names))
                    # for i in range(start_ndx, end_ndx):
                    sys.stdout.write(
                        "\r>> Converting image %d/%d" % (i + 1, len(samples))
                    )
                    sys.stdout.flush()

                    example = image_to_tfexample(sample, test_type)
                    tfrecord_writer.write(example.SerializeToString())
                    i += 1

    sys.stdout.write("\n")
    sys.stdout.flush()


def image_to_tfexample(sample_name, test_type):
    img, class_id, vid_length, first_grasp_frame, last_grasp_frame = _parse_function(
        sample_name, test_type
    )

    return tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                "vid_length": int64_feature(vid_length),
                "first_grasp_frame": int64_feature(first_grasp_frame),
                "last_grasp_frame": int64_feature(last_grasp_frame),
                "sample_name": bytes_feature(tf.compat.as_bytes(sample_name)),
            }
        ),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                "vid": _bytes_feature_list(img),
                "frame_labels": _int64_feature_list(class_id),
            }
        ),
    )


test_types = [
    "newloc_newobj_single",
    "newloc_newobj_clutter",
    "newloc_oldobj_clutter",
    "newloc_oldobj_single",
    "oldloc_newobj_single",
    "oldloc_newobj_clutter",
    "oldloc_oldobj_clutter",
]

print(FLAGS.dataset_dir + "test_set/" + test_types[0] + "/*/*/")

for test_type in test_types:

    next_samples = glob.glob(FLAGS.dataset_dir + "test_set/" + test_type + "/*/*/")

    for i in range(len(next_samples)):
        # Extract the id: 20180312/154902_grasp0
        next_samples[i] = next_samples[i].split(
            FLAGS.dataset_dir + "test_set/" + test_type + "/"
        )[-1][:-1]

    print(len(next_samples))
    _convert_dataset(next_samples, test_type)
