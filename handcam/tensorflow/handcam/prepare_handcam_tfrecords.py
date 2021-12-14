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

# all_samples = glob.glob(FLAGS.dataset_dir + "samples/*/*")
all_samples = glob.glob(FLAGS.dataset_dir + "20180907/214111*")

if FLAGS.shuffle:
    random.shuffle(all_samples)


for i in range(len(all_samples)):
    # Extract the id: 20180312/154902_grasp0
    all_samples[i] = all_samples[i].split(FLAGS.dataset_dir + "samples/")[-1]

# processed_samples = []
#
# new_samples = [i for i in all_samples if i not in processed_samples] # might be slow after finding lots of samples

# tf_records = glob.glob(FLAGS.dataset_dir + "tfrecords/*.tfrecord")
# for i in range(len(tf_records)):
#     # Extract the id: 20180312/154902_grasp0
#     tf_records[i] = int(tf_records[i].split("_")[-1].split(".tfrecord")[0])


print("Found %d total samples" % len(all_samples))

# Get splits for validation
samples_per_grasp = {
    "grasp_1": [],
    "grasp_2": [],
    "grasp_3": [],
    "grasp_4": [],
    "grasp_5": [],
    "grasp_6": [],
    "grasp_7": [],
}
grasp_labels = [
    "grasp_1",
    "grasp_2",
    "grasp_3",
    "grasp_4",
    "grasp_5",
    "grasp_6",
    "grasp_7",
]
grasp_label_to_id = {
    "grasp_1": 0,
    "grasp_2": 1,
    "grasp_3": 2,
    "grasp_4": 3,
    "grasp_5": 4,
    "grasp_6": 5,
    "grasp_7": 6,
}


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


def _parse_function(sample_name):
    oni = OniSampleReader(os.path.join(FLAGS.dataset_dir, "samples", sample_name))
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
    accel = naive_imu_align(oni.accel, vid_length)
    gyro = naive_imu_align(oni.gyro, vid_length)
    try:
        if len(accel) != len(gyro):
            print("\n Difference: {0}".format(len(gyro) - len(accel)))
            raise (
                ValueError(
                    "accel and gyro are different lengths ({0}, {1}). Sample {2}".format(
                        len(accel), len(gyro), sample_name
                    )
                )
            )
        if len(gyro) != vid_length:
            print("\n Difference: {0}".format(len(gyro) - vid_length))
            raise (
                ValueError(
                    "vid_length and gyro are different lengths ({0}, {1}). Sample {2}".format(
                        vid_length, len(gyro), sample_name
                    )
                )
            )
    except ValueError as e:
        print(e)

    accel = [tf.compat.as_bytes(sample.tostring()) for sample in accel]
    gyro = [tf.compat.as_bytes(sample.tostring()) for sample in gyro]
    pose = [tf.compat.as_bytes(sample.tostring()) for sample in oni.pose]
    vid_str = [tf.compat.as_bytes(frame.tostring()) for frame in vid_arr]

    return (
        vid_str,
        ids,
        vid_length,
        first_grasp_frame,
        last_grasp_frame,
        accel,
        gyro,
        pose,
    )


def _get_dataset_filename(sample_name):
    output_filename = "%s.tfrecord" % (sample_name)
    return os.path.join(FLAGS.dataset_dir, output_filename)


def naive_imu_align(imu_data, seq_len):
    # assuming a framerate of 30fps until we reach seq_len samples
    output_data = []
    initial_timestamp = int(imu_data[0][0])
    prev_imu_timestamp = int(imu_data[0][0])
    next_imu_index_start_point = 0

    cleaned_imu_data = []
    cleaned_imu_data.append(imu_data[0])

    for imu_index in range(1, len(imu_data)):
        current_imu_timestamp = int(imu_data[imu_index][0])
        if current_imu_timestamp == prev_imu_timestamp:
            # timestamp is the same, skip it.
            continue
        else:
            # new data, add to cleaned list.
            cleaned_imu_data.append(imu_data[imu_index])
            prev_imu_timestamp = current_imu_timestamp

    for frame_num in range(seq_len):
        best_time_difference = None
        best_index = 0
        for imu_index in range(next_imu_index_start_point, len(cleaned_imu_data)):
            current_imu_timestamp = int(
                cleaned_imu_data[imu_index][0] - initial_timestamp
            )  # milliseconds?
            # Next timestamp, compare to the target value
            target_timestamp = int(frame_num * 1000 * 1.0 / 30.0)  # in milliseconds
            current_time_diff = np.abs(target_timestamp - current_imu_timestamp)
            if best_time_difference is None:
                best_time_difference = current_time_diff
                best_index = imu_index
            elif current_time_diff < best_time_difference:
                best_time_difference = current_time_diff
                best_index = imu_index
            else:
                # we've stopped improving, save the current data and go on to the next frame
                next_imu_index_start_point = best_index
                output_data.append(cleaned_imu_data[best_index][1:])
                break

    return output_data


def _convert_dataset(samples):
    i = 0
    for sample in samples:
        readerOptions = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP
        )
        with tf.Graph().as_default():
            with tf.Session("") as sess:
                # for shard_id in range(start_tfrecord_id, num_shards_needed):
                output_filename = _get_dataset_filename(sample)

                sys.stdout.write("\r>> Converting image %d/%d" % (i + 1, len(samples)))
                sys.stdout.flush()

                # check if file aready exists
                if os.path.isfile(output_filename):
                    i += 1
                    continue

                # make sure the output directory exists.
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)

                with tf.python_io.TFRecordWriter(
                    output_filename, options=readerOptions
                ) as tfrecord_writer:
                    # with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    # start_ndx = (shard_id - start_tfrecord_id) * FLAGS.samples_per_shard
                    # end_ndx = min((shard_id + 1 - start_tfrecord_id) * FLAGS.samples_per_shard, len(sample_names))
                    # for i in range(start_ndx, end_ndx):

                    example = image_to_tfexample(sample)
                    tfrecord_writer.write(example.SerializeToString())
                    i += 1

    sys.stdout.write("\n")
    sys.stdout.flush()


def image_to_tfexample(sample_name):
    (
        img,
        class_id,
        vid_length,
        first_grasp_frame,
        last_grasp_frame,
        accel,
        gyro,
        pose,
    ) = _parse_function(sample_name)
    # TODO: Add IMU sequences. Decide if I should save all IMU data to tfrecord, or pre-align it and save only the data
    # TODO: corresponding to the frames. Look at the plot generation file for the alignment code, move to ltt.Util.
    # TODO: If we have them all, then I could try to use PhasedLSTM.

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
                "accel": _bytes_feature_list(accel),
                "gyro": _bytes_feature_list(gyro),
                "pose": _bytes_feature_list(pose),
                "frame_labels": _int64_feature_list(class_id),
            }
        ),
    )


_convert_dataset(all_samples)
