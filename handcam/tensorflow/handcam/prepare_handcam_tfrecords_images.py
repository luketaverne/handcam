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
from ltt.datasets.handcam.OniProcessingCpp import OniSampleReader
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
flags.DEFINE_integer("samples_per_shard", 400, "Number of samples per shard")
flags.DEFINE_integer("images_per_sequence", 20, "Number of samples per shard")
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

for split_id in range(2, 10):  # regenerate 0 with gzip compression
    print("Working on split %d" % split_id)
    with open(os.path.join(FLAGS.dataset_dir, "lit%d.pckl" % split_id), "rb") as f:
        train_filenames = pickle.load(f)

    with open(
        os.path.join(FLAGS.dataset_dir, "validation_split%d.pckl" % split_id), "rb"
    ) as f:
        validation_filenames = pickle.load(f)

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

    # sample breakdowns
    # train_samples_per_grasp = {'grasp_1':[],'grasp_2':[],'grasp_3':[],'grasp_4':[],'grasp_5':[],'grasp_6':[],'grasp_7':[]}
    # validation_samples_per_grasp = {'grasp_1':[],'grasp_2':[],'grasp_3':[],'grasp_4':[],'grasp_5':[],'grasp_6':[],'grasp_7':[]}
    # train_samples = []
    # validation_samples = []

    if FLAGS.shuffle:
        random.shuffle(train_filenames)
        random.shuffle(train_filenames)
        random.shuffle(validation_filenames)
        random.shuffle(validation_filenames)

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
        # print(sample_name)
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

        replace = False

        if last_grasp_frame - first_grasp_frame < FLAGS.images_per_sequence:
            replace = True

        selected_frames = np.random.choice(
            np.arange(first_grasp_frame, last_grasp_frame),
            FLAGS.images_per_sequence,
            replace=replace,
        )
        vid_arr = vid_arr[selected_frames]
        # vid_str = [tf.compat.as_bytes(frame.tostring()) for frame in vid_arr]

        return vid_arr, [oni.grasp_id] * len(vid_arr), selected_frames

    def _get_dataset_filename(shard_id, shard_type, val_split_num):
        output_filename = "%s/%s_%05d.tfrecord" % (val_split_num, shard_type, shard_id)
        return os.path.join(
            FLAGS.dataset_dir, "single_frames", "tfrecords", output_filename
        )

    def _convert_dataset(sample_names, shard_type, val_split_num):
        readerOptions = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP
        )
        # print("Working on: " + grasp)
        num_shards_needed = int(
            float(FLAGS.images_per_sequence)
            * np.ceil(len(sample_names) / float(FLAGS.samples_per_shard))
        )
        # num_shards_needed_validation = int(np.ceil(len(validation_samples) / float(FLAGS.samples_per_shard)))
        # readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        with tf.Graph().as_default():
            with tf.Session("") as sess:
                for shard_id in range(0, num_shards_needed):
                    output_filename = _get_dataset_filename(
                        shard_id, shard_type, val_split_num
                    )

                    # with tf.python_io.TFRecordWriter(output_filename, options=readerOptions) as tfrecord_writer:
                    with tf.python_io.TFRecordWriter(
                        output_filename, options=readerOptions
                    ) as tfrecord_writer:
                        start_ndx = np.uint(
                            np.floor(
                                (shard_id)
                                * FLAGS.samples_per_shard
                                / FLAGS.images_per_sequence
                            )
                        )
                        end_ndx = min(
                            np.uint(
                                np.floor(
                                    (shard_id + 1)
                                    * FLAGS.samples_per_shard
                                    / FLAGS.images_per_sequence
                                )
                            ),
                            len(sample_names),
                        )
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write(
                                "\r>> Converting image %d/%d shard %d"
                                % (i + 1, len(sample_names), shard_id)
                            )
                            sys.stdout.flush()

                            examples = image_to_tfexamples(sample_names[i])
                            for example in examples:
                                tfrecord_writer.write(example.SerializeToString())

                            # processed_samples.append(sample_names[i])
                            # with open(FLAGS.dataset_dir + "processed_samples.pckl", "wb") as f:
                            #     pickle.dump(processed_samples, f)

        sys.stdout.write("\n")
        sys.stdout.flush()

    def image_to_tfexamples(sample_name):
        imgs, class_ids, frame_nums = _parse_function(sample_name)

        examples = []

        for i in range(len(imgs)):
            examples.append(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image/img": bytes_feature(
                                tf.compat.as_bytes(imgs[i].tostring())
                            ),
                            "image/class/label": int64_feature(class_ids[i]),
                            "sample_name": bytes_feature(
                                tf.compat.as_bytes(sample_name)
                            ),
                            "image/frame_num": int64_feature(frame_nums[i]),
                        }
                    )
                )
            )

        return examples

    if split_id not in [2, 3]:
        _convert_dataset(train_filenames, "train", "split%d" % split_id)
    _convert_dataset(validation_filenames, "validation", "split%d" % split_id)
