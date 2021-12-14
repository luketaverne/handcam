import errno
import subprocess

import tensorflow as tf
import random
from random import shuffle
import glob
import sys
import numpy as np
import six
import os
import pickle
import datetime
from handcam.ltt.network.model.Wide_ResNet import wide_resnet_tf as resnet_model
from handcam.ltt.network.model.RNNModel import LSTMModel as lstm_model

# from handcam.ltt.network.model.CNNModel import CNN_rgb as cnn_model
from handcam.ltt.util.TFTools import plot_confusion_matrix

flags = tf.app.flags
today = (
    datetime.datetime.today().strftime("%Y-%m-%d")
    + "/"
    + datetime.datetime.today().strftime("%H:%M")
)

# Things that might change often
flags.DEFINE_string("input_modality", "rgbd", "rgb, rgbd or depth")
flags.DEFINE_string("model_type", "single_frames", "sequence or single_frames")
flags.DEFINE_integer("resnet_size", 18, "Int: size of resnet, either 18 or 50")
flags.DEFINE_float("learning_rate", 1e-4, "Float: learning rate.")
flags.DEFINE_bool(
    "separate_learning_rates",
    True,
    "Bool: use a separate learning rate for resnet and LSTM",
)
flags.DEFINE_integer(
    "resnet_learning_rate_reduction_factor",
    10,
    "Int: divide the LSTM learning rate by this to get the resnet learning rate",
)
flags.DEFINE_string("mode", "train", "String: train, eval, or frozen_train")
flags.DEFINE_string(
    "resnet_weights_dir",
    None,
    "String: Directory for the resnet weights for evaluation or sequence training.",
)
flags.DEFINE_integer(
    "validation_split_num",
    1,
    "Int: split number for train/validations sets. Probably [0,9]",
)
flags.DEFINE_bool("early_stopping", True, "Bool: Enable early stopping?")
flags.DEFINE_integer(
    "early_stopping_counter",
    40,
    "Int: Quit training after n validation steps with no improvement.",
)
flags.DEFINE_bool(
    "with_naive_IMU",
    False,
    "Bool: Add closest matching IMU data for each frame to input to LSTM.",
)

# Things we can probably leave alone
flags.DEFINE_string(
    "dataset_root",
    "/media/luke/usb_ssd/dataset/handcam",
    "String: Root of Handcam dataset directory",
)
flags.DEFINE_integer("image_size", 112, "Image side length.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("num_classes", 7, "Number of classes.")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs to train for.")
flags.DEFINE_integer("seq_len", 60, "Int: number of frames in a sequence.")
flags.DEFINE_integer("ip_queue_capacity", 100, "Number samples in queue.")
flags.DEFINE_integer(
    "ip_num_read_threads", 6, "Number of reading threads for loading dataset."
)
flags.DEFINE_integer(
    "rnn_hidden_units", 1024, "Int: number of hidden units in RNN cell"
)
flags.DEFINE_integer("rnn_layers", 1, "Int: number of RNN layers")
flags.DEFINE_string("learning_rate_type", "exponential", "String: exponential or fixed")
flags.DEFINE_integer("print_every_step", 10, "Int: Print update after n steps.")
flags.DEFINE_integer("evaluate_every_step", 50, "Int: Print update after n steps.")
flags.DEFINE_integer("checkpoint_every_step", 100, "Int: Print update after n steps.")
flags.DEFINE_integer(
    "keep_latest_n_checkpoints", 21, "Int: Keep the latest n checkpoints."
)

flags.DEFINE_string(
    "train_dir",
    "/tmp/luke/handcam/WRN/" + today + "/train",
    "Directory to keep training outputs.",
)
flags.DEFINE_string(
    "eval_dir",
    "/tmp/luke/handcam/WRN/" + today + "/eval",
    "Directory to keep eval outputs.",
)
flags.DEFINE_integer("eval_batch_count", 20, "Number of batches to eval.")
# flags.DEFINE_string('log_root', '/tmp/luke/handcam/WRN/' + today,
#                            'Directory to keep the checkpoints. Should be a '
#                            'parent directory of FLAGS.train_dir/eval_dir.')

# Seed for repeatability.
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability.")

FLAGS = flags.FLAGS

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


# Sanity check FLAGS
if FLAGS.input_modality not in ["rgb", "rgbd", "depth"]:
    raise (ValueError("input_modality must be one of: rgb, rgbd, depth."))
else:
    print(FLAGS.input_modality)

if FLAGS.model_type not in ["single_frames", "sequence"]:
    raise (ValueError("model_type must be one of: single_frames, sequence."))

if FLAGS.model_type == "single_frames" and FLAGS.with_naive_IMU == True:
    raise (
        ValueError(
            "Cannot use with_naive_IMU on single_frames. Must be sequence data, or turn off IMU data. "
        )
    )

if FLAGS.mode not in ["train", "eval", "frozen_train"]:
    raise (ValueError("mode must be one of: train, eval, frozen_train"))

if FLAGS.resnet_size not in [18, 50]:
    raise (ValueError("resnet size must be one of: 18, 50"))

print(FLAGS.early_stopping)

# Make sure resnet_weights_dir exists if it's required
resnet_dir_exists = FLAGS.resnet_weights_dir is not None and os.path.exists(
    FLAGS.resnet_weights_dir
)

if (FLAGS.mode == "eval" and not resnet_dir_exists) or (
    FLAGS.model_type == "sequence" and not resnet_dir_exists
):
    raise (
        ValueError(
            "Expected resnet_weights_dir to exist. Tried directory: "
            + str(FLAGS.resnet_weights_dir)
        )
    )

if FLAGS.model_type == "sequence":
    # TODO: Set everything up for sequence training
    dataset_dir = FLAGS.dataset_root
    all_filenames = glob.glob(os.path.join(dataset_dir, "tfrecords", "*", "*.tfrecord"))
    with open(
        os.path.join(
            dataset_dir, "train_split" + str(FLAGS.validation_split_num) + ".pckl"
        ),
        "rb",
    ) as f:
        train_samples = pickle.load(f)

    with open(
        os.path.join(
            dataset_dir, "validation_split" + str(FLAGS.validation_split_num) + ".pckl"
        ),
        "rb",
    ) as f:
        validation_samples = pickle.load(f)

    train_tfrecord_filenames = []
    validation_tfrecord_filenames = []

    for filename in all_filenames:
        is_train = False
        for train_sample in train_samples:
            if train_sample in filename:
                # filename is a train sample
                is_train = True
                break

        if is_train:
            train_tfrecord_filenames.append(filename)
        else:
            validation_tfrecord_filenames.append(filename)

elif FLAGS.model_type == "single_frames":
    # TODO: Set everything up for single frames
    dataset_dir = os.path.join(FLAGS.dataset_root, "single_frames")
    train_tfrecord_filenames = glob.glob(
        os.path.join(
            dataset_dir,
            "tfrecords",
            "split" + str(FLAGS.validation_split_num),
            "train*.tfrecord",
        )
    )
    validation_tfrecord_filenames = glob.glob(
        os.path.join(
            dataset_dir,
            "tfrecords",
            "split" + str(FLAGS.validation_split_num),
            "validation*.tfrecord",
        )
    )
    print("Found %d training files" % len(train_tfrecord_filenames))
    print("Found %d validation files" % len(validation_tfrecord_filenames))
# /tmp/luke/handcam/split0/sequence/rgbd/frozen_train/2018-08-13
dir_input_modality = FLAGS.input_modality
if FLAGS.with_naive_IMU:
    dir_input_modality = dir_input_modality + "-imu"
log_root = os.path.join(
    "/tmp/luke/hdd-3tb/models/handcam",
    "split" + str(FLAGS.validation_split_num),
    FLAGS.model_type + "_resnet-%d" % FLAGS.resnet_size,
    dir_input_modality,
    FLAGS.mode,
    today,
)
# Cannot dump FLAGS directly, need to do some magic.
os.makedirs(log_root, exist_ok=True)
# with open(os.path.join(log_root, "FLAGS.pckl"), "wb") as f:
#     pickle.dump(FLAGS.flag_values_dict(), f)

np.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

#############
#
# Prepare the dataset
#
#############

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

samples_per_tfrecord = 400 if FLAGS.model_type == "single_frames" else 1
num_steps_per_epoch = (
    len(train_tfrecord_filenames) * samples_per_tfrecord / FLAGS.batch_size
)

shuffle(train_tfrecord_filenames)
shuffle(validation_tfrecord_filenames)

# Parser for making Features to give to tf model
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
    "accel": tf.FixedLenSequenceFeature([], dtype=tf.string),
    "gyro": tf.FixedLenSequenceFeature([], dtype=tf.string),
    "pose": tf.FixedLenSequenceFeature([], dtype=tf.string),
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


def _parse_function_single_frame(example_proto):
    # parsed_features = tf.parse_single_example(example_proto, features)
    print("next")
    features_parsed = tf.parse_single_example(example_proto, features_single_frame)

    label = features_parsed["image/class/label"]

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

    return img, label


def _parse_function_sequence(example_proto):
    # This function also needs to return aligned IMU data when FLAGS.with_naive_IMU == True.
    # Should be img = [img, imu_data] when FLAGS.with_naive_IMU == True
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
    imu_data = None

    if FLAGS.with_naive_IMU:
        accel = tf.decode_raw(sequence_parsed["accel"], tf.float32)
        accel = tf.reshape(
            accel, [-1, 3]
        )  # [-1,3] because I stripped the timestamps in the tfrecords
        gyro = tf.decode_raw(sequence_parsed["gyro"], tf.float32)
        gyro = tf.reshape(gyro, [-1, 3])

        imu_data = tf.concat([accel, gyro], axis=1)

    return (
        img,
        imu_data,
        one_hot,
        seq_len,
        first_grasp_frame,
        last_grasp_frame,
        sample_name,
    )


def _random_crop_single_frame(img, label):
    if FLAGS.input_modality == "rgb":
        img = tf.random_crop(img, [240, 240, 3])
    elif FLAGS.input_modality == "depth":
        img = tf.random_crop(img, [240, 240, 1])
    elif FLAGS.input_modality == "rgbd":
        img = tf.random_crop(img, [240, 240, 4])

    return img, label


def _center_crop_single_frame(img, label):
    img = img[8:232, 48:272, :]
    return img, label


def _preprocessing_op_single_frame(img, label):
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

    return img, one_hot


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


def trim_sequence(seq_img, imu_data, seq_labels, seq_len):
    # the sequence is long enough to crop, otherwise, it's already below the max seq_len from config
    start = tf.random_uniform(
        [], minval=0, maxval=seq_len - FLAGS.seq_len, dtype=tf.int32
    )
    end = start + FLAGS.seq_len
    seq_len = FLAGS.seq_len
    seq_img = seq_img[start:end]
    seq_labels = seq_labels[start:end]

    if imu_data is not None:
        imu_data = imu_data[start:end]

    return seq_img, imu_data, seq_labels, seq_len


def read_and_decode_sequence(filename_queue):
    readerOptions = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP
    )
    reader = tf.TFRecordReader(options=readerOptions)
    # reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    with tf.name_scope("TFRecordDecoding"):
        # parse sequence
        (
            seq_img,
            imu_data,
            seq_labels,
            seq_len,
            first_grasp_frame,
            last_grasp_frame,
            sample_name,
        ) = _parse_function_sequence(serialized_example)
        # trim sequence
        seq_img, imu_data, seq_labels, seq_len = tf.cond(
            seq_len - FLAGS.seq_len > 0,
            lambda: trim_sequence(seq_img, imu_data, seq_labels, seq_len),
            lambda: (seq_img, imu_data, seq_labels, seq_len),
        )
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

        return [seq_img, imu_data, seq_labels, seq_len, sample_name]


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
        (
            batch_rgb,
            batch_imu,
            batch_labels,
            batch_lens,
            sample_names,
        ) = tf.train.batch_join(
            sample_list,
            batch_size=FLAGS.batch_size,
            capacity=FLAGS.ip_queue_capacity,
            enqueue_many=False,
            dynamic_pad=True,
            allow_smaller_final_batch=False,
            name="batch_join_and_pad",
        )

        return batch_rgb, batch_imu, batch_labels, batch_lens, sample_names


if FLAGS.model_type == "single_frames":
    with tf.variable_scope("preprocessing"):
        # Don't ask me why I used Dataset for single frames and then input pipeline for sequences. I think I tried to use dataset
        # for sequences but it gave me some kind of trouble and I had to switch.
        # Prepare iterators and things for reading in the dataset
        filenames_placeholder_train = tf.placeholder(tf.string, shape=[None])
        filenames_placeholder_validation = tf.placeholder(tf.string, shape=[None])

        dataset_train = tf.data.TFRecordDataset(
            filenames_placeholder_train, compression_type="GZIP"
        )
        dataset_train = dataset_train.prefetch(1)
        dataset_train = dataset_train.map(
            _parse_function_single_frame, num_parallel_calls=4
        )
        dataset_train = dataset_train.shuffle(buffer_size=1000)
        dataset_train = dataset_train.repeat(FLAGS.num_epochs)
        dataset_train = dataset_train.map(
            _random_crop_single_frame, num_parallel_calls=4
        )
        dataset_train = dataset_train.map(
            _preprocessing_op_single_frame, num_parallel_calls=4
        )
        dataset_train = dataset_train.apply(
            tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size)
        )
        dataset_train = dataset_train.prefetch(1)

        dataset_validation = tf.data.TFRecordDataset(
            filenames_placeholder_validation, compression_type="GZIP"
        )
        dataset_validation = dataset_validation.prefetch(1)
        dataset_validation = dataset_validation.shuffle(buffer_size=1000)
        dataset_validation = dataset_validation.map(
            _parse_function_single_frame, num_parallel_calls=4
        )
        dataset_validation = dataset_validation.repeat(None)
        dataset_validation = dataset_validation.map(
            _center_crop_single_frame, num_parallel_calls=4
        )
        dataset_validation = dataset_validation.map(
            _preprocessing_op_single_frame, num_parallel_calls=4
        )
        dataset_validation = dataset_validation.apply(
            tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size)
        )
        dataset_validation = dataset_validation.prefetch(1)

        iterator_train = tf.data.Iterator.from_structure(
            dataset_train.output_types, dataset_train.output_shapes
        )
        initializer_train = iterator_train.make_initializer(dataset_train)
        images_train, labels_train = iterator_train.get_next()

        iterator_validation = tf.data.Iterator.from_structure(
            dataset_validation.output_types, dataset_validation.output_shapes
        )
        initializer_validation = iterator_validation.make_initializer(
            dataset_validation
        )
        images_validation, labels_validation = iterator_validation.get_next()

    train_feed_dict = {filenames_placeholder_train: train_tfrecord_filenames}
    validation_feed_dict = {
        filenames_placeholder_validation: validation_tfrecord_filenames
    }

else:
    (
        train_batch_samples_op,
        train_batch_imu_op,
        train_batch_labels_op,
        train_batch_seq_len_op,
        train_names,
    ) = input_pipeline_sequence(
        train_tfrecord_filenames, name="training_input_pipeline"
    )
    (
        valid_batch_samples_op,
        valid_batch_imu_op,
        valid_batch_labels_op,
        valid_batch_seq_len_op,
        validation_names,
    ) = input_pipeline_sequence(
        validation_tfrecord_filenames, name="validation_input_pipeline"
    )

#############
#
# Training and evaluation functions
#
#############


class ValidationLossError(Exception):
    def __init__(self, msg):
        self.msg = msg


best_val_acc = 0.0
accuracy_decrease_counter = 0

loss_avg_op = tf.placeholder(tf.float32, name="loss_avg")
accuracy_avg_op = tf.placeholder(tf.float32, name="accuracy_avg")

# Generate a variable to contain a counter for the global training step.
# Note that it is useful if you save/restore your network.
global_step = tf.Variable(1, name="global_step", trainable=False)

hps = resnet_model.HParams(
    batch_size=FLAGS.batch_size,
    num_classes=FLAGS.num_classes,
    weight_decay_rate=0.0002,
    relu_leakiness=0.1,
)

with tf.name_scope("Training"):
    # Create model

    if FLAGS.model_type == "single_frames":
        trainModel = resnet_model.ResNet(
            hps,
            images_train,
            labels_train,
            mode=FLAGS.mode,
            batch_size=FLAGS.batch_size,
            model_type=FLAGS.model_type,
            input_modality=FLAGS.input_modality,
            resnet_size=FLAGS.resnet_size,
            imu_data=None,
        )
        trainModel.build_graph()
    else:
        cnnModel = resnet_model.ResNet(
            hps,
            train_batch_samples_op,
            train_batch_labels_op,
            mode=FLAGS.mode,
            batch_size=FLAGS.batch_size,
            model_type=FLAGS.model_type,
            input_modality=FLAGS.input_modality,
            resnet_size=FLAGS.resnet_size,
            imu_data=train_batch_imu_op,
        )
        cnnModel.build_graph()
        cnn_representations = cnnModel.cnn_representations

        # test_rep = tf.placeholder(tf.float32,[8,25,2048])
        # cnnModel = VGGModel(config=config['vgg'],
        #                     input_op=train_batch_samples_op,
        #                     mode='training')
        # print("cnn rep shape: " + str(cnn_representations.shape))

        trainModel = lstm_model(
            config=rnn_config,
            input_op=cnn_representations,
            labels_op=train_batch_labels_op,
            seq_len_op=train_batch_seq_len_op,
            mode="training",
        )
        trainModel.build_graph()
        print("\n# of parameters: %s" % trainModel.num_parameters)

    # Optimization routine.
    # Learning rate is decayed in time. This enables our model using higher learning rates in the beginning.
    # In time the learning rate is decayed so that gradients don't explode and training staurates.
    # If you observe slow training, feel free to modify decay_steps and decay_rate arguments.
    if FLAGS.learning_rate_type == "exponential":
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate,
            global_step=global_step,
            decay_steps=1500,
            decay_rate=0.9,
            staircase=False,
        )
    elif FLAGS.learning_rate_type == "fixed":
        learning_rate = FLAGS.learning_rate
    else:
        print("Invalid learning rate type")
        raise ValueError()

if (
    FLAGS.model_type == "sequence"
    and FLAGS.separate_learning_rates
    and FLAGS.mode != "frozen_train"
):
    resnet_optimizer = tf.train.AdamOptimizer(
        learning_rate / FLAGS.resnet_learning_rate_reduction_factor
    )
    lstm_optimizer = tf.train.AdamOptimizer(learning_rate)

    resnet_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "WRN")
    lstm_train_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "rnn_stack/rnn"
    )

    grads = tf.gradients(trainModel.loss, resnet_train_vars + lstm_train_vars)
    resnet_grads = grads[: len(resnet_train_vars)]
    lstm_grads = grads[len(resnet_train_vars) :]
    resnet_train_op = resnet_optimizer.apply_gradients(
        zip(resnet_grads, resnet_train_vars)
    )
    lstm_train_op = lstm_optimizer.apply_gradients(
        zip(lstm_grads, lstm_train_vars), global_step=global_step
    )
    train_op = tf.group(resnet_train_op, lstm_train_op)

    # resnet_train_op = resnet_optimizer.minimize(trainModel.loss, global_step=None, var_list=resnet_train_vars)
    # lstm_train_op = lstm_optimizer.minimize(trainModel.loss, global_step=global_step, var_list=lstm_train_vars)
    # train_op = tf.group(resnet_train_op, lstm_train_op)
    # train_op = tf.group(resnet_train_op)
    # FLAGS.resnet_learning_rate_reduction_factor
else:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(trainModel.loss, global_step=global_step)

with tf.name_scope("Evaluation"):
    # Create model
    if FLAGS.model_type == "single_frames":
        validModel = resnet_model.ResNet(
            hps,
            images_validation,
            labels_validation,
            "eval",
            batch_size=FLAGS.batch_size,
            model_type=FLAGS.model_type,
            input_modality=FLAGS.input_modality,
            resnet_size=FLAGS.resnet_size,
            imu_data=None,
        )
        validModel.build_graph()
    else:
        validCnnModel = resnet_model.ResNet(
            hps,
            valid_batch_samples_op,
            valid_batch_labels_op,
            "eval",
            batch_size=FLAGS.batch_size,
            model_type=FLAGS.model_type,
            input_modality=FLAGS.input_modality,
            resnet_size=FLAGS.resnet_size,
            imu_data=valid_batch_imu_op,
        )
        validCnnModel.build_graph()
        valid_cnn_representations = validCnnModel.cnn_representations

        validModel = lstm_model(
            config=rnn_config,
            input_op=valid_cnn_representations,
            labels_op=valid_batch_labels_op,
            seq_len_op=valid_batch_seq_len_op,
            mode="validation",
        )
        validModel.build_graph()
        print("\n# of parameters: %s" % validModel.num_parameters)


# Create summary ops for monitoring the training.
# Each summary op annotates a node in the computational graph and collects
# data data from it.
summary_train_loss = tf.summary.scalar("loss", trainModel.loss)
summary_train_acc = tf.summary.scalar("accuracy_training", trainModel.batch_accuracy)
summary_avg_accuracy = tf.summary.scalar("accuracy_avg", accuracy_avg_op)
summary_avg_loss = tf.summary.scalar("loss_avg", loss_avg_op)
summary_learning_rate = tf.summary.scalar("learning_rate", learning_rate)

# Group summaries.
# summaries_training is used during training and reported after every step.
summaries_training = tf.summary.merge(
    [summary_train_loss, summary_train_acc, summary_learning_rate]
)
# summaries_evaluation is used by both trainig and validation in order to report the performance on the dataset.
summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

# Create session object
sess = tf.Session()
# Add the ops to initialize variables.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# Actually intialize the variables
sess.run(init_op)

# Register summary ops.
train_summary_dir = os.path.join(log_root, "summary", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
valid_summary_dir = os.path.join(log_root, "summary", "validation")
valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
img_d_summary_dir = os.path.join(log_root, "summary", "img")
img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
# print(batch_samples_op)
# rgb_image_train_op = train_batch_samples_op[:, tf.mod(global_step, 20),:,:,0:3]
# depth_image_train_op = train_batch_samples_op[:, tf.mod(global_step, 20),:,:,3:]
# rgb_images_summary_dir = os.path.join(config['model_dir'], "summary", "rgb_images")
# rgb_images_summary_writer = tf.summary.FileWriter(rgb_images_summary_dir, sess.graph)
# depth_images_summary_dir = os.path.join(config['model_dir'], "summary", "depth_images")
# depth_images_summary_writer = tf.summary.FileWriter(depth_images_summary_dir, sess.graph)
# rgb_summary_image = tf.summary.image("plot_train_rgb", rgb_image_train_op)
# depth_summary_image = tf.summary.image("plot_train_depth", depth_image_train_op)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver(max_to_keep=FLAGS.keep_latest_n_checkpoints)
if FLAGS.resnet_weights_dir is not None:
    saver_wrn = tf.train.Saver([v for v in tf.all_variables() if "WRN" in v.name])
    checkpoint_path = None
    if "/media/luke/hdd-3tb" in FLAGS.resnet_weights_dir:
        # checkpoints were moved, need to get latest checkpoint manually.
        checkpoint_id = get_latest_checkpoint_id(FLAGS.resnet_weights_dir)
        if type(checkpoint_id) is not str:
            checkpoint_id = str(checkpoint_id)
        if "model-" not in checkpoint_id:
            checkpoint_id = "model-" + checkpoint_id
        checkpoint_path = os.path.join(FLAGS.resnet_weights_dir, checkpoint_id)
    else:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.resnet_weights_dir)

    print("Restoring weights from: " + checkpoint_path)
    saver_wrn.restore(sess, checkpoint_path)

# Define counters in order to accumulate measurements.
counter_correct_predictions_training = 0.0
counter_loss_training = 0.0
counter_correct_predictions_validation = 0.0
counter_loss_validation = 0.0

predictions_stacked = []
labels_stacked = []

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


######
# Training Loop
######
step = 0
if FLAGS.model_type == "single_frames":
    iterator_is_training = True
    sess.run(initializer_train, feed_dict=train_feed_dict)
    sess.run(initializer_validation, feed_dict=validation_feed_dict)

try:
    while not coord.should_stop():
        step = tf.train.global_step(sess, global_step)
        # if iterator_is_training == False:
        #     sess.run(initializer, feed_dict=train_feed_dict)
        #     iterator_is_training = True

        if (step % FLAGS.checkpoint_every_step) == 0:
            ckpt_save_path = saver.save(
                sess, os.path.join(log_root, "model"), global_step
            )
            print("Model saved in file: %s" % ckpt_save_path)

        # Run the optimizer to update weights.
        # Note that "train_op" is responsible from updating network weights.
        # Only the operations that are fed are evaluated.
        # Run the optimizer to update weights.
        # train_summary, batch_acc, loss, rgb_image, depth_image, _ = sess.run([summaries_training,
        train_summary, batch_acc, loss, batch_preds, batch_labels, _ = sess.run(
            [
                summaries_training,
                trainModel.batch_accuracy,
                trainModel.loss,
                trainModel.predictions,
                trainModel.labels,
                # rgb_summary_image,
                # depth_summary_image,
                train_op,
            ],
            feed_dict={},
        )
        # Update counters.
        if FLAGS.model_type == "single_frames":
            if (predictions_stacked == []) or (labels_stacked == []):
                predictions_stacked = np.argmax(batch_preds, axis=1)
                labels_stacked = np.argmax(batch_labels, axis=1)
            else:
                predictions_stacked = np.concatenate(
                    [predictions_stacked, np.argmax(batch_preds, axis=1)], axis=0
                )
                labels_stacked = np.concatenate(
                    [labels_stacked, np.argmax(batch_labels, axis=1)], axis=0
                )
            # Write summary data.
        counter_correct_predictions_training += batch_acc
        counter_loss_training += loss
        # Write summary data.
        train_summary_writer.add_summary(train_summary, step)
        # rgb_images_summary_writer.add_summary(rgb_image)
        # depth_images_summary_writer.add_summary(depth_image)

        # Report training performance
        if (step % FLAGS.print_every_step) == 0:
            if FLAGS.model_type == "single_frames":
                # Handle confusion matrix
                img_d_summary = plot_confusion_matrix(
                    labels_stacked,
                    predictions_stacked,
                    labels=[
                        "grasp_1",
                        "grasp_2",
                        "grasp_3",
                        "grasp_4",
                        "grasp_5",
                        "grasp_6",
                        "grasp_7",
                    ],
                    tensor_name="dev/cm",
                )
                img_d_summary_writer.add_summary(img_d_summary, step)
                predictions_stacked = []
                labels_stacked = []
            accuracy_avg = counter_correct_predictions_training / FLAGS.print_every_step
            loss_avg = counter_loss_training / (FLAGS.print_every_step)
            summary_report = sess.run(
                summaries_evaluation,
                feed_dict={accuracy_avg_op: accuracy_avg, loss_avg_op: loss_avg},
            )
            train_summary_writer.add_summary(summary_report, step)
            print(
                "[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f"
                % (step / num_steps_per_epoch, step, accuracy_avg, loss_avg)
            )

            counter_correct_predictions_training = 0.0
            counter_loss_training = 0.0

        if (step % FLAGS.evaluate_every_step) == 0:
            # It is possible to create only one input pipelene queue. Hence, we create a validation queue
            # in the begining for multiple epochs and control it via a foor loop.
            # Note that we only approximate 1 validation epoch (validation doesn't have to be accurate.)
            # In other words, number of unique validation samples may differ everytime.
            # sess.run(initializer, feed_dict=validation_feed_dict)
            # iterator_is_training = False
            for eval_step in range(FLAGS.eval_batch_count):
                # Calculate average validation accuracy.
                val_batch_acc, loss = sess.run(
                    [validModel.batch_accuracy, validModel.loss], feed_dict={}
                )
                # Update counters.
                counter_correct_predictions_validation += val_batch_acc
                counter_loss_validation += loss

            # Report validation performance
            accuracy_avg = (
                counter_correct_predictions_validation / FLAGS.eval_batch_count
            )
            loss_avg = counter_loss_validation / FLAGS.eval_batch_count
            summary_report = sess.run(
                summaries_evaluation,
                feed_dict={accuracy_avg_op: accuracy_avg, loss_avg_op: loss_avg},
            )
            valid_summary_writer.add_summary(summary_report, step)
            print(
                "[%d/%d] [Validation] Accuracy: %.3f, Loss: %.3f"
                % (step / num_steps_per_epoch, step, accuracy_avg, loss_avg)
            )
            if FLAGS.early_stopping:
                if accuracy_avg < best_val_acc:
                    accuracy_decrease_counter += 1
                else:
                    accuracy_decrease_counter = 0
                    best_val_acc = accuracy_avg

                if accuracy_decrease_counter >= FLAGS.early_stopping_counter:
                    raise (
                        ValidationLossError(
                            "Accuracy failed to improve over %d validation steps"
                            % FLAGS.early_stopping_counter
                        )
                    )

            counter_correct_predictions_validation = 0.0
            counter_loss_validation = 0.0
            # sess.run(initializer, feed_dict=train_feed_dict)

except (tf.errors.OutOfRangeError, ValidationLossError) as e:
    if type(e) == tf.errors.OutOfRangeError:
        print("Model is trained for %d epochs, %d steps." % (FLAGS.num_epochs, step))
        print("Done.")
    elif type(e) == ValidationLossError:
        print(e.msg)

finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)

ckpt_save_path = saver.save(sess, os.path.join(log_root, "model"), global_step)
print("Model saved in file: %s" % ckpt_save_path)
sess.close()
