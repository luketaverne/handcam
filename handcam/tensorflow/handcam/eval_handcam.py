from handcam.ltt.util.TFTools import _DatasetInitializerHook, shuffle_dataset

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
from handcam.ltt.network.model.Wide_ResNet import (
    wide_resnet_tf_official as resnet_model,
)
from handcam.ltt.network.model.RNNModel import LSTMModel as lstm_model
from handcam.ltt.network.model.CNNModel import CNN_depth as cnn_model

flags = tf.app.flags

desired_model_dir = "2018-04-01/11:51/"
checkpoint_id = "model-6300"  # None or a number

flags.DEFINE_string(
    "dataset_dir",
    "/local/home/luke/datasets/handcam/",
    "String: Your dataset directory",
)
flags.DEFINE_string("mode", "eval", "train or eval.")
flags.DEFINE_integer("image_size", 224, "Image side length.")
flags.DEFINE_integer("batch_size", 1, "Batch size.")
flags.DEFINE_integer("num_classes", 7, "Number of classes.")
flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("num_hidden_units", 512, "Number of hidden units.")
flags.DEFINE_integer("ip_queue_capacity", 100, "Number samples in queue.")
flags.DEFINE_integer(
    "ip_num_read_threads", 1, "Number of reading threads for loading dataset."
)
flags.DEFINE_integer("eval_batch_count", 1, "Number of batches to eval.")
flags.DEFINE_bool("eval_once", False, "Whether evaluate the model only once.")
flags.DEFINE_bool("shuffle", False, "Shuffle the filenames and the batches")
flags.DEFINE_string(
    "log_root",
    "/tmp/luke/handcam/" + desired_model_dir,
    "Directory to keep the checkpoints. Should be a "
    "parent directory of FLAGS.train_dir/eval_dir.",
)

# Seed for repeatability.
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability.")

FLAGS = flags.FLAGS

config = {}
config["learning_rate_type"] = "exponential"
config["learning_rate"] = 1e-5
config["checkpoint_every_step"] = 100
config["print_every_step"] = 10
config["batch_size"] = FLAGS.batch_size
config["model_dir"] = FLAGS.log_root
config["num_epochs"] = 1
config["checkpoint_id"] = checkpoint_id
# CNN model parameters
config["cnn"] = {}
config["cnn"]["cnn_filters"] = [
    16,
    32,
    64,
    128,
]  # Number of filters for every convolutional layer.
config["cnn"][
    "num_hidden_units"
] = 512  # Number of output units, i.e. representation size.
config["cnn"]["dropout_rate"] = 0.5
config["cnn"]["initializer"] = tf.contrib.layers.xavier_initializer()
config["rnn"] = {}
config["rnn"]["num_hidden_units"] = 512  # Number of units in an LSTM cell.
config["rnn"]["num_layers"] = 1  # Number of LSTM stack.
config["rnn"]["num_class_labels"] = FLAGS.num_classes
config["rnn"]["initializer"] = tf.contrib.layers.xavier_initializer()
config["rnn"]["batch_size"] = FLAGS.batch_size
config["rnn"][
    "loss_type"
] = "last_step"  # or 'last_step' # In the case of 'average', average of all time-steps is used instead of the last time-step.


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

# validation_tfrecord_filenames = glob.glob(FLAGS.dataset_dir + 'uw-rgbd_validation*.tfrecord')
test_tfrecord_filenames = glob.glob(
    os.path.join(FLAGS.dataset_dir, "tfrecords", "*.tfrecord")
)
# Only use a few of the files
test_tfrecord_filenames = test_tfrecord_filenames[0:20]
print(len(test_tfrecord_filenames))

config["num_steps_per_epoch"] = len(test_tfrecord_filenames) * 2 / FLAGS.batch_size

if FLAGS.shuffle:
    # shuffle(validation_tfrecord_filenames)
    shuffle(test_tfrecord_filenames)
    shuffle(test_tfrecord_filenames)

# Parser for making Features to give to tf model
context_features = {
    "vid_length": tf.FixedLenFeature((), tf.int64),
    "first_grasp_frame": tf.FixedLenFeature((), tf.int64),
    "last_grasp_frame": tf.FixedLenFeature((), tf.int64),
    "sample_name": tf.FixedLenFeature((), tf.string),
}

sequence_features = {
    "vid": tf.FixedLenSequenceFeature([], dtype=tf.string),
    "frame_labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
}


def _parse_function(example_proto):
    # parsed_features = tf.parse_single_example(example_proto, features)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        example_proto, context_features, sequence_features
    )
    seq_len = tf.to_int32(context_parsed["vid_length"])
    first_grasp_frame = tf.to_int32(context_parsed["first_grasp_frame"])
    last_grasp_frame = tf.to_int32(context_parsed["last_grasp_frame"])
    sample_name = tf.decode_raw(context_parsed["sample_name"], tf.uint8)

    # frame_labels = tf.to_int32(sequence_parsed["frame_labels"])
    img = tf.decode_raw(sequence_parsed["vid"], tf.uint16)
    img = tf.reshape(img, [-1, 240, 320, 4])
    img = tf.image.resize_image_with_crop_or_pad(img, 112, 112)

    img = tf.cast(img, tf.float32)
    # labels = tf.reshape(labels, [seq_len])
    one_hot = tf.one_hot(
        sequence_parsed["frame_labels"], FLAGS.num_classes, dtype=tf.int64
    )
    # sparse_labels = tf.to_int64(sequence_parsed["frame_labels"])
    # print(one_hot.shape)

    # testing direct input to lstm
    # img = img[...,0:3]
    # img = tf.image.per_image_standardization(img)
    # img = tf.reshape(img,[-1, 56*56*3])

    return img, one_hot, seq_len, first_grasp_frame, last_grasp_frame, sample_name


#############
#
# Training and evaluation functions
#
#############
def preprocessing_op(image_op, config):
    """
    Creates preprocessing operations that are going to be applied on a single frame.

    TODO: Customize for your needs.
    You can do any preprocessing (masking, normalization/scaling of inputs, augmentation, etc.) by using tensorflow operations.
    Built-in image operations: https://www.tensorflow.org/api_docs/python/tf/image
    """
    with tf.name_scope("preprocessing"):
        # Reshape serialized image.
        # image_op = tf.reshape(image_op, (config['img_height'],
        #                                  config['img_width'],
        #                                  config['img_num_channels'])
        #                       )
        # # Convert from RGB to grayscale.
        # image_op = tf.image.rgb_to_grayscale(image_op)
        #
        # # Integer to float.
        # image_op = tf.to_float(image_op)
        # # Crop
        # image_op = tf.image.resize_image_with_crop_or_pad(image_op, 60, 60)
        #
        # # Resize operation requires 4D tensors (i.e., batch of images).
        # # Reshape the image so that it looks like a batch of one sample: [1,60,60,1]
        # image_op = tf.expand_dims(image_op, 0)
        # # Resize
        # image_op = tf.image.resize_bilinear(image_op, np.asarray([32, 32]))
        # # Reshape the image: [32,32,1]
        # image_op = tf.squeeze(image_op, 0)
        #
        # # Normalize (zero-mean unit-variance) the image locally, i.e., by using statistics of the
        # # image not the whole data or sequence.
        # image_op = tf.image.per_image_standardization(image_op[...,0:3])

        # Flatten image
        # image_op = tf.reshape(image_op, [-1])

        return image_op


def read_and_decode_sequence(serialized_example):

    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)

    with tf.name_scope("TFRecordDecoding"):
        (
            seq_img,
            seq_labels,
            seq_len,
            first_grasp_frame,
            last_grasp_frame,
            sample_name,
        ) = _parse_function(serialized_example)
        # print(seq_labels.shape)
        # start = tf.random_uniform([], minval=0, maxval=tf.cast(tf.scalar_mul(0.5,tf.cast(seq_len, tf.float32)), tf.int32), dtype=tf.int32)
        start = first_grasp_frame
        # end = tf.random_uniform([], minval=tf.cast(tf.scalar_mul(0.5,tf.cast(seq_len, tf.float32)), tf.int32), maxval=seq_len, dtype=tf.int32)
        end = last_grasp_frame
        seq_img = seq_img[start:end]
        seq_labels = seq_labels[start:end]
        seq_len = tf.shape(seq_img)[0]
        seq_img = tf.map_fn(
            lambda x: preprocessing_op(x, config),
            elems=seq_img,
            dtype=tf.float32,
            back_prop=False,
        )

        print("seq_img shape: " + str(seq_img.shape))
        print("seq_len shape: " + str(seq_len.shape))
        print("seq_labels shape: " + str(seq_labels.shape))

        return [seq_img, seq_labels, seq_len, sample_name]


def input_pipeline(filenames, name="input_pipeline", shuffle=True):
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


# Prepare iterators and things for reading in the dataset
filenames_placeholder = tf.placeholder(tf.string, shape=[None])
phase_train = tf.placeholder(tf.bool, name="phase_train")

dataset = tf.data.TFRecordDataset(filenames_placeholder)
dataset = dataset.map(read_and_decode_sequence, num_parallel_calls=4)
dataset = dataset.repeat(FLAGS.num_epochs)
# dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
# dataset = dataset.prefetch(1)

iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
initializer = iterator.make_initializer(dataset)
(
    test_batch_samples_op,
    test_batch_labels_op,
    test_batch_seq_len_op,
    sample_names,
) = iterator.get_next()

test_feed_dict = {filenames_placeholder: test_tfrecord_filenames, phase_train: False}

#############
#
# Model setup
#
#############

# Generate a variable to contain a counter for the global training step.
# Note that it is useful if you save/restore your network.

sess = tf.Session()
# init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

with tf.name_scope("Inference"):
    # Create model
    # wrn_model = resnet_model.ResNet(hps, train_batch_samples_op,'training')
    # cnn_representations = wrn_model.build_graph()
    cnnModel = cnn_model(config["cnn"], test_batch_samples_op, mode="inference")
    cnn_representations = cnnModel.build_graph()

    inferModel = lstm_model(
        config=config["rnn"],
        input_op=cnn_representations,
        target_op=test_batch_labels_op,
        seq_len_op=test_batch_seq_len_op,
        mode="inference",
    )
    inferModel.build_graph()
    # trainModel = VGGModel(config=config['vgg'],
    #                     input_op=train_batch_samples_op,
    #                     mode='training')
    # trainModel.build_graph()
    print("\n# of parameters: %s" % inferModel.num_parameters)


# Restore computation graph.
saver = tf.train.Saver()
# Restore variables.
checkpoint_path = os.path.join(config["model_dir"], config["checkpoint_id"])
if checkpoint_path is None:
    checkpoint_path = tf.train.latest_checkpoint(config["model_dir"])
print("Evaluating " + checkpoint_path)
saver.restore(sess, checkpoint_path)

######
# Eval loop
######
step = 0
test_predictions = []
test_correct_labels = []
test_sample_names = []

# init
sess.run(initializer, feed_dict=test_feed_dict)

try:
    while not coord.should_stop():
        # Get predicted labels and sample ids for submission csv.
        [predictions, sample_ids, out_sample_name] = sess.run(
            [inferModel.predictions, test_batch_labels_op, sample_names], feed_dict={}
        )
        test_predictions.extend(predictions)
        test_correct_labels.extend(sample_ids)
        print(sample_ids.shape)
        for name in out_sample_name:
            test_sample_names.append(str(name.flatten(), "ascii"))
            print(str(name.flatten(), "ascii"))


except tf.errors.OutOfRangeError:
    print("Done.")
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)

with open("/local/home/luke/predictions.pckl", "wb") as f:
    pickle.dump(test_predictions, f)

with open("/local/home/luke/labels.pckl", "wb") as f:
    pickle.dump(test_correct_labels, f)

with open("/local/home/luke/sample_names.pckl", "wb") as f:
    pickle.dump(test_sample_names, f)

sess.close()
