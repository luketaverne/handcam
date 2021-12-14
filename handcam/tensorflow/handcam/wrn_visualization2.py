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
from handcam.ltt.network.model.Wide_ResNet import (
    wide_resnet_tf_depth as resnet_model_rgbd,
)
from handcam.ltt.network.model.Wide_ResNet import wide_resnet_tf as resnet_model
from handcam.ltt.network.model.RNNModel import LSTMModel as lstm_model
import tf_cnnvis

flags = tf.app.flags

flags.DEFINE_string(
    "dataset_dir",
    "/local/home/luke/datasets/handcam/",
    "String: Your dataset directory",
)
flags.DEFINE_string("mode", "eval", "train or eval.")
flags.DEFINE_integer("image_size", 224, "Image side length.")
flags.DEFINE_integer("batch_size", 1, "Batch size.")
flags.DEFINE_integer("num_classes", 7, "Number of classes.")
flags.DEFINE_integer("ip_queue_capacity", 100, "Number samples in queue.")
flags.DEFINE_integer(
    "ip_num_read_threads", 6, "Number of reading threads for loading dataset."
)


# Seed for repeatability.
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability.")

FLAGS = flags.FLAGS

config = {}
config["batch_size"] = FLAGS.batch_size
config["test_split"] = "split0"

# CNN model parameters

model_root = "/tmp/luke"

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

single_frame_test_tfrecord_filenames = glob.glob(
    os.path.join(
        FLAGS.dataset_dir,
        "single_frames",
        "tfrecords",
        config["test_split"],
        "validation*.tfrecord",
    )
)

print("Found %d single_frame filenames" % len(single_frame_test_tfrecord_filenames))

single_frame_test_tfrecord_filenames = [single_frame_test_tfrecord_filenames[0]]

print("Only using one tfrecord file")

single_frame_features = {
    "image/img": tf.FixedLenFeature((), tf.string, default_value=""),
    "sample_name": tf.FixedLenFeature((), tf.string, default_value=""),
    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
    "image/frame_num": tf.FixedLenFeature((), tf.int64, default_value=0),
}


def _parse_function_single_frame(example_proto):
    features_parsed = tf.parse_single_example(example_proto, single_frame_features)

    label = features_parsed["image/class/label"]
    sample_name = tf.decode_raw(features_parsed["sample_name"], tf.uint8)

    img = tf.decode_raw(features_parsed["image/img"], tf.uint16)
    img = tf.reshape(img, [240, 320, 4])

    img = tf.cast(img, tf.float32)
    one_hot = tf.one_hot(label, FLAGS.num_classes, dtype=tf.int64)

    return img, one_hot, sample_name


def preprocessing_op(image_op, img_type):
    """
    Creates preprocessing operations that are going to be applied on a single frame.

    """
    assert img_type in ["rgb", "rgbd", "depth"]

    # center crop to 224x224
    image_op = image_op[8:232, 48:272, :]

    with tf.name_scope("preprocessing"):
        if img_type == "rgb":
            image_op = tf.image.per_image_standardization(image_op[..., 0:3])
            image_op.set_shape([224, 224, 3])

        elif img_type == "rgbd":
            rgb = tf.image.per_image_standardization(image_op[..., 0:3])
            depth = image_op[..., 3:] - 4000
            image_op = tf.concat([rgb, depth], axis=2)
            image_op.set_shape([224, 224, 4])

        elif img_type == "depth":
            image_op = image_op[..., 3:] - 4000
            image_op.set_shape([224, 224, 1])

        return image_op


def read_and_decode_sequence(filename_queue, is_sequence, max_seq_len, img_type):

    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)
    assert is_sequence is False
    if is_sequence:
        assert max_seq_len in [60, 15]
    assert img_type in ["rgb", "rgbd", "depth"]

    with tf.name_scope("TFRecordDecoding"):
        img, labels, sample_name = _parse_function_single_frame(filename_queue)

        img = preprocessing_op(img, img_type)

        print("img shape: " + str(img.shape))
        print("labels shape: " + str(labels.shape))

        return [img, labels, sample_name]


# Now single_frames
# tf.reset_default_graph() #graph needs to be cleared for reuse in loop

with tf.Session() as sess:

    filenames_placeholder = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(filenames_placeholder)
    dataset = dataset.map(
        lambda x: read_and_decode_sequence(x, False, None, "rgb"), num_parallel_calls=4
    )
    dataset = dataset.repeat(1)
    # dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))
    dataset = dataset.prefetch(1)

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes
    )
    initializer = iterator.make_initializer(dataset)
    test_batch_samples_op, test_batch_labels_op, sample_names = iterator.get_next()

    test_feed_dict = {filenames_placeholder: single_frame_test_tfrecord_filenames}

    # importing InceptionV5 model
    with tf.gfile.FastGFile("frozen_handcamWRN_rgb.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # t_input = tf.placeholder(np.float32, name='import/IteratorGetNext') # define the input tensor

    tf.import_graph_def(graph_def, {})

    graph = tf.get_default_graph()

    # transpose_thing = graph.get_operation_by_name('import/Inference/WRN/transpose')
    transpose_thing = graph.get_tensor_by_name("import/Inference/WRN/transpose/perm:0")

    # print(tf.shape(transpose_thing))

    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    ######
    # Eval loop
    ######
    step = 0
    test_predictions = []
    test_correct_labels = []
    test_sample_names = []
    test_accuracy = 0
    batch_counter = 0

    # init
    # sess = tf.get_default_session()
    sess.run(initializer, feed_dict=test_feed_dict)

    # print(type(sess.graph_def))

    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    result = tf_cnnvis.deepdream_visualization(
        tf.get_default_graph(),
        {transpose_thing: np.zeros((1, 224, 224, 3), np.float32)},
        "import/Inference/accuracy/Softmax",
        classes=[0, 1, 2, 3, 4, 5, 6],
        input_tensor=None,
        path_logdir="./Log",
        path_outdir="./Output",
    )
