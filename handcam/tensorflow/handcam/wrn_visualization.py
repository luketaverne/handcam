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
from tensorflow.python.tools import freeze_graph
import sys
from tensorflow.python.tools import optimize_for_inference_lib

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

single_nets_to_eval = {
    "handcam_wrn_rgb": {
        "config": {
            "model_dir": os.path.join(model_root, "handcam/WRN/2018-04-11/20:15"),
            "resnet_model": resnet_model,
            "img_type": "rgb",
        }
    },
    # "handcam_wrn_depth": {
    #     "config":{
    #         "model_dir": os.path.join(model_root,"handcam/WRN/2018-04-09/18:37"),
    #         "resnet_model": resnet_model,
    #         "img_type": "depth"
    #     }
    # },
    # "handcam_wrn_rgbd": {
    #     "config": {
    #         "model_dir": os.path.join(model_root, "handcam/WRN/2018-04-09/16:32"),
    #         "resnet_model": resnet_model_rgbd,
    #         "img_type": "rgbd"
    #     }
    # },
}

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
for single_frame_name in single_nets_to_eval.keys():
    print("Evaluating %s" % single_frame_name)
    single_frame_params = single_nets_to_eval[single_frame_name]["config"]
    is_sequence = False
    max_seq_len = None
    loss_type = None
    img_type = single_frame_params["img_type"]
    current_resnet_model = single_frame_params["resnet_model"]
    model_dir = single_frame_params["model_dir"]
    filenames = single_frame_test_tfrecord_filenames
    checkpoint_id = tf.train.latest_checkpoint(model_dir)

    tf.reset_default_graph()  # graph needs to be cleared for reuse in loop
    filenames_placeholder = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(filenames_placeholder)
    dataset = dataset.map(
        lambda x: read_and_decode_sequence(x, is_sequence, max_seq_len, img_type),
        num_parallel_calls=4,
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

    test_feed_dict = {filenames_placeholder: filenames}

    sess = tf.Session()
    # init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    hps = resnet_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=FLAGS.num_classes,
        min_lrn_rate=0.0001,
        lrn_rate=0.1,
        num_residual_units=4,
        use_bottleneck=True,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer="adam",
    )

    with tf.name_scope("Inference"):
        # Create model
        # wrn_model = resnet_model.ResNet(hps, train_batch_samples_op,'training')
        # cnn_representations = wrn_model.build_graph()
        cnnModel = current_resnet_model.ResNet(
            hps,
            test_batch_samples_op,
            test_batch_labels_op,
            "inference",
            batch_size=FLAGS.batch_size,
        )
        cnnModel.build_graph()

    # Restore computation graph.
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(model_dir, checkpoint_id)
    print("Checkpoint path: " + checkpoint_path)
    saver.restore(sess, checkpoint_path)

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
    sess.run(initializer, feed_dict=test_feed_dict)

    print(type(sess.graph_def))

    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    # result = tf_cnnvis.deepdream_visualization(sess, {filenames_placeholder:test_feed_dict}, 'Inference/WRN/conv2d_53/Conv2D', classes=[0,1,2,3,4,5,6], input_tensor=None,
    #                                   path_logdir='./Log', path_outdir='./Output')

    # save the graph
    MODEL_NAME = "handcamWRN_rgb"

    tf.train.write_graph(sess.graph_def, ".", "handcamWRN_rgb.pbtxt", True)

    # Freeze the graph

    # input_graph_path = MODEL_NAME + '.pbtxt'
    # # checkpoint_path = './' + MODEL_NAME + '.ckpt'
    # input_saver_def_path = ""
    # input_binary = False
    # output_node_names = "Inference/accuracy/Softmax"
    # restore_op_name = "save/restore_all"
    # filename_tensor_name = "save/Const:0"
    # output_frozen_graph_name = 'frozen_' + MODEL_NAME + '.pb'
    # output_optimized_graph_name = 'optimized_' + MODEL_NAME + '.pb'
    # clear_devices = False
    #
    # freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
    #                           input_binary, checkpoint_path, output_node_names,
    #                           restore_op_name, filename_tensor_name,
    #                           output_frozen_graph_name, clear_devices, "")

    temp_placeholder = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

    temp_string_list = [3.0112] * 7

    result = tf_cnnvis.deepdream_visualization(
        sess,
        {temp_placeholder: test_batch_samples_op},
        "Inference/accuracy/Softmax",
        classes=[0, 1, 2, 3, 4, 5, 6],
        input_tensor=None,
        path_logdir="./Log",
        path_outdir="./Output",
    )

    # Optimize for inference

    # input_graph_def = tf.GraphDef()
    # with tf.gfile.Open(output_frozen_graph_name, "r") as f:
    #     data = f.read()
    #     input_graph_def.ParseFromString(data)
    #
    # output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    #     input_graph_def,
    #     ["I"],  # an array of the input node(s)
    #     ["O"],  # an array of output nodes
    #     tf.float32.as_datatype_enum)

    # Save the optimized graph

    # f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
    # f.write(output_graph_def.SerializeToString())

    # tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)
    # tf.train.write_graph(sess.graph_def, '.', 'handcamWRN_rgb.pb', False)

    # print("result: " + str(result))

    # try:
    #     while not coord.should_stop():
    #         # Get predicted labels and sample ids for submission csv.
    #         [predictions, sample_ids, out_sample_name, acc] = sess.run(
    #             [cnnModel.predictions, test_batch_labels_op, sample_names, cnnModel.batch_accuracy], feed_dict={})
    #         test_accuracy += acc
    #         batch_counter += 1
    #         test_predictions.extend(predictions)
    #         test_correct_labels.extend(sample_ids)
    #         # print(sample_ids.shape)
    #         for name in out_sample_name:
    #             test_sample_names.append(str(name.flatten(), 'ascii'))
    #             # print(str(name.flatten(), 'ascii'))
    #
    #
    # except tf.errors.OutOfRangeError:
    #     print('Done.')
    # finally:
    #     # When done, ask the threads to stop.
    #     coord.request_stop()
    #
    # # Wait for threads to finish.
    coord.join(threads)

    # test_accuracy = test_accuracy / batch_counter
    #
    # out_dict = {
    #     "preds": test_predictions,
    #     "sample_names": test_sample_names,
    #     "labels": test_correct_labels,
    #     "val_accuracy": test_accuracy
    # }
    #
    # with open(os.path.join(FLAGS.dataset_dir, single_frame_name + "_results.pckl"), "wb") as f:
    #     pickle.dump(out_dict, f)

    sess.close()
