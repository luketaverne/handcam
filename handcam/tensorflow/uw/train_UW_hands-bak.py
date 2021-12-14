from handcam.ltt.util.TFTools import _DatasetInitializerHook, shuffle_dataset

import tensorflow as tf
import random
from random import shuffle
import glob
import sys
import numpy as np
import six
import pickle
import os
import datetime
from handcam.ltt.network.model.Wide_ResNet import wide_resnet_tf as resnet_model
from handcam.ltt.network.model.Wide_ResNet import (
    wide_resnet_tf_official as resnet_model_official,
)

today = (
    datetime.datetime.today().strftime("%Y-%m-%d")
    + "/"
    + datetime.datetime.today().strftime("%H:%M")
)

flags = tf.app.flags

flags.DEFINE_string(
    "dataset_dir",
    "/local/home/luke/datasets/rgbd-dataset/",
    "String: Your dataset directory",
)
flags.DEFINE_string("mode", "eval", "train or eval.")
flags.DEFINE_integer("image_size", 224, "Image side length.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("num_classes", 51, "Number of classes.")
flags.DEFINE_string(
    "train_dir",
    "/tmp/luke/WRN/" + today + "/train",
    "Directory to keep training outputs.",
)
flags.DEFINE_string(
    "eval_dir", "/tmp/luke/WRN/" + today + "/eval", "Directory to keep eval outputs."
)
flags.DEFINE_integer("eval_batch_count", 10, "Number of batches to eval.")
flags.DEFINE_bool("eval_once", False, "Whether evaluate the model only once.")
flags.DEFINE_bool("shuffle", True, "Shuffle the filenames and the batches")
flags.DEFINE_string(
    "log_root",
    "/tmp/luke/WRN/" + today + "/",
    "Directory to keep the checkpoints. Should be a "
    "parent directory of FLAGS.train_dir/eval_dir.",
)
flags.DEFINE_integer("ip_queue_capacity", 100, "Number samples in queue.")
flags.DEFINE_integer(
    "ip_num_read_threads", 6, "Number of reading threads for loading dataset."
)

# Seed for repeatability.
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability.")

FLAGS = flags.FLAGS

np.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

config = {}
config["learning_rate_type"] = "exponential"
config["learning_rate"] = 1e-4
config["checkpoint_every_step"] = 100
config["evaluate_every_step"] = 20
config["print_every_step"] = 10
config["num_validation_steps"] = FLAGS.eval_batch_count
config["batch_size"] = FLAGS.batch_size
config["model_dir"] = FLAGS.log_root
config["num_epochs"] = 3
config["num_steps_per_epoch"] = int(np.floor(208410 / config["batch_size"]))


#############
#
# Prepare the dataset
#
#############

with open(FLAGS.dataset_dir + "class_names.pickle", "rb") as handle:
    (class_names, class_names_to_index) = pickle.load(handle)
with open(os.path.join(FLAGS.dataset_dir, "test_splits.pckl"), "rb") as f:
    test_splits = pickle.load(f)

tfrecord_filenames = glob.glob(os.path.join(FLAGS.dataset_dir, "*.tfrecord"))
# validation_tfrecord_filenames = glob.glob(FLAGS.dataset_dir + 'uw-rgbd_validation*.tfrecord')
# train_tfrecord_filenames = glob.glob(FLAGS.dataset_dir + 'uw-rgbd_train*.tfrecord')

if FLAGS.shuffle:
    shuffle(tfrecord_filenames)
    # shuffle(validation_tfrecord_filenames)
    # shuffle(train_tfrecord_filenames)

# Parser for making Features to give to tf model
features = {
    "image/img": tf.FixedLenFeature((), tf.string, default_value=""),
    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
}


def _parse_function(example_proto):
    parsed_features = tf.parse_single_example(example_proto, features)

    img = tf.decode_raw(parsed_features["image/img"], tf.float32)

    img = tf.reshape(img, [224, 224, 4])

    one_hot = tf.one_hot(
        parsed_features["image/class/label"], len(class_names), dtype=tf.int32
    )

    return img, one_hot


def read_and_decode_sequence(filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    with tf.name_scope("TFRecordDecoding"):
        img, one_hot = _parse_function(serialized_example)

        return [img, one_hot]


# train_feed_dict = {filenames_placeholder: train_tfrecord_filenames,
#                    phase_train: True}
# validation_feed_dict = {filenames_placeholder: validation_tfrecord_filenames,
#                    phase_train: False}

#############
#
# Training and evaluation functions
#
#############


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

for i in range(0, len(test_splits)):
    print("K-fold " + str(i))
    tf.reset_default_graph()
    test_instances = test_splits[i]
    train_tfrecord_filenames = []
    validation_tfrecord_filenames = []

    for filename in tfrecord_filenames:
        is_train_filename = True
        for instance in test_instances:
            if instance in filename:
                is_train_filename = False
                validation_tfrecord_filenames.append(filename)
                break

        if is_train_filename:
            train_tfrecord_filenames.append(filename)

    def input_pipeline(
        filenames, name="input_pipeline", shuffle=True, max_epochs=config["num_epochs"]
    ):
        with tf.name_scope(name):
            # Create a queue of TFRecord input files.
            filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=max_epochs, shuffle=shuffle
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
            batch_rgb, batch_labels = tf.train.batch_join(
                sample_list,
                batch_size=FLAGS.batch_size,
                capacity=FLAGS.ip_queue_capacity,
                enqueue_many=False,
                dynamic_pad=True,
                allow_smaller_final_batch=False,
                name="batch_join_and_pad",
            )

            return batch_rgb, batch_labels

    train_batch_samples_op, train_batch_labels_op = input_pipeline(
        train_tfrecord_filenames, name="training_input_pipeline"
    )
    valid_batch_samples_op, valid_batch_labels_op = input_pipeline(
        validation_tfrecord_filenames, name="validation_input_pipeline", max_epochs=None
    )

    loss_avg_op = tf.placeholder(tf.float32, name="loss_avg")
    accuracy_avg_op = tf.placeholder(tf.float32, name="accuracy_avg")

    # Generate a variable to contain a counter for the global training step.
    # Note that it is useful if you save/restore your network.
    global_step = tf.Variable(1, name="global_step", trainable=False)

    with tf.name_scope("Training"):
        # Create model
        trainModel = resnet_model.ResNet(
            hps,
            train_batch_samples_op,
            train_batch_labels_op,
            "training",
            batch_size=FLAGS.batch_size,
        )
        trainModel.build_graph()

        # Optimization routine.
        # Learning rate is decayed in time. This enables our model using higher learning rates in the beginning.
        # In time the learning rate is decayed so that gradients don't explode and training staurates.
        # If you observe slow training, feel free to modify decay_steps and decay_rate arguments.
        if config["learning_rate_type"] == "exponential":
            learning_rate = tf.train.exponential_decay(
                config["learning_rate"],
                global_step=global_step,
                decay_steps=1000,
                decay_rate=0.9,
                staircase=False,
            )
        elif config["learning_rate_type"] == "fixed":
            learning_rate = config["learning_rate"]
        else:
            print("Invalid learning rate type")
            raise ValueError()

        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(trainModel.loss, global_step=global_step)

    with tf.name_scope("Evaluation"):
        # Create model
        validModel = resnet_model.ResNet(
            hps,
            valid_batch_samples_op,
            valid_batch_labels_op,
            "validation",
            batch_size=FLAGS.batch_size,
        )
        validModel.build_graph()

    # Create summary ops for monitoring the training.
    # Each summary op annotates a node in the computational graph and collects
    # data data from it.
    summary_train_loss = tf.summary.scalar("loss", trainModel.loss)
    summary_train_acc = tf.summary.scalar(
        "accuracy_training", trainModel.batch_accuracy
    )
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
    init_op = tf.group(
        tf.global_variables_initializer(), tf.local_variables_initializer()
    )
    # Actually intialize the variables
    sess.run(init_op)

    # Register summary ops.
    train_summary_dir = os.path.join(
        config["model_dir"] + "split_" + str(i), "summary", "train"
    )
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    valid_summary_dir = os.path.join(
        config["model_dir"] + "split_" + str(i), "summary", "validation"
    )
    valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
    # print(batch_samples_op)
    rgb_image_train_op = train_batch_samples_op[:, :, :, 0:3]
    depth_image_train_op = train_batch_samples_op[:, :, :, 3:]
    rgb_images_summary_dir = os.path.join(config["model_dir"], "summary", "rgb_images")
    rgb_images_summary_writer = tf.summary.FileWriter(
        rgb_images_summary_dir, sess.graph
    )
    depth_images_summary_dir = os.path.join(
        config["model_dir"], "summary", "depth_images"
    )
    depth_images_summary_writer = tf.summary.FileWriter(
        depth_images_summary_dir, sess.graph
    )
    rgb_summary_image = tf.summary.image("plot_train_rgb", rgb_image_train_op)
    depth_summary_image = tf.summary.image("plot_train_depth", depth_image_train_op)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=3)

    # Define counters in order to accumulate measurements.
    counter_correct_predictions_training = 0.0
    counter_loss_training = 0.0
    counter_correct_predictions_validation = 0.0
    counter_loss_validation = 0.0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batch norm

    ######
    # Training Loop
    ######
    step = 0
    iterator_is_training = True
    # sess.run(initializer, feed_dict=train_feed_dict)
    try:
        while not coord.should_stop():
            step = tf.train.global_step(sess, global_step)
            # if iterator_is_training == False:
            #     sess.run(initializer, feed_dict=train_feed_dict)
            #     iterator_is_training = True

            if (step % config["checkpoint_every_step"]) == 0:
                ckpt_save_path = saver.save(
                    sess,
                    os.path.join(config["model_dir"] + "split_" + str(i), "model"),
                    global_step,
                )
                print("Model saved in file: %s" % ckpt_save_path)

            # Run the optimizer to update weights.
            # Note that "train_op" is responsible from updating network weights.
            # Only the operations that are fed are evaluated.
            # Run the optimizer to update weights.
            # train_summary, batch_acc, loss, rgb_image, depth_image, _ = sess.run([summaries_training,
            train_summary, batch_acc, loss, rgb_image, depth_image, _ = sess.run(
                [
                    summaries_training,
                    trainModel.batch_accuracy,
                    trainModel.loss,
                    rgb_summary_image,
                    depth_summary_image,
                    train_op,
                ],
                feed_dict={},
            )
            # Update counters.
            counter_correct_predictions_training += batch_acc
            counter_loss_training += loss
            # Write summary data.
            train_summary_writer.add_summary(train_summary, step)
            rgb_images_summary_writer.add_summary(rgb_image)
            depth_images_summary_writer.add_summary(depth_image)

            # Report training performance
            if (step % config["print_every_step"]) == 0:
                accuracy_avg = (
                    counter_correct_predictions_training / config["print_every_step"]
                )
                loss_avg = counter_loss_training / (config["print_every_step"])
                summary_report = sess.run(
                    summaries_evaluation,
                    feed_dict={accuracy_avg_op: accuracy_avg, loss_avg_op: loss_avg},
                )
                train_summary_writer.add_summary(summary_report, step)
                print(
                    "[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f"
                    % (
                        step / config["num_steps_per_epoch"],
                        step,
                        accuracy_avg,
                        loss_avg,
                    )
                )

                counter_correct_predictions_training = 0.0
                counter_loss_training = 0.0

            if (step % config["evaluate_every_step"]) == 0:
                # It is possible to create only one input pipelene queue. Hence, we create a validation queue
                # in the begining for multiple epochs and control it via a foor loop.
                # Note that we only approximate 1 validation epoch (validation doesn't have to be accurate.)
                # In other words, number of unique validation samples may differ everytime.
                # sess.run(initializer, feed_dict=validation_feed_dict)
                # iterator_is_training = False
                for eval_step in range(config["num_validation_steps"]):
                    # Calculate average validation accuracy.
                    val_batch_acc, loss = sess.run(
                        [validModel.batch_accuracy, validModel.loss], feed_dict={}
                    )
                    # Update counters.
                    counter_correct_predictions_validation += val_batch_acc
                    counter_loss_validation += loss

                # Report validation performance
                accuracy_avg = (
                    counter_correct_predictions_validation
                    / config["num_validation_steps"]
                )
                loss_avg = counter_loss_validation / (config["num_validation_steps"])
                summary_report = sess.run(
                    summaries_evaluation,
                    feed_dict={accuracy_avg_op: accuracy_avg, loss_avg_op: loss_avg},
                )
                valid_summary_writer.add_summary(summary_report, step)
                print(
                    "[%d/%d] [Validation] Accuracy: %.3f, Loss: %.3f"
                    % (
                        step / config["num_steps_per_epoch"],
                        step,
                        accuracy_avg,
                        loss_avg,
                    )
                )

                counter_correct_predictions_validation = 0.0
                counter_loss_validation = 0.0
                # sess.run(initializer, feed_dict=train_feed_dict)

    except tf.errors.OutOfRangeError:
        print(
            "Model is trained for %d epochs, %d steps." % (config["num_epochs"], step)
        )
        print("Done.")
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    ckpt_save_path = saver.save(
        sess,
        os.path.join(config["model_dir"] + "split_" + str(i), "model"),
        global_step,
    )
    print("Model saved in file: %s" % ckpt_save_path)
    sess.close()
