import tensorflow as tf
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
import cv2

from handcam.ltt.network.model import CNNModel, RNNModel, VGGModel

# from config import config
# Note that the evaluation code should have the same configuration.
config = {}
# Get from dataset.
config["num_test_samples"] = 2174
config["num_validation_samples"] = 1765
config["num_training_samples"] = 5722

config["batch_size"] = 50
config["learning_rate"] = 1e-3
# Learning rate is annealed exponentally in 'exponential' case. Don't forget to change annealing configuration in the code.
config["learning_rate_type"] = "fixed"  #'exponential'

config["num_steps_per_epoch"] = int(
    config["num_training_samples"] / config["batch_size"]
)

config["num_epochs"] = 1000
config["evaluate_every_step"] = config["num_steps_per_epoch"] * 3
config["checkpoint_every_step"] = config["num_steps_per_epoch"] * 10
config["num_validation_steps"] = int(
    config["num_validation_samples"] / config["batch_size"]
)
config["print_every_step"] = config["num_steps_per_epoch"]
config["log_dir"] = "/home/ltaverne/kaggle3/runs/"

config["img_height"] = 80
config["img_width"] = 80
config["img_num_channels"] = 3
config["skeleton_size"] = 180

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
# RNN model parameters
config["rnn"] = {}
config["rnn"]["num_hidden_units"] = 512  # Number of units in an LSTM cell.
config["rnn"]["num_layers"] = 1  # Number of LSTM stack.
config["rnn"]["num_class_labels"] = 20
config["rnn"]["initializer"] = tf.contrib.layers.xavier_initializer()
config["rnn"]["batch_size"] = config["batch_size"]
config["rnn"][
    "loss_type"
] = "average"  # or 'last_step' # In the case of 'average', average of all time-steps is used instead of the last time-step.

config["ip_queue_capacity"] = config["batch_size"] * 50
config["ip_num_read_threads"] = 6

config["train_data_dir"] = "/home/ltaverne/kaggle3/data/train/"
config["train_file_format"] = "dataTrain_%d.tfrecords"
config["train_file_ids"] = list(range(1, 41))
config["valid_data_dir"] = "/home/ltaverne/kaggle3/data/validation/"
config["valid_file_format"] = "dataValidation_%d.tfrecords"
config["valid_file_ids"] = list(range(1, 16))


# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
config["model_dir"] = os.path.abspath(os.path.join(config["log_dir"], timestamp))
print("Writing to {}\n".format(config["model_dir"]))
# def garbage_segmentation_remover_op(segmentation_op):
#     print(segmentation_op)
#     print(segmentation_op.get_shape().as_list())
#     tf.sha
#     if len(segmentation_op) == 19200:
#         return 0
#     else:
#         return 1


def preprocessing_op(image_op, segmentation_op, depth_op, config):
    """
    Creates preprocessing operations that are going to be applied on a single frame.

    TODO: Customize for your needs.
    You can do any preprocessing (masking, normalization/scaling of inputs, augmentation, etc.) by using tensorflow operations.
    Built-in image operations: https://www.tensorflow.org/api_docs/python/tf/image
    """
    with tf.name_scope("preprocessing"):
        # Reshape serialized image.
        if (image_op is not None) and (segmentation_op is not None):
            image_op = tf.reshape(
                image_op, (config["img_height"], config["img_width"], 3)
            )
            depth_op = tf.reshape(
                depth_op, (config["img_height"], config["img_width"], 1)
            )

            segmentation_op = tf.reshape(
                segmentation_op, (config["img_height"], config["img_width"], 3)
            )
            # Integer to float.
            segmentation_op = tf.to_float(segmentation_op)
            # image_op = tf.to_float(image_op)

            # Need 4 tensor for next operations
            segmentation_op = tf.expand_dims(segmentation_op, 0)
            segmentation_op = tf.nn.dilation2d(
                segmentation_op,
                filter=np.ones((3, 3, 3), dtype=np.float32),
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME",
            )
            segmentation_op = tf.nn.erosion2d(
                segmentation_op,
                kernel=np.ones((3, 3, 3), dtype=np.float32),
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME",
            )

            # Make 3 tensor again
            segmentation_op = tf.squeeze(segmentation_op, 0)

            # Get segmentation mask
            segmentation_op = tf.reduce_mean(segmentation_op, axis=2) > 150

            # While it's boolean, make the thing for the cropping
            rows = tf.where(tf.reduce_any(segmentation_op[:, :], 0))
            cols = tf.where(tf.reduce_any(segmentation_op[:, :], 1))

            max_row = tf.clip_by_value(tf.reduce_max(rows), 40, 80)
            min_row = tf.clip_by_value(tf.reduce_min(rows), 0, 39)
            min_col = tf.clip_by_value(tf.reduce_min(cols), 0, 39)
            max_col = tf.clip_by_value(tf.reduce_max(cols), 40, 80)

            # offset_height = tf.clip_by_value(min_col,0,80)
            # offset_width = tf.clip_by_value(min_row,0,80)
            # target_height = tf.clip_by_value(max_col - min_col,0,80)
            # target_width = tf.clip_by_value(max_row - min_row, 0, 80)
            offset_height = min_col
            offset_width = min_row
            target_height = max_col - min_col
            target_width = max_row - min_row

            segmentation_op = tf.stack(
                [segmentation_op, segmentation_op, segmentation_op], axis=2
            )
            segmentation_op = tf.to_int32(segmentation_op)

            # image_op = tf.image.per_image_standardization(image_op)
            # depth_op = tf.image.per_image_standardization(depth_op)
            image_op = tf.to_int32(image_op)
            depth_op = tf.to_int32(depth_op)

            min_depth = tf.reduce_min(tf.to_float(depth_op))

            image_op = tf.multiply(image_op, segmentation_op)
            depth_op = tf.multiply(depth_op, segmentation_op)

            # return
            depth_op = tf.to_float(depth_op)
            max_depth = tf.reduce_max(depth_op)
            depth_op = tf.clip_by_value(
                (depth_op - min_depth) / (max_depth - min_depth + 0.01) * 255, 0, 255
            )

            # depth_op = depth_op[min_col:max_col, min_row:max_row]
            # image_op = image_op[ min_col:max_col, min_row:max_row, :]

            # image_op = tf.expand_dims(image_op, 0)
            image_op = tf.image.crop_to_bounding_box(
                image_op,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=target_height,
                target_width=target_width,
            )
            image_op = tf.image.resize_image_with_crop_or_pad(image_op, 80, 80)
            depth_op = tf.image.crop_to_bounding_box(
                depth_op,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=target_height,
                target_width=target_width,
            )
            depth_op = tf.image.resize_image_with_crop_or_pad(depth_op, 80, 80)
            # image_op = tf.image.resize_bilinear(image_op, np.asarray([80,80]))
            # image_op = tf.squeeze(image_op, 0)

            # depth_op = cv2.convertScaleAbs(depth_op)
            # depth_op =
            # Get down to 1 channel each (specifying 0 instead of :1 makes a squeeze)
            image_op = tf.image.rgb_to_grayscale(image_op)
            image_op = image_op[:, :, 0]
            # depth_op = tf.image.rgb_to_grayscale(depth_op)
            depth_op = depth_op[:, :, 0]

            # Integer to float.
            image_op = tf.to_float(image_op)
            depth_op = tf.to_float(depth_op)

            image_op = tf.stack([image_op, depth_op], axis=2)

            # Stack
            # Crop
            # image_op = tf.image.resize_image_with_crop_or_pad(image_op, 60, 60)

            # Resize operation requires 4D tensors (i.e., batch of images).
            # Reshape the image so that it looks like a batch of one sample: [1,60,60,1]
            # image_op = tf.expand_dims(image_op, 0)
            # Resize
            # image_op = tf.image.resize_bilinear(image_op, np.asarray([32,32]))
            # Reshape the image: [32,32,1]
            # image_op = tf.squeeze(image_op, 0)

            # Normalize (zero-mean unit-variance) the image locally, i.e., by using statistics of the
            # image not the whole data or sequence.
            # image_op = tf.image.per_image_standardization(image_op)

            # Flatten image
            # image_op = tf.reshape(image_op, [-1])

        return image_op


def read_and_decode_sequence(filename_queue, config):
    # Create a TFRecordReader.
    readerOptions = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP
    )
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)

    # Read one sequence sample.
    # The training and validation files contains the following fields:
    # - label: label of the sequence which take values between 1 and 20.
    # - length: length of the sequence, i.e., number of frames.
    # - depth: sequence of depth images. [length x height x width x numChannels]
    # - rgb: sequence of rgb images. [length x height x width x numChannels]
    # - segmentation: sequence of segmentation maskes. [length x height x width x numChannels]
    # - skeleton: sequence of flattened skeleton joint positions. [length x numJoints]
    #
    # The test files doesn't contain "label" field.
    # [height, width, numChannels] = [80, 80, 3]
    with tf.name_scope("TFRecordDecoding"):
        context_encoded, sequence_encoded = tf.parse_single_sequence_example(
            serialized_example,
            # "label" and "lenght" are encoded as context features.
            context_features={
                "label": tf.FixedLenFeature([], dtype=tf.int64),
                "length": tf.FixedLenFeature([], dtype=tf.int64),
            },
            # "depth", "rgb", "segmentation", "skeleton" are encoded as sequence features.
            sequence_features={
                "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
            },
        )

        # Fetch required data fields.
        # TODO: Customize for your design. Assume that only the RGB images are used for now.
        # Decode the serialized RGB images.
        seq_rgb = tf.decode_raw(sequence_encoded["rgb"], tf.uint8)
        seq_depth = tf.decode_raw(sequence_encoded["depth"], tf.uint8)
        seq_segmentation = tf.decode_raw(sequence_encoded["segmentation"], tf.uint8)
        # print(seq_segmentation.shape)
        seq_label = tf.to_int64(context_encoded["label"])
        # Tensorflow requires the labels start from 0. Before you create submission csv,
        # increment the predictions by 1.
        seq_label = seq_label - 1

        # bad_indicies = tf.map_fn(lambda x: garbage_segmentation_remover_op(x),
        #                     elems=seq_segmentation,
        #                     dtype=tf.float32,
        #                     back_prop=False)

        # Bad indicies has 0 if the mask at that index is okay, and 1 if it's bad
        # seq_rgb = [seq_rgb[i] for i in range(len(bad_indicies)) if bad_indicies[i] == 1]
        # seq_segmentation = [seq_segmentation[i] for i in range(len(bad_indicies)) if bad_indicies[i] == 1]
        # seq_label = [seq_label[i] for i in range(len(bad_indicies)) if bad_indicies[i] == 1]

        # seq_len = tf.to_int32(context_encoded['length']) - np.sum(bad_indicies)
        seq_len = tf.to_int32(context_encoded["length"])

        # Output dimnesionality: [seq_len, height, width, numChannels]
        # tf.map_fn applies the preprocessing function on every image in the sequence, i.e., frame.
        seq_rgb = tf.map_fn(
            lambda x: preprocessing_op(x[0], x[1], x[2], config),
            elems=(seq_rgb, seq_segmentation, seq_depth),
            dtype=tf.float32,
            back_prop=False,
        )

        """
        # Use skeleton only.
        seq_skeleton = tf.decode_raw(sequence_encoded['skeleton'], tf.float32)
        # Normalize skeleton so that every pose is a unit length vector.
        seq_skeleton = tf.nn.l2_normalize(seq_skeleton, dim=1)
        seq_skeleton.set_shape([None, config['skeleton_size']])

        seq_len = tf.to_int32(context_encoded['length'])
        seq_label = context_encoded['label']
        # Tensorflow requires the labels start from 0. Before you create submission csv, 
        # increment the predictions by 1.
        seq_label = seq_label - 1
        """

        return [seq_rgb, seq_label, seq_len]


def input_pipeline(filenames, config, name="input_pipeline", shuffle=True):
    with tf.name_scope(name):
        # Create a queue of TFRecord input files.
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=config["num_epochs"], shuffle=shuffle
        )
        # Read the data from TFRecord files, decode and create a list of data samples by using threads.
        sample_list = [
            read_and_decode_sequence(filename_queue, config)
            for _ in range(config["ip_num_read_threads"])
        ]
        # Create batches.
        # Since the data consists of variable-length sequences, allow padding by setting dynamic_pad parameter.
        # "batch_join" creates batches of samples and pads the sequences w.r.t the max-length sequence in the batch.
        # Hence, the padded sequence length can be different for different batches.
        batch_rgb, batch_labels, batch_lens = tf.train.batch_join(
            sample_list,
            batch_size=config["batch_size"],
            capacity=config["ip_queue_capacity"],
            enqueue_many=False,
            dynamic_pad=True,
            allow_smaller_final_batch=False,
            name="batch_join_and_pad",
        )

        return batch_rgb, batch_labels, batch_lens


######
#
# Model setup
#
######


# Create a list of tfRecord input files.
train_filenames = [
    os.path.join(config["train_data_dir"], config["train_file_format"] % i)
    for i in config["train_file_ids"]
]
# Create data loading operators. This will be represented as a node in the computational graph.
train_batch_samples_op, train_batch_labels_op, train_batch_seq_len_op = input_pipeline(
    train_filenames, config, name="training_input_pipeline"
)

# Create a list of tfRecord input files.
valid_filenames = [
    os.path.join(config["valid_data_dir"], config["valid_file_format"] % i)
    for i in config["valid_file_ids"]
]
# Create data loading operators. This will be represented as a node in the computational graph.
valid_batch_samples_op, valid_batch_labels_op, valid_batch_seq_len_op = input_pipeline(
    valid_filenames, config, name="validation_input_pipeline", shuffle=False
)

# Create placeholders for training and monitoring variables.
loss_avg_op = tf.placeholder(tf.float32, name="loss_avg")
accuracy_avg_op = tf.placeholder(tf.float32, name="accuracy_avg")

# Generate a variable to contain a counter for the global training step.
# Note that it is useful if you save/restore your network.
global_step = tf.Variable(1, name="global_step", trainable=False)

# Create seperate graphs for training and validation.
# Training graph
# Note that our model is optimized by using the training graph.
with tf.name_scope("Training"):
    # Create model
    cnnModel = CNNModel.CNN(
        config=config["cnn"], input_op=train_batch_samples_op, mode="training"
    )
    # cnnModel = VGGModel(config=config['vgg'],
    #                     input_op=train_batch_samples_op,
    #                     mode='training')
    cnn_representations = cnnModel.build_graph()

    trainModel = RNNModel.RNN(
        config=config["rnn"],
        input_op=cnn_representations,
        target_op=train_batch_labels_op,
        seq_len_op=train_batch_seq_len_op,
        mode="training",
    )
    trainModel.build_graph()
    # trainModel = VGGModel(config=config['vgg'],
    #                     input_op=train_batch_samples_op,
    #                     mode='training')
    # trainModel.build_graph()
    print("\n# of parameters: %s" % trainModel.num_parameters)

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
        raise Exception

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    train_op = optimizer.minimize(trainModel.loss, global_step=global_step)

# Validation graph.
with tf.name_scope("Evaluation"):
    # Create model
    validCnnModel = CNNModel.CNN(
        config=config["cnn"], input_op=valid_batch_samples_op, mode="validation"
    )
    # validCnnModel = VGGModel(config=config['vgg'],
    #                          input_op=valid_batch_samples_op,
    #                          mode='validation')
    valid_cnn_representations = validCnnModel.build_graph()

    validModel = RNNModel.RNN(
        config=config["rnn"],
        input_op=valid_cnn_representations,
        target_op=valid_batch_labels_op,
        seq_len_op=valid_batch_seq_len_op,
        mode="validation",
    )
    #
    # validModel = VGGModel(config=config['vgg'],
    #                          input_op=valid_batch_samples_op,
    #                          mode='validation')
    validModel.build_graph()

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
train_summary_dir = os.path.join(config["model_dir"], "summary", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
valid_summary_dir = os.path.join(config["model_dir"], "summary", "validation")
valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
print(train_batch_samples_op)
rgb_image_train_op = train_batch_samples_op[:, tf.mod(global_step, 20), :, :, :1]
depth_image_train_op = train_batch_samples_op[:, tf.mod(global_step, 20), :, :, 1:]
rgb_images_summary_dir = os.path.join(config["model_dir"], "summary", "rgb_images")
rgb_images_summary_writer = tf.summary.FileWriter(rgb_images_summary_dir, sess.graph)
depth_images_summary_dir = os.path.join(config["model_dir"], "summary", "depth_images")
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

######
# Training Loop
######
try:
    while not coord.should_stop():
        step = tf.train.global_step(sess, global_step)

        if (step % config["checkpoint_every_step"]) == 0:
            ckpt_save_path = saver.save(
                sess, os.path.join(config["model_dir"], "model"), global_step
            )
            print("Model saved in file: %s" % ckpt_save_path)

        # Run the optimizer to update weights.
        # Note that "train_op" is responsible from updating network weights.
        # Only the operations that are fed are evaluated.
        # Run the optimizer to update weights.
        (
            train_summary,
            num_correct_predictions,
            loss,
            rgb_image,
            depth_image,
            _,
        ) = sess.run(
            [
                summaries_training,
                trainModel.num_correct_predictions,
                trainModel.loss,
                rgb_summary_image,
                depth_summary_image,
                train_op,
            ],
            feed_dict={},
        )
        # Update counters.
        counter_correct_predictions_training += num_correct_predictions
        counter_loss_training += loss
        # Write summary data.
        train_summary_writer.add_summary(train_summary, step)
        rgb_images_summary_writer.add_summary(rgb_image)
        depth_images_summary_writer.add_summary(depth_image)

        # Report training performance
        if (step % config["print_every_step"]) == 0:
            accuracy_avg = counter_correct_predictions_training / (
                config["batch_size"] * config["print_every_step"]
            )
            loss_avg = counter_loss_training / (config["print_every_step"])
            summary_report = sess.run(
                summaries_evaluation,
                feed_dict={accuracy_avg_op: accuracy_avg, loss_avg_op: loss_avg},
            )
            train_summary_writer.add_summary(summary_report, step)
            print(
                "[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f"
                % (step / config["num_steps_per_epoch"], step, accuracy_avg, loss_avg)
            )

            counter_correct_predictions_training = 0.0
            counter_loss_training = 0.0

        if (step % config["evaluate_every_step"]) == 0:
            # It is possible to create only one input pipelene queue. Hence, we create a validation queue
            # in the begining for multiple epochs and control it via a foor loop.
            # Note that we only approximate 1 validation epoch (validation doesn't have to be accurate.)
            # In other words, number of unique validation samples may differ everytime.
            for eval_step in range(config["num_validation_steps"]):
                # Calculate average validation accuracy.
                num_correct_predictions, loss = sess.run(
                    [validModel.num_correct_predictions, validModel.loss], feed_dict={}
                )
                # Update counters.
                counter_correct_predictions_validation += num_correct_predictions
                counter_loss_validation += loss

            # Report validation performance
            accuracy_avg = counter_correct_predictions_validation / (
                config["batch_size"] * config["num_validation_steps"]
            )
            loss_avg = counter_loss_validation / (config["num_validation_steps"])
            summary_report = sess.run(
                summaries_evaluation,
                feed_dict={accuracy_avg_op: accuracy_avg, loss_avg_op: loss_avg},
            )
            valid_summary_writer.add_summary(summary_report, step)
            print(
                "[%d/%d] [Validation] Accuracy: %.3f, Loss: %.3f"
                % (step / config["num_steps_per_epoch"], step, accuracy_avg, loss_avg)
            )

            counter_correct_predictions_validation = 0.0
            counter_loss_validation = 0.0

except tf.errors.OutOfRangeError:
    print("Model is trained for %d epochs, %d steps." % (config["num_epochs"], step))
    print("Done.")
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)

ckpt_save_path = saver.save(
    sess, os.path.join(config["model_dir"], "model"), global_step
)
print("Model saved in file: %s" % ckpt_save_path)
sess.close()
