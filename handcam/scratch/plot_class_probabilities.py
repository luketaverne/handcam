import glob
import pickle

font = {"family": "normal", "weight": "bold", "size": 16}
import matplotlib

matplotlib.rc("font", **font)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import imageio

from handcam.ltt.util.Utils import softmax

imageio.plugins.ffmpeg.download()
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import cv2
import os
import tensorflow as tf

#
flags = tf.app.flags

# State your dataset directory
# flags.DEFINE_string('dataset_dir', '/local/home/luke/datasets/handcam/', 'String: Your dataset directory')
flags.DEFINE_string(
    "dataset_dir",
    "/local/home/luke/datasets/handcam/tfrecords",
    "String: Your dataset directory",
)
flags.DEFINE_string(
    "model_path",
    "/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgbd/train/*/*/",
    "path to model to use the results for plotting",
)
flags.DEFINE_bool(
    "load_new_from_tfrecord",
    True,
    "Bool: Should load new or try to load the pickle file.",
)
FLAGS = flags.FLAGS

# model_list = glob.glob(os.path.join(FLAGS.model_path, "results_long_vid.pckl"))
model_list = glob.glob(os.path.join(FLAGS.model_path, "results_probabilities.pckl"))
sample_id = 0
# print(results['preds'][sample_id])
with open(model_list[0], "rb") as f:
    results = pickle.load(f)
sample_name = results["sample_names"][sample_id]  # "20180406/165402-grasp_5"

# for pred in results['preds'][sample_id]:
#     print(pred)

class_probabilities = []
for class_id in range(7):
    class_probabilities.append([])
    for pred in results["preds"][sample_id]:
        temp = softmax(pred)
        class_probabilities[class_id].append(temp[class_id])

class_probabilities = np.array([np.array(i) for i in class_probabilities])

print(len(class_probabilities[0]))
labels = np.array([np.array(np.argmax(i)) for i in results["labels"][sample_id]])
# labels = np.argmax(labels, axis=1)
print(labels)
# object_touched_time = np.argwhere(labels != 6)[-1]
object_touched_time = 10
object_touched_frame = object_touched_time / 30.0
# arm_ready_time = np.argwhere(labels != 6)[0]
arm_ready_time = 100
arm_ready_frame = arm_ready_time / 30.0
#
# for frame_id in range(len(results['labels'][sample_id])):
#     frame = results['labels'][frame_id]
#     print(frame)
#     if np.argmax(frame, axis=1) != 6:
#         arm_ready_time = frame_id
#
# for frame_id in reversed(range(len(results['labels']))):
#     frame = results['labels'][frame_id]
#     if np.argmax(frame) != 6:
#         object_touched_time = frame_id

print("Arm ready: %d" % arm_ready_time)
print("Object touched: %d" % object_touched_time)

for class_ in class_probabilities:
    print(class_)

grasp_icons = []

for name in [
    "grasp0.png",
    "grasp1.png",
    "grasp2.png",
    "grasp3.png",
    "grasp4.png",
    "grasp5.png",
    "grasp_none.png",
]:
    img = cv2.imread(os.path.join("plot_icons", name))
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img.shape)
    grasp_icons.append(img)

# load the sample as tfrecord using sample_name
sample_path = os.path.join(FLAGS.dataset_dir, sample_name + ".tfrecord")

if FLAGS.load_new_from_tfrecord:

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

    def _parse_function_sequence(example_proto):
        # parsed_features = tf.parse_single_example(example_proto, features)
        print("next")
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
        one_hot = tf.one_hot(sequence_parsed["frame_labels"], 7, dtype=tf.int64)

        return img, one_hot, seq_len, first_grasp_frame, last_grasp_frame, sample_name

    def preprocessing_op_sequence(image_op):
        """
        Creates preprocessing operations that are going to be applied on a single frame.

        TODO: Customize for your needs.
        You can do any preprocessing (masking, normalization/scaling of inputs, augmentation, etc.) by using tensorflow operations.
        Built-in image operations: https://www.tensorflow.org/api_docs/python/tf/image
        """
        with tf.name_scope("preprocessing"):
            # crop
            # image_op = image_op[8:232, 48:272, :]
            image_op.set_shape([240, 320, 4])

            return image_op

    def read_and_decode_sequence(filename_queue):
        # reader = tf.TFRecordReader()
        # _, serialized_example = reader.read(filename_queue)

        with tf.name_scope("TFRecordDecoding"):
            # parse sequence
            (
                seq_img,
                seq_labels,
                seq_len,
                first_grasp_frame,
                last_grasp_frame,
                sample_name,
            ) = _parse_function_sequence(filename_queue)

            # preprocessing each frame
            seq_img = tf.map_fn(
                lambda x: preprocessing_op_sequence(x),
                elems=seq_img,
                dtype=tf.float32,
                back_prop=False,
            )

            return [seq_img, seq_labels, seq_len, sample_name]

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
            batch_rgb, batch_labels, batch_lens, sample_names = tf.train.batch_join(
                sample_list,
                batch_size=1,
                capacity=FLAGS.ip_queue_capacity,
                enqueue_many=False,
                dynamic_pad=True,
                allow_smaller_final_batch=False,
                name="batch_join_and_pad",
            )

            return batch_rgb, batch_labels, batch_lens, sample_names

    with tf.variable_scope("preprocessing"):
        filenames_placeholder = tf.placeholder(tf.string, shape=[None])

        dataset = tf.data.TFRecordDataset(
            filenames_placeholder, compression_type="GZIP"
        )
        dataset = dataset.map(
            lambda x: read_and_decode_sequence(x), num_parallel_calls=4
        )
        dataset = dataset.repeat(1)
        # dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))
        dataset = dataset.prefetch(1)

        iterator = tf.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes
        )
        initializer = iterator.make_initializer(dataset)
        (
            test_batch_samples_op,
            test_batch_labels_op,
            test_batch_seq_len_op,
            sample_names,
        ) = iterator.get_next()

        test_feed_dict = {filenames_placeholder: [sample_path]}

    sess = tf.Session()
    # init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(initializer, feed_dict=test_feed_dict)

    vid = sess.run(test_batch_samples_op)
    with open("vid.pckl", "wb") as f:
        pickle.dump(vid, f)

    sess.close()
else:
    with open("vid.pckl", "rb") as f:
        vid = pickle.load(f)

# TODO: res[0] is shape [105,240,320,4] and ready to be used in the graphs below.
# some functions for processing the images
def rotate_for_display(vid):
    out_vid = []
    for frame in vid:
        frame = np.rot90(frame)
        frame = np.fliplr(frame)
        out_vid.append(frame)

    out_vid = np.asarray(out_vid, dtype=vid.dtype)

    return out_vid


font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (255, 255, 255)


def label_vid(vid):
    for frame_id in range(vid.shape[0]):
        label_style = "only_pred"
        if label_style == "only_pred":
            # Ground Truth
            cv2.putText(
                vid[frame_id],
                "Prediction:",
                (5, 255),
                font,
                0.6,
                text_color,
                1,
                cv2.LINE_AA,
            )
            # cv2.putText(vid[frame_id], '%d' % labels[frame_id], (40, 310), font, 0.6, text_color, 2, cv2.LINE_AA)
            vid[frame_id][320 - 48 - 10 : 320 - 10, 30:74] = grasp_icons[
                int(np.argmax(class_probabilities[:, frame_id]))
            ]

        else:
            # Ground Truth
            cv2.putText(
                vid[frame_id], "Label:", (5, 255), font, 0.6, text_color, 1, cv2.LINE_AA
            )
            # cv2.putText(vid[frame_id], '%d' % labels[frame_id], (40, 310), font, 0.6, text_color, 2, cv2.LINE_AA)
            vid[frame_id][320 - 48 - 10 : 320 - 10, 10:54] = grasp_icons[
                labels[frame_id]
            ]

            # Pred
            cv2.putText(
                vid[frame_id], "Pred:", (80, 255), font, 0.6, text_color, 1, cv2.LINE_AA
            )
            # cv2.putText(vid[frame_id], '%d' % np.argmax(class_probabilities[:,frame_id]), (100, 310), font, 0.6, text_color, 2, cv2.LINE_AA)
            vid[frame_id][320 - 48 - 10 : 320 - 10, 82 : 82 + 44] = grasp_icons[
                int(np.argmax(class_probabilities[:, frame_id]))
            ]
    return vid


# accel = data_handler.get_sample_accel(sample_index)
# gyro = data_handler.get_sample_gyro(sample_index)
# pose = data_handler.get_sample_pose(sample_index)
# arm_ready_time = data_handler.get_sample_arm_ready_time(sample_index) / 1000.0
# object_touched_time = data_handler.get_sample_object_touched_time(sample_index) / 1000.0
depth_vid = rotate_for_display(vid[0, :, :, :, 3:])
rgb_vid = rotate_for_display(np.asarray(vid[0, :, :, :, 0:3], dtype=np.uint8))
rgb_vid = label_vid(rgb_vid)
start_num = 220
rgb_vid = rgb_vid[start_num:]

class_probabilities = class_probabilities[:, start_num:]

# accel = self.get_sample_accel(sample_index)
# gyro = self.get_sample_gyro(sample_index)
# pose = self.get_sample_pose(sample_index)
# arm_ready_time = self.get_sample_arm_ready_time(sample_index) / 1000.0
# object_touched_time = self.get_sample_object_touched_time(sample_index) / 1000.0
# depth_vid = self.get_sample_depth(sample_index)
# rgb_vid = self.get_sample_rgb(sample_index)

# time_acc = accel[:, 0] / 1000.0
# acc_x = accel[:, 1]
# acc_y = accel[:, 2]
# acc_z = accel[:, 3]
#
# time_gyro = gyro[:, 0] / 1000.0
# gyro_x = gyro[:, 1]
# gyro_y = gyro[:, 2]
# gyro_z = gyro[:, 3]
#
# time_pose = pose[:, 0] / 1000.0
# pose_x = pose[:, 1]
# pose_y = pose[:, 2]
# pose_z = pose[:, 3]
# pose_w = pose[:, 4]

# Plotting
# num of horiz grids
horiz_grid_num = 20
border_of_lr = int(horiz_grid_num / 2)
prob_plot_grid_start = 3
gs = GridSpec(7, horiz_grid_num)
gs.update(left=0.08, right=0.95, wspace=0.4, top=0.9, bottom=0.1, hspace=0.13)
fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.0025)
# plt.suptitle(sample_path)
gs.tight_layout(fig)
ax_image = plt.subplot(gs[:, border_of_lr:])
ax_image.axis("off")

# No grasp
ax_no_grasp = plt.subplot(gs[0, prob_plot_grid_start:border_of_lr])
ax_no_grasp_yaxis_image = plt.subplot(gs[0, 0:prob_plot_grid_start])
ax_no_grasp_yaxis_image.imshow(grasp_icons[6])
ax_no_grasp_yaxis_image.get_xaxis().set_ticks([])
ax_no_grasp_yaxis_image.get_yaxis().set_ticks([])
ax_no_grasp_yaxis_image.axis("off")
# ax_no_grasp_yaxis_image.patch.set_visible(False)
# ax_no_grasp.set_ylabel("No Grasp", rotation='horizontal')
# ax_no_grasp.set_ylabel("No Grasp")
ax_no_grasp.get_xaxis().set_visible(False)

# Grasp0
ax_grasp0 = plt.subplot(gs[1, prob_plot_grid_start:border_of_lr])
ax_grasp0_yaxis_image = plt.subplot(gs[1, 0:prob_plot_grid_start])
ax_grasp0_yaxis_image.imshow(grasp_icons[0])
ax_grasp0_yaxis_image.get_xaxis().set_ticks([])
ax_grasp0_yaxis_image.get_yaxis().set_ticks([])
ax_grasp0_yaxis_image.axis("off")
# ax_grasp0.set_ylabel('Power Sphere')
ax_grasp0.get_xaxis().set_visible(False)
# Grasp1
ax_grasp1 = plt.subplot(gs[2, prob_plot_grid_start:border_of_lr])
ax_grasp1_yaxis_image = plt.subplot(gs[2, 0:prob_plot_grid_start])
ax_grasp1_yaxis_image.imshow(grasp_icons[1])
ax_grasp1_yaxis_image.get_xaxis().set_ticks([])
ax_grasp1_yaxis_image.get_yaxis().set_ticks([])
ax_grasp1_yaxis_image.axis("off")
# ax_grasp1.set_ylabel('Medium Wrap')
ax_grasp1.get_xaxis().set_visible(False)
# Grasp2
ax_grasp2 = plt.subplot(gs[3, prob_plot_grid_start:border_of_lr])
ax_grasp2_yaxis_image = plt.subplot(gs[3, 0:prob_plot_grid_start])
ax_grasp2_yaxis_image.imshow(grasp_icons[2])
ax_grasp2_yaxis_image.get_xaxis().set_ticks([])
ax_grasp2_yaxis_image.get_yaxis().set_ticks([])
ax_grasp2_yaxis_image.axis("off")
# ax_grasp2.set_ylabel('Tip Pinch')
ax_grasp2.get_xaxis().set_visible(False)
# Grasp3
ax_grasp3 = plt.subplot(gs[4, prob_plot_grid_start:border_of_lr])
ax_grasp3_yaxis_image = plt.subplot(gs[4, 0:prob_plot_grid_start])
ax_grasp3_yaxis_image.imshow(grasp_icons[3])
ax_grasp3_yaxis_image.get_xaxis().set_ticks([])
ax_grasp3_yaxis_image.get_yaxis().set_ticks([])
ax_grasp3_yaxis_image.axis("off")
# ax_grasp3.set_ylabel('Precision Disc')
ax_grasp3.get_xaxis().set_visible(False)
# Grasp4
ax_grasp4 = plt.subplot(gs[5, prob_plot_grid_start:border_of_lr])
ax_grasp4_yaxis_image = plt.subplot(gs[5, 0:prob_plot_grid_start])
ax_grasp4_yaxis_image.imshow(grasp_icons[4])
ax_grasp4_yaxis_image.get_xaxis().set_ticks([])
ax_grasp4_yaxis_image.get_yaxis().set_ticks([])
ax_grasp4_yaxis_image.axis("off")
# ax_grasp4.set_ylabel('Lateral Pinch')
ax_grasp4.get_xaxis().set_visible(False)
# grasp5
ax_grasp5 = plt.subplot(gs[6, prob_plot_grid_start:border_of_lr])
ax_grasp5_yaxis_image = plt.subplot(gs[6, 0:prob_plot_grid_start])
ax_grasp5_yaxis_image.imshow(grasp_icons[5])
ax_grasp5_yaxis_image.get_xaxis().set_ticks([])
ax_grasp5_yaxis_image.get_yaxis().set_ticks([])
ax_grasp5_yaxis_image.axis("off")
# ax_grasp5.set_ylabel('Writing Tripod')
ax_grasp5.set_xlabel("time (s)")

# Axis sharing
# ax_acc.get_shared_x_axes().join(ax_acc, ax_gyro, ax_pose)
im = ax_image.imshow(rgb_vid[0], animated=True)
time = np.asarray(range(rgb_vid.shape[0]), dtype=np.float) / 30.0
line_width = 3

# No grasp lines
# ax_no_grasp.axvline(arm_ready_frame, color='black', linestyle="--", alpha=0.5)
# ax_no_grasp.axvline(object_touched_frame, color='black', linestyle='--', alpha=0.5)
ax_no_grasp.plot(
    time, class_probabilities[6], alpha=0.25, color="C0", linewidth=line_width
)  # Faded line
ax_no_grasp.set_ylim(-0.05, 1.05)
ax_no_grasp.set_xlim(0, time[-1])
(line_no_grasp,) = ax_no_grasp.plot(
    time, class_probabilities[6], color="C0", linewidth=line_width
)  # actual current line

# Grasp0 lines
# ax_grasp0.axvline(arm_ready_frame, color='black', linestyle="--", alpha=0.5)
# ax_grasp0.axvline(object_touched_frame, color='black', linestyle='--', alpha=0.5)
ax_grasp0.plot(
    time, class_probabilities[0], alpha=0.25, color="C0", linewidth=line_width
)  # Fade line
ax_grasp0.set_ylim(-0.05, 1.05)
ax_grasp0.set_xlim(0, time[-1])
(line_grasp0,) = ax_grasp0.plot(
    time, class_probabilities[0], color="C0", linewidth=line_width
)  # actual current line

# Grasp1 lines
# ax_grasp1.axvline(arm_ready_frame, color='black', linestyle="--", alpha=0.5)
# ax_grasp1.axvline(object_touched_frame, color='black', linestyle='--', alpha=0.5)
ax_grasp1.plot(
    time, class_probabilities[1], alpha=0.25, color="C0", linewidth=line_width
)  # Fade line
ax_grasp1.set_ylim(-0.05, 1.05)
ax_grasp1.set_xlim(0, time[-1])
(line_grasp1,) = ax_grasp1.plot(
    time, class_probabilities[1], color="C0", linewidth=line_width
)  # actual current line

# Grasp2 lines
# ax_grasp2.axvline(arm_ready_frame, color='black', linestyle="--", alpha=0.5)
# ax_grasp2.axvline(object_touched_frame, color='black', linestyle='--', alpha=0.5)
ax_grasp2.plot(
    time, class_probabilities[2], alpha=0.25, color="C0", linewidth=line_width
)  # Fade line
ax_grasp2.set_ylim(-0.05, 1.05)
ax_grasp2.set_xlim(0, time[-1])
(line_grasp2,) = ax_grasp2.plot(
    time, class_probabilities[2], color="C0", linewidth=line_width
)  # actual current line

# Grasp3 lines
# ax_grasp3.axvline(arm_ready_frame, color='black', linestyle="--", alpha=0.5)
# ax_grasp3.axvline(object_touched_frame, color='black', linestyle='--', alpha=0.5)
ax_grasp3.plot(
    time, class_probabilities[3], alpha=0.25, color="C0", linewidth=line_width
)  # Fade line
ax_grasp3.set_ylim(-0.05, 1.05)
ax_grasp3.set_xlim(0, time[-1])
(line_grasp3,) = ax_grasp3.plot(
    time, class_probabilities[3], color="C0", linewidth=line_width
)  # actual current line

# Grasp4 lines
# ax_grasp4.axvline(arm_ready_frame, color='black', linestyle="--", alpha=0.5)
# ax_grasp4.axvline(object_touched_frame, color='black', linestyle='--', alpha=0.5)
ax_grasp4.plot(
    time, class_probabilities[4], alpha=0.25, color="C0", linewidth=line_width
)  # Fade line
ax_grasp4.set_ylim(-0.05, 1.05)
ax_grasp4.set_xlim(0, time[-1])
(line_grasp4,) = ax_grasp4.plot(
    time, class_probabilities[4], color="C0", linewidth=line_width
)  # actual current line

# Grasp5 lines
# ax_grasp5.axvline(arm_ready_frame, color='black', linestyle="--", alpha=0.5)
# ax_grasp5.axvline(object_touched_frame, color='black', linestyle='--', alpha=0.5)
ax_grasp5.plot(
    time, class_probabilities[5], alpha=0.25, color="C0", linewidth=line_width
)  # Fade line
ax_grasp5.set_ylim(-0.05, 1.05)
ax_grasp5.set_xlim(0, time[-1])
(line_grasp5,) = ax_grasp5.plot(
    time, class_probabilities[5], color="C0", linewidth=line_width
)  # actual current line


# Handle the plot backgrounds for ground truth
bad_color = "xkcd:apricot"
good_color = "xkcd:dirty blue"
good_alpha = 0.25
bad_alpha = 0.25
# ax_no_grasp.axvspan(0, arm_ready_frame, facecolor=good_color, alpha=good_alpha)
# ax_no_grasp.axvspan(arm_ready_frame, object_touched_frame, facecolor=bad_color, alpha=bad_alpha)
# ax_no_grasp.axvspan(object_touched_frame, time[-1], facecolor=good_color, alpha=good_alpha)

grasp_axes = [ax_grasp0, ax_grasp1, ax_grasp2, ax_grasp3, ax_grasp4, ax_grasp5]
correct_grasp = np.min(labels)
print("Correct grasp %d" % correct_grasp)
# for axes_index in range(len(grasp_axes)):
#    if axes_index != correct_grasp:
#        grasp_axes[axes_index].axvspan(0, time[-1], facecolor=bad_color, alpha=bad_alpha)
#    else:
#        grasp_axes[axes_index].axvspan(0, arm_ready_frame, facecolor=bad_color, alpha=bad_alpha)
#        grasp_axes[axes_index].axvspan(arm_ready_frame, object_touched_frame, facecolor=good_color, alpha=good_alpha)
#        grasp_axes[axes_index].axvspan(object_touched_frame, time[-1], facecolor=bad_color, alpha=bad_alpha)

rgb_frames = rgb_vid.shape[0]
plot_length = rgb_vid.shape[0]
scale_factor = 1

current_rgb_index = 0


def rgb_frame_index_holding(current_plot_index):
    return np.floor(current_plot_index * scale_factor)


def update(index):
    im.set_array(rgb_vid[index])
    line_no_grasp.set_data(
        time[:index], class_probabilities[6][:index]
    )  # update the data
    line_grasp0.set_data(
        time[:index], class_probabilities[0][:index]
    )  # update the data
    line_grasp1.set_data(
        time[:index], class_probabilities[1][:index]
    )  # update the data
    line_grasp2.set_data(
        time[:index], class_probabilities[2][:index]
    )  # update the data
    line_grasp3.set_data(
        time[:index], class_probabilities[3][:index]
    )  # update the data
    line_grasp4.set_data(
        time[:index], class_probabilities[4][:index]
    )  # update the data
    line_grasp5.set_data(
        time[:index], class_probabilities[5][:index]
    )  # update the data

    return (
        im,
        line_no_grasp,
        line_grasp0,
        line_grasp1,
        line_grasp2,
        line_grasp3,
        line_grasp4,
        line_grasp5,
    )
    # return im


# Init only required for blitting to give a clean slate.
def init():
    im.set_array(rgb_vid[0])
    # line_acc_x.set_ydata(np.ma.array(acc_x, mask=True))
    line_no_grasp.set_ydata(
        np.ma.array(class_probabilities[6][0], mask=True)
    )  # update the data
    line_grasp0.set_ydata(
        np.ma.array(class_probabilities[0][0], mask=True)
    )  # update the data
    line_grasp1.set_ydata(
        np.ma.array(class_probabilities[1][0], mask=True)
    )  # update the data
    line_grasp2.set_ydata(
        np.ma.array(class_probabilities[2][0], mask=True)
    )  # update the data
    line_grasp3.set_ydata(
        np.ma.array(class_probabilities[3][0], mask=True)
    )  # update the data
    line_grasp4.set_ydata(
        np.ma.array(class_probabilities[4][0], mask=True)
    )  # update the data
    line_grasp5.set_ydata(
        np.ma.array(class_probabilities[5][0], mask=True)
    )  # update the data

    return (
        im,
        line_no_grasp,
        line_grasp0,
        line_grasp1,
        line_grasp2,
        line_grasp3,
        line_grasp4,
        line_grasp5,
    )


print(np.max(rgb_vid[0]))

# plt.imshow(rgb_vid[0])
# plt.show()

# while True:
#     pass
#
ani = animation.FuncAnimation(
    fig, update, rgb_vid.shape[0], init_func=init, interval=2, blit=True
)
save_mp4 = True
if save_mp4:
    writer = animation.FFMpegFileWriter(
        fps=20, metadata=dict(artist="Luke T. Taverne"), bitrate=1800
    )
    ani.save("test_plot_class_probs.mp4", writer=writer)
else:
    plt.show()
