import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import imageio

imageio.plugins.ffmpeg.download()
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import cv2
import os

from handcam.ltt.datasets.handcam import HandCamDataHandler
from handcam.ltt.datasets.handcam.OniProcessingCpp import OniSampleReader

data_handler = HandCamDataHandler.Handler()

# Looking at these tutorials for plotting samples:
# <https://stackoverflow.com/questions/35281427/fast-python-plotting-library-to-draw-plots-directly-on-2d-numpy-array-image-buff>
# <http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/>

# plot_hist = False
# if plot_hist:
#     plt.hist(np.ndarray.flatten(depth_vid[depth_vid != 0]), bins=100, range=(0, 65535))
#     plt.show()

plot_static = False
if plot_static:
    sample_index = 4
    accel = data_handler.get_sample_accel(sample_index)
    gyro = data_handler.get_sample_gyro(sample_index)
    pose = data_handler.get_sample_pose(sample_index)
    arm_ready_time = data_handler.get_sample_arm_ready_time(sample_index) / 1000.0
    object_touched_time = (
        data_handler.get_sample_object_touched_time(sample_index) / 1000.0
    )

    time_acc = accel[:, 0] / 1000.0
    acc_x = accel[:, 1]
    acc_y = accel[:, 2]
    acc_z = accel[:, 3]

    time_gyro = gyro[:, 0] / 1000.0
    gyro_x = gyro[:, 1]
    gyro_y = gyro[:, 2]
    gyro_z = gyro[:, 3]

    time_pose = pose[:, 0] / 1000.0
    pose_x = pose[:, 1]
    pose_y = pose[:, 2]
    pose_z = pose[:, 3]
    pose_w = pose[:, 4]

    print(accel.shape)
    plt.subplot(3, 1, 1)
    plt.plot(time_acc, acc_x, time_acc, acc_y, time_acc, acc_z)
    plt.axvline(arm_ready_time, color="red")
    plt.axvline(object_touched_time, color="blue")
    plt.title("acc")

    plt.subplot(3, 1, 2)
    plt.plot(time_gyro, gyro_x, time_gyro, gyro_y, time_gyro, gyro_z)
    plt.axvline(arm_ready_time, color="red")
    plt.axvline(object_touched_time, color="blue")
    plt.title("gyro")

    plt.subplot(3, 1, 3)
    plt.plot(time_pose, pose_x, time_pose, pose_y, time_pose, pose_z, time_pose, pose_w)
    plt.axvline(arm_ready_time, color="red")
    plt.axvline(object_touched_time, color="blue")
    plt.title("pose")
    plt.show()

plot_animated = False
save_animated = False
if plot_animated:
    sample_index = 4

    accel = data_handler.get_sample_accel(sample_index)
    gyro = data_handler.get_sample_gyro(sample_index)
    pose = data_handler.get_sample_pose(sample_index)
    arm_ready_time = data_handler.get_sample_arm_ready_time(sample_index) / 1000.0
    object_touched_time = (
        data_handler.get_sample_object_touched_time(sample_index) / 1000.0
    )

    time_acc = accel[:, 0] / 1000.0
    acc_x = accel[:, 1]
    acc_y = accel[:, 2]
    acc_z = accel[:, 3]

    time_gyro = gyro[:, 0] / 1000.0
    gyro_x = gyro[:, 1]
    gyro_y = gyro[:, 2]
    gyro_z = gyro[:, 3]

    time_pose = pose[:, 0] / 1000.0
    pose_x = pose[:, 1]
    pose_y = pose[:, 2]
    pose_z = pose[:, 3]
    pose_w = pose[:, 4]

    print(len(time_acc))
    print(len(time_gyro))
    print(len(time_pose))
    print(time_acc[-1])

    # Plotting
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.0025)
    ax_acc = fig.add_subplot(3, 1, 1)
    ax_acc.set_ylabel("acc")
    ax_gyro = fig.add_subplot(3, 1, 2)
    ax_gyro.set_ylabel("gyro")
    ax_pose = fig.add_subplot(3, 1, 3)
    ax_pose.set_ylabel("pose")

    ax_acc.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_acc.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_acc.plot(time_acc, acc_x, alpha=0.25, color="C0")
    ax_acc.plot(time_acc, acc_y, alpha=0.25, color="C1")
    ax_acc.plot(time_acc, acc_z, alpha=0.25, color="C2")
    (line_acc_x,) = ax_acc.plot(time_acc, acc_x, color="C0")
    (line_acc_y,) = ax_acc.plot(time_acc, acc_y, color="C1")
    (line_acc_z,) = ax_acc.plot(time_acc, acc_z, color="C2")

    ax_gyro.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_gyro.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_gyro.plot(time_gyro, gyro_x, alpha=0.25, color="C0")
    ax_gyro.plot(time_gyro, gyro_y, alpha=0.25, color="C1")
    ax_gyro.plot(time_gyro, gyro_z, alpha=0.25, color="C2")
    (line_gyro_x,) = ax_gyro.plot(time_gyro, gyro_x, color="C0")
    (line_gyro_y,) = ax_gyro.plot(time_gyro, gyro_y, color="C1")
    (line_gyro_z,) = ax_gyro.plot(time_gyro, gyro_z, color="C2")

    ax_pose.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_pose.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_pose.plot(time_acc, pose_x, alpha=0.25, color="C0")
    ax_pose.plot(time_acc, pose_y, alpha=0.25, color="C1")
    ax_pose.plot(time_acc, pose_z, alpha=0.25, color="C2")
    ax_pose.plot(time_acc, pose_w, alpha=0.25, color="C3")
    (line_pose_x,) = ax_pose.plot(time_pose, pose_x, color="C0")
    (line_pose_y,) = ax_pose.plot(time_pose, pose_y, color="C1")
    (line_pose_z,) = ax_pose.plot(time_pose, pose_z, color="C2")
    (line_pose_w,) = ax_pose.plot(time_pose, pose_w, color="C3")

    def update(index):
        line_acc_x.set_data(time_acc[:index], acc_x[:index])  # update the data
        line_acc_y.set_data(time_acc[:index], acc_y[:index])  # update the data
        line_acc_z.set_data(time_acc[:index], acc_z[:index])  # update the data

        line_gyro_x.set_data(time_gyro[:index], gyro_x[:index])  # update the data
        line_gyro_y.set_data(time_gyro[:index], gyro_y[:index])  # update the data
        line_gyro_z.set_data(time_gyro[:index], gyro_z[:index])  # update the data

        line_pose_x.set_data(time_pose[:index], pose_x[:index])  # update the data
        line_pose_y.set_data(time_pose[:index], pose_y[:index])  # update the data
        line_pose_z.set_data(time_pose[:index], pose_z[:index])  # update the data
        line_pose_w.set_data(time_pose[:index], pose_w[:index])  # update the data
        return (
            line_acc_x,
            line_acc_y,
            line_acc_z,
            line_gyro_x,
            line_gyro_y,
            line_gyro_z,
            line_pose_x,
            line_pose_y,
            line_pose_z,
            line_pose_w,
        )

    # Init only required for blitting to give a clean slate.
    def init():
        line_acc_x.set_ydata(np.ma.array(acc_x, mask=True))
        line_acc_y.set_ydata(np.ma.array(acc_y, mask=True))
        line_acc_z.set_ydata(np.ma.array(acc_z, mask=True))

        line_gyro_x.set_ydata(np.ma.array(gyro_x, mask=True))
        line_gyro_y.set_ydata(np.ma.array(gyro_y, mask=True))
        line_gyro_z.set_ydata(np.ma.array(gyro_z, mask=True))

        line_pose_x.set_ydata(np.ma.array(pose_x, mask=True))
        line_pose_y.set_ydata(np.ma.array(pose_y, mask=True))
        line_pose_z.set_ydata(np.ma.array(pose_z, mask=True))
        line_pose_w.set_ydata(np.ma.array(pose_w, mask=True))
        return (
            line_acc_x,
            line_acc_y,
            line_acc_z,
            line_gyro_x,
            line_gyro_y,
            line_gyro_z,
            line_pose_x,
            line_pose_y,
            line_pose_z,
            line_pose_w,
        )

    ani = animation.FuncAnimation(
        fig, update, len(time_acc), init_func=init, interval=5, blit=True
    )
    if save_animated:
        writer = animation.FFMpegFileWriter(
            fps=30, metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save("test_sub.mp4", writer=writer)
    plt.show()

plot_combined = False
save_combined = False
if plot_combined:
    sample_index = 6

    accel = data_handler.get_sample_accel(sample_index)
    gyro = data_handler.get_sample_gyro(sample_index)
    pose = data_handler.get_sample_pose(sample_index)
    arm_ready_time = data_handler.get_sample_arm_ready_time(sample_index) / 1000.0
    object_touched_time = (
        data_handler.get_sample_object_touched_time(sample_index) / 1000.0
    )
    depth_vid = data_handler.get_sample_depth(sample_index)
    rgb_vid = data_handler.get_sample_rgb(sample_index)

    time_acc = accel[:, 0] / 1000.0
    acc_x = accel[:, 1]
    acc_y = accel[:, 2]
    acc_z = accel[:, 3]

    time_gyro = gyro[:, 0] / 1000.0
    gyro_x = gyro[:, 1]
    gyro_y = gyro[:, 2]
    gyro_z = gyro[:, 3]

    time_pose = pose[:, 0] / 1000.0
    pose_x = pose[:, 1]
    pose_y = pose[:, 2]
    pose_z = pose[:, 3]
    pose_w = pose[:, 4]

    print(rgb_vid.shape[0])
    print(depth_vid.shape[0])

    print(len(time_acc))
    print(len(time_gyro))
    print(len(time_pose))
    print(time_acc[-1])

    # Plotting
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.0025)
    ax_acc = fig.add_subplot(3, 1, 1)
    ax_acc.set_ylabel("acc")
    ax_gyro = fig.add_subplot(3, 1, 2)
    ax_gyro.set_ylabel("gyro")
    ax_pose = fig.add_subplot(3, 1, 3)
    ax_pose.set_ylabel("pose")

    ax_acc.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_acc.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_acc.plot(time_acc, acc_x, alpha=0.25, color="C0")
    ax_acc.plot(time_acc, acc_y, alpha=0.25, color="C1")
    ax_acc.plot(time_acc, acc_z, alpha=0.25, color="C2")
    (line_acc_x,) = ax_acc.plot(time_acc, acc_x, color="C0")
    (line_acc_y,) = ax_acc.plot(time_acc, acc_y, color="C1")
    (line_acc_z,) = ax_acc.plot(time_acc, acc_z, color="C2")

    ax_gyro.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_gyro.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_gyro.plot(time_gyro, gyro_x, alpha=0.25, color="C0")
    ax_gyro.plot(time_gyro, gyro_y, alpha=0.25, color="C1")
    ax_gyro.plot(time_gyro, gyro_z, alpha=0.25, color="C2")
    (line_gyro_x,) = ax_gyro.plot(time_gyro, gyro_x, color="C0")
    (line_gyro_y,) = ax_gyro.plot(time_gyro, gyro_y, color="C1")
    (line_gyro_z,) = ax_gyro.plot(time_gyro, gyro_z, color="C2")

    ax_pose.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_pose.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_pose.plot(time_pose, pose_x, alpha=0.25, color="C0")
    ax_pose.plot(time_pose, pose_y, alpha=0.25, color="C1")
    ax_pose.plot(time_pose, pose_z, alpha=0.25, color="C2")
    ax_pose.plot(time_pose, pose_w, alpha=0.25, color="C3")
    (line_pose_x,) = ax_pose.plot(time_pose, pose_x, color="C0")
    (line_pose_y,) = ax_pose.plot(time_pose, pose_y, color="C1")
    (line_pose_z,) = ax_pose.plot(time_pose, pose_z, color="C2")
    (line_pose_w,) = ax_pose.plot(time_pose, pose_w, color="C3")

    def update(index):
        line_acc_x.set_data(time_acc[:index], acc_x[:index])  # update the data
        line_acc_y.set_data(time_acc[:index], acc_y[:index])  # update the data
        line_acc_z.set_data(time_acc[:index], acc_z[:index])  # update the data

        line_gyro_x.set_data(time_gyro[:index], gyro_x[:index])  # update the data
        line_gyro_y.set_data(time_gyro[:index], gyro_y[:index])  # update the data
        line_gyro_z.set_data(time_gyro[:index], gyro_z[:index])  # update the data

        line_pose_x.set_data(time_pose[:index], pose_x[:index])  # update the data
        line_pose_y.set_data(time_pose[:index], pose_y[:index])  # update the data
        line_pose_z.set_data(time_pose[:index], pose_z[:index])  # update the data
        line_pose_w.set_data(time_pose[:index], pose_w[:index])  # update the data
        return (
            line_acc_x,
            line_acc_y,
            line_acc_z,
            line_gyro_x,
            line_gyro_y,
            line_gyro_z,
            line_pose_x,
            line_pose_y,
            line_pose_z,
            line_pose_w,
        )

    # Init only required for blitting to give a clean slate.
    def init():
        line_acc_x.set_ydata(np.ma.array(acc_x, mask=True))
        line_acc_y.set_ydata(np.ma.array(acc_y, mask=True))
        line_acc_z.set_ydata(np.ma.array(acc_z, mask=True))

        line_gyro_x.set_ydata(np.ma.array(gyro_x, mask=True))
        line_gyro_y.set_ydata(np.ma.array(gyro_y, mask=True))
        line_gyro_z.set_ydata(np.ma.array(gyro_z, mask=True))

        line_pose_x.set_ydata(np.ma.array(pose_x, mask=True))
        line_pose_y.set_ydata(np.ma.array(pose_y, mask=True))
        line_pose_z.set_ydata(np.ma.array(pose_z, mask=True))
        line_pose_w.set_ydata(np.ma.array(pose_w, mask=True))
        return (
            line_acc_x,
            line_acc_y,
            line_acc_z,
            line_gyro_x,
            line_gyro_y,
            line_gyro_z,
            line_pose_x,
            line_pose_y,
            line_pose_z,
            line_pose_w,
        )

    ani = animation.FuncAnimation(
        fig, update, len(time_acc), init_func=init, interval=5, blit=True
    )
    if save_combined:
        writer = animation.FFMpegFileWriter(
            fps=30, metadata=dict(artist="Me"), bitrate=1800
        )
        ani.save("test_sub.mp4", writer=writer)
    plt.show()

test_myo_video_alignment = True
saved_aligned = True
add_depth_map = True
v_stack = False

if test_myo_video_alignment:
    font = {"family": "normal", "weight": "bold", "size": 22}
    import matplotlib

    matplotlib.rc("font", **font)
    sample_index = 1  # sample 6 is a video of the screen being pressed. Should be able to record frame numbers
    # where this is touched, and use the myo timestamps to sync the video and myo
    sample_name = "20180404/201811-grasp_2/"

    sample_path = os.path.join(
        "/local/home/luke/datasets/handcam/samples/", sample_name
    )

    oni = OniSampleReader(sample_path)

    # accel = data_handler.get_sample_accel(sample_index)
    # gyro = data_handler.get_sample_gyro(sample_index)
    # pose = data_handler.get_sample_pose(sample_index)
    # arm_ready_time = data_handler.get_sample_arm_ready_time(sample_index)/1000.0
    # object_touched_time = data_handler.get_sample_object_touched_time(sample_index)/1000.0
    # depth_vid = data_handler.get_sample_depth(sample_index)
    # rgb_vid = data_handler.get_sample_rgb(sample_index)

    accel = oni.accel
    gyro = oni.gyro
    pose = oni.pose
    arm_ready_time = oni.misc_attrs["armReadyTime_ms"] / 1000.0
    object_touched_time = oni.misc_attrs["objectTouched_ms"] / 1000.0

    vid = oni.vid

    depth_vid = vid[..., 3:]
    rgb_vid = np.asarray(vid[..., 0:3], dtype=np.uint8)

    if add_depth_map:
        # Put the mapped depth histogram onto the RGB frames
        rgb_vid = oni.get_depth_overlay(reverse_channels=True)

    time_acc = accel[:, 0] / 1000.0
    acc_x = accel[:, 1]
    acc_y = accel[:, 2]
    acc_z = accel[:, 3]

    time_gyro = gyro[:, 0] / 1000.0
    gyro_x = gyro[:, 1]
    gyro_y = gyro[:, 2]
    gyro_z = gyro[:, 3]

    time_pose = pose[:, 0] / 1000.0
    pose_x = pose[:, 1]
    pose_y = pose[:, 2]
    pose_z = pose[:, 3]
    pose_w = pose[:, 4]

    print(rgb_vid.shape[0])
    print(depth_vid.shape[0])

    print(len(time_acc))
    print(len(time_gyro))
    print(len(time_pose))
    print(time_acc[-1])

    # Plotting

    if v_stack:
        gs = GridSpec(7, 1)
        gs.update(left=0.08, right=0.95, wspace=0.05, top=0.95, bottom=0.05)
        fig = plt.figure(figsize=(10, 20))
        # fig.subplots_adjust(hspace=0.0025)
        plt.suptitle(sample_name)
        gs.tight_layout(fig)
        ax_image = plt.subplot(gs[0:4, :])
        ax_image.axis("off")
        ax_acc = plt.subplot(gs[4, :])
        ax_acc.set_ylabel("acc")
        ax_acc.get_xaxis().set_visible(False)
        ax_gyro = plt.subplot(gs[5, :])
        ax_gyro.set_ylabel("gyro")
        ax_gyro.get_xaxis().set_visible(False)
        ax_pose = plt.subplot(gs[6, :])
        ax_pose.set_ylabel("pose")
    else:
        gs = GridSpec(3, 4)
        gs.update(left=0.08, right=0.95, wspace=0.4, top=0.9, bottom=0.1, hspace=0.05)
        fig = plt.figure(figsize=(20, 10))
        # fig.subplots_adjust(hspace=0.0025)
        plt.suptitle(sample_name)
        gs.tight_layout(fig)
        ax_image = plt.subplot(gs[:, 0:2])
        ax_image.axis("off")
        ax_acc = plt.subplot(gs[0, 2:])
        ax_acc.set_ylabel("acc (g)")
        ax_acc.get_xaxis().set_visible(False)
        ax_gyro = plt.subplot(gs[1, 2:])
        ax_gyro.set_ylabel("gyro (deg/s)")
        ax_gyro.get_xaxis().set_visible(False)
        ax_pose = plt.subplot(gs[2, 2:])
        ax_pose.set_ylabel("pose")
        ax_pose.set_xlabel("time (s)")

    # Axis sharing
    # ax_acc.get_shared_x_axes().join(ax_acc, ax_gyro, ax_pose)

    im = ax_image.imshow(rgb_vid[0], animated=True)

    ax_acc.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_acc.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_acc.plot(time_acc, acc_x, alpha=0.25, color="C0")
    ax_acc.plot(time_acc, acc_y, alpha=0.25, color="C1")
    ax_acc.plot(time_acc, acc_z, alpha=0.25, color="C2")
    (line_acc_x,) = ax_acc.plot(time_acc, acc_x, color="C0")
    (line_acc_y,) = ax_acc.plot(time_acc, acc_y, color="C1")
    (line_acc_z,) = ax_acc.plot(time_acc, acc_z, color="C2")

    ax_gyro.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_gyro.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_gyro.plot(time_gyro, gyro_x, alpha=0.25, color="C0")
    ax_gyro.plot(time_gyro, gyro_y, alpha=0.25, color="C1")
    ax_gyro.plot(time_gyro, gyro_z, alpha=0.25, color="C2")
    (line_gyro_x,) = ax_gyro.plot(time_gyro, gyro_x, color="C0")
    (line_gyro_y,) = ax_gyro.plot(time_gyro, gyro_y, color="C1")
    (line_gyro_z,) = ax_gyro.plot(time_gyro, gyro_z, color="C2")

    ax_pose.axvline(arm_ready_time, color="black", linestyle="--", alpha=0.5)
    ax_pose.axvline(object_touched_time, color="black", linestyle="-", alpha=0.5)
    ax_pose.plot(time_pose, pose_x, alpha=0.25, color="C0")
    ax_pose.plot(time_pose, pose_y, alpha=0.25, color="C1")
    ax_pose.plot(time_pose, pose_z, alpha=0.25, color="C2")
    ax_pose.plot(time_pose, pose_w, alpha=0.25, color="C3")
    (line_pose_x,) = ax_pose.plot(time_pose, pose_x, color="C0")
    (line_pose_y,) = ax_pose.plot(time_pose, pose_y, color="C1")
    (line_pose_z,) = ax_pose.plot(time_pose, pose_z, color="C2")
    (line_pose_w,) = ax_pose.plot(time_pose, pose_w, color="C3")

    rgb_frames = rgb_vid.shape[0]
    plot_length = len(time_acc)
    scale_factor = (rgb_frames - 1) / float(plot_length)

    current_rgb_index = 0

    def rgb_frame_index_holding(current_plot_index):
        return int(np.floor(current_plot_index * scale_factor))

    def update(index):
        im.set_array(rgb_vid[rgb_frame_index_holding(index)])
        line_acc_x.set_data(time_acc[:index], acc_x[:index])  # update the data
        line_acc_y.set_data(time_acc[:index], acc_y[:index])  # update the data
        line_acc_z.set_data(time_acc[:index], acc_z[:index])  # update the data

        line_gyro_x.set_data(time_gyro[:index], gyro_x[:index])  # update the data
        line_gyro_y.set_data(time_gyro[:index], gyro_y[:index])  # update the data
        line_gyro_z.set_data(time_gyro[:index], gyro_z[:index])  # update the data

        line_pose_x.set_data(time_pose[:index], pose_x[:index])  # update the data
        line_pose_y.set_data(time_pose[:index], pose_y[:index])  # update the data
        line_pose_z.set_data(time_pose[:index], pose_z[:index])  # update the data
        line_pose_w.set_data(time_pose[:index], pose_w[:index])  # update the data
        return (
            im,
            line_acc_x,
            line_acc_y,
            line_acc_z,
            line_gyro_x,
            line_gyro_y,
            line_gyro_z,
            line_pose_x,
            line_pose_y,
            line_pose_z,
            line_pose_w,
        )

    # Init only required for blitting to give a clean slate.
    def init():
        im.set_array(rgb_vid[0])
        line_acc_x.set_ydata(np.ma.array(acc_x, mask=True))
        line_acc_y.set_ydata(np.ma.array(acc_y, mask=True))
        line_acc_z.set_ydata(np.ma.array(acc_z, mask=True))

        line_gyro_x.set_ydata(np.ma.array(gyro_x, mask=True))
        line_gyro_y.set_ydata(np.ma.array(gyro_y, mask=True))
        line_gyro_z.set_ydata(np.ma.array(gyro_z, mask=True))

        line_pose_x.set_ydata(np.ma.array(pose_x, mask=True))
        line_pose_y.set_ydata(np.ma.array(pose_y, mask=True))
        line_pose_z.set_ydata(np.ma.array(pose_z, mask=True))
        line_pose_w.set_ydata(np.ma.array(pose_w, mask=True))
        return (
            im,
            line_acc_x,
            line_acc_y,
            line_acc_z,
            line_gyro_x,
            line_gyro_y,
            line_gyro_z,
            line_pose_x,
            line_pose_y,
            line_pose_z,
            line_pose_w,
        )

    ani = animation.FuncAnimation(
        fig, update, len(time_acc), init_func=init, interval=2, blit=True
    )
    if saved_aligned:
        writer = animation.FFMpegFileWriter(
            fps=54, metadata=dict(artist="Luke T. Taverne"), bitrate=1800
        )
        ani.save("test_aligned_full.mp4", writer=writer)
    else:
        plt.show()

    getting_frame_index_on_quit = False
    if getting_frame_index_on_quit:
        for i in range(rgb_vid.shape[0]):
            print("Frame: %s" % i)

            cv2.imshow("frame", cv2.cvtColor(rgb_vid[i], cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(delay=200)

            if k == 27:
                break
