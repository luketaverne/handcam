import sys
import threading
from typing import Any, Tuple

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
# import h5py #put this back in after the h5py package has a new update on pip. See <https://github.com/h5py/h5py/issues/995>
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from dask import delayed, threaded, compute
from scipy.ndimage.interpolation import shift
import cv2
from handcam.ltt.util.Preprocessing import simple_crop_batch
from scipy.ndimage import zoom

if sys.version_info[0] < 3:
    raise Exception("Please use python 3, I can't guarantee it will work otherwise.")


class Handler:
    """
    This class assumes that you already have access to/ created the h5py file
    """

    h5py_file_path = "/local/home/luke/datasets/handcam/handcam.hdf5"

    def __init__(
        self,
        validation_split=0.10,
        random_seed=315,
        clutter_allowed=True,
        controlled_lighting=False,
    ):
        self.f = h5py.File(self.h5py_file_path, "r")
        self.sample_ids = self.f[
            "sample_ids"
        ]  # ex: self.sample_ids[0] = '153434-grasp_3'
        self.num_examples = len(self.sample_ids)

        self.grasp_index_dict = {
            "grasp_1": 0,
            "grasp_2": 1,
            "grasp_3": 2,
            "grasp_4": 3,
            "grasp_5": 4,
            "grasp_6": 5,
        }

        self.valid_sample_indicies = [
            str(x) for x in range(self.num_examples)
        ]  # all possible samples

        self.__calc_dataset_stats__()
        self.__handle_clutter__(
            clutter_allowed
        )  # remove clutter from self.valid_sample_indicies or not
        self.__handle_lighting__(
            controlled_lighting
        )  # remove bad lighting from self.valid_sample_indicies or not
        self.__handle_validation_split__(
            validation_split, random_seed
        )  # split self.valid_sample_indicies into train and test

    def train_generator(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        pass

    def validation_generator(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        pass

    def plot_sample(self, sample_index, save_mp4=False):
        accel = self.get_sample_accel(sample_index)
        gyro = self.get_sample_gyro(sample_index)
        pose = self.get_sample_pose(sample_index)
        arm_ready_time = self.get_sample_arm_ready_time(sample_index) / 1000.0
        object_touched_time = self.get_sample_object_touched_time(sample_index) / 1000.0
        depth_vid = self.get_sample_depth(sample_index)
        rgb_vid = self.get_sample_rgb(sample_index)

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

        # Plotting
        gs = GridSpec(7, 1)
        gs.update(left=0.08, right=0.95, wspace=0.05, top=0.95, bottom=0.05)
        fig = plt.figure(figsize=(10, 20))
        # fig.subplots_adjust(hspace=0.0025)
        plt.suptitle(self.get_sample_name(sample_index))
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
            return np.floor(current_plot_index * scale_factor)

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
        if save_mp4:
            writer = animation.FFMpegFileWriter(
                fps=54, metadata=dict(artist="Luke T. Taverne"), bitrate=1800
            )
            ani.save(
                self.get_sample_name(sample_index).replace("/", "_") + ".mp4",
                writer=writer,
            )
        else:
            plt.show()

    def get_sample(self, sample_index):
        """
        Return the h5py Group for sample_index, if it exists
        :param sample_index:
        :return:
        """
        return self.__get_sample_part__(sample_index)

    def get_sample_rgb(self, sample_index):
        """
        Return the rgb video for sample_index, if it exists.
        :param sample_index:
        :return:
        """
        return self.__get_sample_part__(sample_index, sample_part="rgb")

    def get_sample_depth(self, sample_index):
        """
        Return the depth video for sample_index, if it exists.
        :param sample_index:
        :return:
        """
        return self.__get_sample_part__(sample_index, sample_part="depth")

    def get_sample_accel(self, sample_index):
        """
        Return the accel for sample_index, if it exists
        :param sample_index:
        :return:
        """
        return self.__get_sample_part__(sample_index, sample_part="accel")

    def get_sample_gyro(self, sample_index):
        """
        Return the accel for sample_index, if it exists
        :param sample_index:
        :return:
        """
        return self.__get_sample_part__(sample_index, sample_part="gyro")

    def get_sample_pose(self, sample_index):
        """
        Return the accel for sample_index, if it exists
        :param sample_index:
        :return:
        """
        return self.__get_sample_part__(sample_index, sample_part="pose")

    def get_sample_name(self, sample_index):
        """
        Return the sample_name for sample_index, if it exists
        :param sample_index:
        :return:
        """
        return self.__get_sample_attrs_part__(sample_index, sample_attr="sample_name")

    def get_sample_label(self, sample_index):
        """
        Return the sample_label for sample_index, if it exists
        :param sample_index:
        :return:
        """
        return self.__get_sample_attrs_part__(sample_index, sample_attr="label")

    def get_sample_attrs(self, sample_index):
        """
        Return all attributes as dict for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return self.__get_sample_attrs_part__(sample_index)

    def get_sample_arm_ready_time(self, sample_index):
        """
        Return armReadyTime_ms for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return int(
            self.__get_sample_attrs_part__(sample_index, sample_attr="armReadyTime_ms")
        )

    def get_sample_object_touched_time(self, sample_index):
        """
        Return objectTouched_ms for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return int(
            self.__get_sample_attrs_part__(sample_index, sample_attr="objectTouched_ms")
        )

    def get_sample_lighting(self, sample_index):
        """
        Return lighting for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return self.__get_sample_attrs_part__(sample_index, sample_attr="lighting")

    def get_sample_clutter(self, sample_index):
        """
        Return armReadyTime_ms for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return self.__get_sample_attrs_part__(sample_index, sample_attr="clutter")

    def get_sample_handedness(self, sample_index):
        """
        Return handedness for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return self.__get_sample_attrs_part__(sample_index, sample_attr="handedness")

    def get_sample_cameraid(self, sample_index):
        """
        Return cameraid for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return self.__get_sample_attrs_part__(sample_index, sample_attr="cameraid")

    def get_sample_arm_subjectid(self, sample_index: int):
        """
        Return subjectid for sample_index, if it exists
        :param sample_index:
        :return:
        """

        return self.__get_sample_attrs_part__(sample_index, sample_attr="subjectid")

    def __calc_dataset_stats__(self) -> None:
        """
        Calculate the number of grasps for each class, handedness, number per subject, etc.
        :return:
        """
        self.grasp_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.subject_counts = {}
        self.handedness_count = {0: 0, 1: 0}

    def __handle_validation_split__(
        self, validation_split: float, random_seed: int
    ) -> None:
        """
        Split the training samples into train and test
        :param validation_split:
        :param random_seed:
        :return:
        """
        pass

    def __get_sample_attrs_part__(
        self, sample_index: int, sample_attr: str = ""
    ) -> Any:
        """
        Get attributes for the sample
        :param sample_index:
        :param sample_attr:
        :return:
        """
        if (int(sample_index) > self.num_examples - 1) | (int(sample_index) < 0):
            raise ValueError(
                "sample_index "
                + str(sample_index)
                + " is out of range for dataset: [0,"
                + str(sample_index)
                + "]"
            )

        if not sample_attr:
            # Empty string, return the Group attributes as a dict
            return self.f[str(sample_index)].attrs

        else:
            if sample_attr not in [
                "label",
                "sample_name",
                "armReadyTime_ms",
                "objectTouched_ms",
                "lighting",
                "clutter",
                "handedness",
                "cameraid",
                "subjectid",
            ]:
                raise ValueError("Uknown requested sample_attr " + sample_attr)

            return self.f[str(sample_index)].attrs[sample_attr]

    def __get_sample_part__(self, sample_index: int, sample_part: str = ""):
        """
        Get parts (or all of) the sample
        :param sample_index:
        :param sample_part:
        :return:
        """
        if (int(sample_index) > self.num_examples - 1) | (int(sample_index) < 0):
            raise ValueError(
                "sample_index "
                + str(sample_index)
                + " is out of range for dataset: [0,"
                + str(sample_index)
                + "]"
            )

        if not sample_part:
            # Empty string, return the Group
            return self.f[str(sample_index)]

        else:
            if sample_part not in ["rgb", "depth", "accel", "gyro", "pose"]:
                raise ValueError("Unknown requested sample_part " + sample_part)

            return self.f[str(sample_index)][sample_part]

    def __handle_clutter__(self, clutter_allowed: bool) -> None:
        """
        If clutter is allowed, we leave the dataset alone. If clutter is disallowed, then we only take samples with clutter=0
        :param clutter_allowed:
        :return:
        """
        if not clutter_allowed:
            # Only samples with clutter (modify self.valid_sample_indicies)
            cluttered_indicies = self.sample_ids.attrs["cluttered_indicies"].split(",")
            self.valid_sample_indicies = list(
                np.setdiff1d(self.valid_sample_indicies, cluttered_indicies)
            )

    def __handle_lighting__(self, controlled_lighting: bool) -> None:
        """
        If controlled lighting is required, we only use samples with lighting=1
        :param controlled_lighting:
        :return:
        """
        if controlled_lighting:
            # Keep only samples with controlled lighting (modify self.valid_sample_indicies)
            lighting_indicies = self.sample_ids.attrs["lighting_indicies"].split(",")
            self.valid_sample_indicies = list(
                np.setdiff1d(self.valid_sample_indicies, lighting_indicies)
            )


class HandGenerator(object):
    def __init__(self, f: h5py.File, batch_size: int, im_size: Tuple = (240, 320)):
        self.f = f
        self.batch_size = batch_size
        self.greenscreen_labels = np.random.permutation(
            list(self.f["greenscreens"].keys())
        )
        self.num_greenscreens = len(self.greenscreen_labels)

        self.num_batches_before_repeat = int(
            np.floor(self.num_greenscreens / self.batch_size)
        )
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            while True:
                shuffled_labels = np.random.permutation(
                    self.greenscreen_labels
                )  # have to access with string values
                for batch_num in range(self.num_batches_before_repeat):
                    start_index = batch_num * self.batch_size
                    end_index = (batch_num + 1) * self.batch_size

                    batch_labels = shuffled_labels[start_index:end_index]
                    # yield zoom(np.asarray(compute([delayed(self.__hand_randomization__)(i) for i in batch_labels], get=threaded.get), dtype=np.uint8)[0], zoom=(1,0.5,0.5,1))
                    return np.asarray(
                        compute(
                            [
                                delayed(self.__hand_randomization__)(i)
                                for i in batch_labels
                            ],
                            get=threaded.get,
                        ),
                        dtype=np.uint8,
                    )[0]

    def __hand_randomization__(self, sample_label: str) -> np.ndarray:
        """
        Apply some basic transformations to the hand images (one at a time)

        """
        im = self.f["greenscreens"][sample_label]["image"]
        mask = self.f["greenscreens"][sample_label]["mask"]

        # Random shifts left and right (original hands are 620x340
        max_shift_lr = 35
        shift_lr = np.random.random_integers(-max_shift_lr, max_shift_lr)
        max_shift_up = 35
        shift_up = np.random.random_integers(
            -max_shift_up, 0
        )  # Negative moves hand up and off screen, Don't shift down at all

        im = shift(
            im, [shift_up, shift_lr, 0]
        )  # need the third dimension for the channels
        # print(mask.shape)
        mask = shift(mask, [shift_up, shift_lr, 0])

        # # Apply (slight) color modifications in HSV space
        # max_h_shift = 15 #Max is 179
        # max_s_shift = 15 #Max is 255
        # max_v_shift = 15 # Max is 255
        #
        # h_shift = np.random.randint(-max_h_shift,max_h_shift)
        # s_shift = np.random.randint(-max_s_shift,max_s_shift)
        # v_shift = np.random.randint(-max_v_shift,max_v_shift)
        #
        # h, s, v = cv2.split(cv2.cvtColor(im,cv2.COLOR_RGB2HSV))
        # lim_h = 179 - h_shift if h_shift > 0 else -h_shift
        # lim_s = 255 - s_shift if s_shift > 0 else -s_shift
        # lim_v = 255 - v_shift if v_shift > 0 else -v_shift
        # if h_shift > 0:
        #     h[h > lim_h] = 179
        #     h[h <= lim_h] += h_shift
        # else:
        #     h[h < lim_h] = 179
        #     h[h >= lim_h] -= np.uint8(abs(h_shift))
        #
        # if s_shift > 0:
        #     s[s > lim_s] = 255
        #     s[s <= lim_s] += s_shift
        # else:
        #     s[s < lim_s] = 255
        #     s[s >= lim_s] += np.uint8(abs(s_shift))
        #
        # if v_shift > 0:
        #     v[v > lim_v] = 255
        #     v[v <= lim_v] += v_shift
        # else:
        #     v[v < lim_v] = 255
        #     v[v >= lim_v] += np.uint8(abs(v_shift))
        #
        # im = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

        out = np.concatenate((im, np.expand_dims(mask[:, :, 0], axis=2)), axis=2)

        return out  # out is RGBA
