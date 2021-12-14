import cv2
import numpy as np
import itertools
import os

# import h5py #put this back in after the h5py package has a new update on pip. See <https://github.com/h5py/h5py/issues/995>
import sys
from handcam.ltt.datasets.handcam.OrbbecCamParams import OrbbecCamParams

# from handcam.ltt.util.Utils import write_progress_bar
import glob

# from subprocess import Popen, PIPE
import read_oni_as_np


class OniSampleReader:
    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.vid_w = 320
        self.vid_h = 240
        self.is_valid_sample()

        # grasp_labels = ['grasp_1', 'grasp_2', 'grasp_3', 'grasp_4', 'grasp_5', 'grasp_6', 'grasp_7']
        grasp_label_to_id = {
            "grasp_1": 0,
            "grasp_2": 1,
            "grasp_3": 2,
            "grasp_4": 3,
            "grasp_5": 4,
            "grasp_6": 5,
            "grasp_7": 6,
        }

        self.grasp_id = grasp_label_to_id[self.grasp_label]

        self.cam_params = OrbbecCamParams(
            int(self.misc_attrs["cameraid"]), (self.vid_w, self.vid_h)
        )

        ret_tuple = read_oni_as_np.read_oni_as_np(
            os.path.join(self.sample_path, "video.oni"),
            self.cam_params.mat,
            self.grasp_id,
            self.misc_attrs["armReadyTime_ms"],
            self.misc_attrs["objectTouched_ms"],
        )

        self.vid = ret_tuple[0]
        self.frame_labels = ret_tuple[1]

    def __read_misc_txt__(self):
        with open(os.path.join(self.sample_path, "misc.txt"), "r") as file:
            misc_list = [line.strip() for line in file]

        self.misc_attrs = {}

        for line in misc_list:
            key = line.split(":")[0]
            value = line.split(":")[1]
            self.misc_attrs[key] = str(value)

        required_misc_properties = [
            u"armReadyTime_ms",
            u"objectTouched_ms",
            u"lighting",
            u"clutter",
            u"greenScreen",
            u"handedness",
            u"cameraid",
            u"subjectid",
        ]

        if set(required_misc_properties) != set(self.misc_attrs.keys()):
            raise ValueError("Sample is missing some required information in misc.txt")
        self.misc_attrs[u"armReadyTime_ms"] = int(self.misc_attrs[u"armReadyTime_ms"])
        self.misc_attrs[u"objectTouched_ms"] = int(self.misc_attrs[u"objectTouched_ms"])
        self.misc_attrs[u"cameraid"] = int(self.misc_attrs[u"cameraid"])

        if int(self.misc_attrs[u"cameraid"]) not in [1, 2]:
            raise ValueError(
                "Invalid camera selected. Please choose 1 (Luke's) or 2 (Matteo's)."
            )

        # Need to convert lightigng, clutter, greenScreen, handedness to boolean
        modify_misc_properties = [
            u"lighting",
            u"clutter",
            u"greenScreen",
            u"handedness",
        ]
        for prop in modify_misc_properties:
            if self.misc_attrs[prop] in [
                u"true",
                u"True",
                u"TRUE",
                u"right",
                u"Right",
                u"RIGHT",
            ]:
                # right handed will be 1
                self.misc_attrs[prop] = 1
            else:
                # false or left handed will be zero
                self.misc_attrs[prop] = 0

    def __read_accel_txt__(self):
        self.accel = np.genfromtxt(
            os.path.join(self.sample_path, "accel.txt"), skip_header=1, delimiter=","
        )
        if self.accel.shape[1] != 4:
            raise ValueError(
                "accel.txt has the wrong shape. Should be (None, 4) but is "
                + str(self.accel.shape)
            )

    def __read_gyro_txt__(self):
        self.gyro = np.genfromtxt(
            os.path.join(self.sample_path, "gyro.txt"), skip_header=1, delimiter=","
        )
        if self.gyro.shape[1] != 4:
            raise ValueError(
                "gyro.txt has the wrong shape. Should be (None, 4) but is "
                + str(self.gyro.shape)
            )

    def __read_pose_txt__(self):
        self.pose = np.genfromtxt(
            os.path.join(self.sample_path, "pose.txt"), skip_header=1, delimiter=","
        )
        if self.pose.shape[1] != 5:
            raise ValueError(
                "pose.txt has the wrong shape. Should be (None, 5) but is "
                + str(self.pose.shape)
            )

    def __process_Myo_data__(self):
        # Align the MyoData.
        # It seems that every time ANY of the 3 Myo things get new data, they all write to their buffer.

        # # We just need the time since recording started, so subtract the minimum timestamp
        min_timestamp = np.min([self.accel[:, 0], self.gyro[:, 0], self.pose[:, 0]])
        self.accel[:, 0] = self.accel[:, 0] - min_timestamp
        self.gyro[:, 0] = self.gyro[:, 0] - min_timestamp
        self.pose[:, 0] = self.pose[:, 0] - min_timestamp
        #
        # # Now we need to get rid of duplicate datapoints (rows)
        # self.accel = np.unique(self.accel, axis=0) # only removes exact duplicates, we still have some duplicate timestamps. Discuss what to do with them later
        # self.pose = np.unique(self.pose, axis=0)
        # self.gyro = np.unique(self.gyro, axis=0)

    def is_valid_sample(self):
        """
        Check if  data folder contains correct files with the correct properties.

        :return:
        """
        is_valid = True

        # has misc?
        self.__read_misc_txt__()
        # Has accel?
        self.__read_accel_txt__()
        # Has pose?
        self.__read_pose_txt__()
        # Has vel?
        self.__read_gyro_txt__()
        # Has video?
        if not os.path.isfile(os.path.join(self.sample_path, "video.oni")):
            IOError("video.oni doesn't exist for this sample")

        # Normalize the myo data
        self.__process_Myo_data__()

        # Make sure the video timestamps are there/are created
        # self.__read_timestamps_txt__()

        # Set the grasp_label (string)
        self.grasp_label = self.sample_path.split("-")[-1].split("/")[0]

        return True

    def getDepthHistogram(self, src):
        size = 256
        if src.dtype == np.uint16:
            size = 65536

        depthHistogram = np.zeros(
            (size), dtype=np.float
        )  # would be 65536 if we kept the 16-bit
        depthHist = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
        # depthHist = rgb
        number_of_points = 0

        for y, x in itertools.product(range(src.shape[1]), range(src.shape[0])):
            depth_cell = src[x, y]
            if depth_cell != 0:
                depthHistogram[depth_cell] += 1
                number_of_points += 1

        for nIndex in range(1, int(depthHistogram.shape[0])):
            depthHistogram[nIndex] += depthHistogram[nIndex - 1]

        for nIndex in range(1, int(depthHistogram.shape[0])):
            depthHistogram[nIndex] = (
                number_of_points - depthHistogram[nIndex]
            ) / number_of_points

        for y, x in itertools.product(range(src.shape[1]), range(src.shape[0])):
            depth_cell = src[x, y]
            depth_value = depthHistogram[depth_cell] * 255  # converting to uint8

            depthHist[x, y, 0] = 0
            depthHist[x, y, 1] = depth_value
            depthHist[x, y, 2] = depth_value
            # cv2.bitwise_or()

        return depthHist

    def get_depth_overlay(self, reverse_channels=False):
        rgb_vid = np.asarray(self.vid[..., 0:3], dtype=np.uint8)
        if reverse_channels:
            rgb_vid = np.rot90(rgb_vid, axes=(1, 2))
            rgb_vid = np.flip(rgb_vid, axis=2)
        vid = np.empty(shape=rgb_vid.shape, dtype=np.uint8)

        for i in range(self.vid.shape[0]):

            img = rgb_vid[i].copy()
            depth_hist = self.getDepthHistogram(self.vid[i, ..., 3:])
            if reverse_channels:
                depth_hist = depth_hist[..., ::-1]
                depth_hist = np.rot90(depth_hist)
                depth_hist = np.fliplr(depth_hist)

            # print(depth_img.dtype)
            # print(rgb_img.dtype)
            cv2.addWeighted(depth_hist, 0.5, img, 0.5, 0.5, img, dtype=cv2.CV_8UC3)
            vid[i] = img.copy()

        return vid
