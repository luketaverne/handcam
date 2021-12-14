import cv2
import numpy as np
import itertools
from primesense import openni2
from primesense import _openni2 as c_api
import matplotlib

matplotlib.use("TkAgg")
import scipy.misc
import matplotlib.pyplot as plt  # noqa: ignore=E402
import os
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
# import h5py #put this back in after the h5py package has a new update on pip. See <https://github.com/h5py/h5py/issues/995>
import sys
from handcam.ltt.datasets.handcam.OrbbecCamParams import OrbbecCamParams
from handcam.ltt.util.Utils import write_progress_bar
import glob
from subprocess import Popen, PIPE
import depth_to_color_mapping


if sys.version_info[0] > 2:
    raise Exception("Must be using Python 2 for openni2 to work properly")


class OniSampleReader:
    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.is_valid_sample()

        openni2.initialize("/local/home/luke/programming/OpenNI-Linux-x64-2.3/Redist")
        self.dev = openni2.Device.open_file(os.path.join(sample_path, "video.oni"))
        self.dev.set_depth_color_sync_enabled(True)

        # -1 means we manually advance to the next frame by asking for a new frame
        self.dev.playback.set_speed(-1)

        # Depth setup
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()

        # RGB setup
        self.rgb_stream = self.dev.create_color_stream()
        self.rgb_stream.start()

        self.num_depth_frames = self.depth_stream.get_number_of_frames()
        self.num_rgb_frames = self.rgb_stream.get_number_of_frames()

        # Check that timestamps have the right number of values
        expected_timestamp_count = np.max([self.num_depth_frames, self.num_rgb_frames])
        if self.timestamps.shape[0] != expected_timestamp_count:
            raise ValueError(
                "Mismatch in timestamps.txt and frame count in the oni. Expected %d but got %d in timestamps.txt."
                % (expected_timestamp_count, self.timestamps.shape[0])
            )

        self.__get_vid_frame_size__()
        self.cam_params = OrbbecCamParams(
            int(self.misc_attrs["cameraid"]), (self.vid_w, self.vid_h)
        )

    def __read_misc_txt__(self):
        with open(os.path.join(self.sample_path, "misc.txt"), "r") as file:
            misc_list = [line.strip() for line in file]

        self.misc_attrs = {}

        for line in misc_list:
            key = line.split(":")[0]
            value = line.split(":")[1]
            self.misc_attrs[key.decode("utf-8")] = str(value).decode("utf-8")

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

    def __read_timestamps_txt__(self):
        c_exec_path = os.path.join(
            os.getcwd(), "/".join(__file__.split("/")[:-1]), "WriteOniFileTimestamps"
        )
        try:
            self.timestamps = np.genfromtxt(
                os.path.join(self.sample_path, "timestamps.txt"),
                skip_header=1,
                delimiter=",",
            )
        except IOError:
            # File doesn't exist, try to create it using the C++ program.
            process = Popen(
                [c_exec_path, os.path.join(self.sample_path, "video.oni")], stdout=PIPE
            )
            (output, err) = process.communicate()
            exit_code = process.wait()

            if exit_code == 0:
                # should be successful, try to read the file again
                try:
                    self.timestamps = np.genfromtxt(
                        os.path.join(self.sample_path, "timestamps.txt"),
                        skip_header=1,
                        delimiter=",",
                    )
                except IOError as e:
                    print("Error reading/generating timestamps for " + self.sample_path)
                    print("WriteOniFileTimestamps C++ error" + str(output))
                    raise (e)
            else:
                print("Unable to generate timestamps for " + self.sample_path)
                print("WriteOniFileTimestamps C++ error" + str(output))
                raise (IOError)

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

    def __get_vid_frame_size__(self):
        """
        Get the first frame of the RGB data to set the recorded frame size. Needed to properly set up the camera parameters.

        :return:
        """
        self.dev.playback.seek(self.rgb_stream, 0)
        rgb_frame = self.rgb_stream.read_frame()
        frame_data = rgb_frame.get_buffer_as_uint8()
        img = np.frombuffer(frame_data, dtype=np.uint8)

        # Writing this for two resolutions: 640x480 and 320x240
        try:
            img.shape = (240, 320, 3)
            self.vid_w = 320
            self.vid_h = 240
        except ValueError:
            # Can't do the reshape. Try 640x480
            try:
                img.shape = (480, 640, 3)
                self.vid_w = 640
                self.vid_h = 480
            except ValueError as e:
                print("Unexpected video resolution in " + self.sample_path)
                raise (e)

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
        self.__read_timestamps_txt__()

        # Set the grasp_label (string)
        self.grasp_label = self.sample_path.split("-")[-1].split("/")[0]

        return True

    def read_depth_stream(self):
        pass

    def read_color_stream(self):
        # Will use RGB order, remember to flip before displaying with OpenCV
        pass

    def process_depth_frame(self, frame):
        frame_data = frame.get_buffer_as_uint16()
        img = np.frombuffer(frame_data, dtype=np.uint16)
        img.shape = (1, self.cam_params.h, self.cam_params.w)
        img = np.concatenate((img, img, img), axis=0)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        img = img[:, :, 0]

        img = np.fliplr(img)
        img = np.expand_dims(img, axis=2)

        return img

    def process_color_frame(self, frame):
        frame_data = frame.get_buffer_as_uint8()
        img = np.frombuffer(frame_data, dtype=np.uint8)
        img.shape = (self.cam_params.h, self.cam_params.w, 3)
        img = np.fliplr(img)

        return img

    def convert_to_np(self):
        # For now, just make the video "length" the lesser of the two frame counts
        num_depth_frames = self.depth_stream.get_number_of_frames()
        num_rgb_frames = self.rgb_stream.get_number_of_frames()

        # shape -> [n_frames, width, height, channel (rgbd)]
        out_rgb = np.empty(
            (
                self.rgb_stream.get_number_of_frames(),
                self.cam_params.h,
                self.cam_params.w,
                3,
            ),
            dtype=np.uint8,
        )
        out_depth = np.empty(
            (
                self.depth_stream.get_number_of_frames(),
                self.cam_params.h,
                self.cam_params.w,
                1,
            ),
            dtype=np.uint16,
        )

        frame_generator = self.frame_generator()

        prog_bar_length = (
            np.max(
                [
                    self.depth_stream.get_number_of_frames(),
                    self.rgb_stream.get_number_of_frames(),
                ]
            )
            - 1
        )

        for i, (rgb, depth) in enumerate(frame_generator):
            if rgb is not None:
                out_rgb[i] = rgb
            if depth is not None:
                out_depth[i] = depth
            write_progress_bar(current_step=i, total_steps=prog_bar_length)

        return out_rgb, out_depth

    def frame_generator(self):
        test_difference = False

        for i in range(
            np.max(
                [
                    self.depth_stream.get_number_of_frames(),
                    self.rgb_stream.get_number_of_frames(),
                ]
            )
        ):
            rgb_img = None
            depth_img = None

            if i < self.depth_stream.get_number_of_frames() - 1:
                self.dev.playback.seek(self.depth_stream, i)
                depth_frame = self.depth_stream.read_frame()
                depth_img = self.process_depth_frame(depth_frame)

                depth_img = self.map_depth_cpp(
                    depth_img
                )  # relies on the cpp boost library in ltt.util (run ./compile.sh)

            if i < self.rgb_stream.get_number_of_frames() - 1:
                self.dev.playback.seek(self.rgb_stream, i)

                rgb_frame = self.rgb_stream.read_frame()
                rgb_img = self.process_color_frame(
                    rgb_frame
                )  # TODO: is this actually RGB ordering? Or BGR?

            yield (rgb_img, depth_img)

    def getDepthHistogram(self, src):
        size = 256
        if src.dtype == np.uint16:
            size = 65536

        depthHistogram = np.zeros(
            (size), dtype=np.float
        )  # would be 65536 if we kept the 16-bit
        depthHist = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
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

        return depthHist

    def mapping_depth_to_color(self, src):
        new_depth = np.zeros(src.shape, dtype=src.dtype)
        max_u = np.float64(new_depth.shape[1])
        max_v = np.float64(new_depth.shape[0])

        for u, v in itertools.product(range(src.shape[1]), range(src.shape[0])):
            d = src[v, u]

            if d == 0:
                continue

            z = np.double(d)

            u_rgb = (
                self.cam_params.mat[0] * np.double(u)
                + self.cam_params.mat[1] * np.double(v)
                + self.cam_params.mat[2]
                + self.cam_params.mat[3] / z
            )
            v_rgb = (
                self.cam_params.mat[4] * np.double(u)
                + self.cam_params.mat[5] * np.double(v)
                + self.cam_params.mat[6]
                + self.cam_params.mat[7] / z
            )

            if (u_rgb < 0) | (u_rgb >= max_u) | (v_rgb < 0) | (v_rgb >= max_v):
                continue
            # print(v_rgb)

            new_depth[np.uint16(v_rgb), np.uint16(u_rgb)] = d

        return new_depth

    def map_depth_cpp(self, src):
        new_depth = depth_to_color_mapping.map(src, self.cam_params.mat)

        return new_depth

    def map_depth_vid_cpp(self, src_vid):
        new_depth = depth_to_color_mapping.map(src_vid, self.cam_params.mat)

        return new_depth

    def apply_depth_colormap(self, depth_img, colormap=cv2.COLORMAP_HOT):
        if np.max(depth_img) > 255:
            # Need to normalize it first.
            depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0 / 65535.0))

        # if np.dtype(depth_img) != np.uint8:
        #     depth_img = np.uint8(depth_img)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_HOT)

        return depth_img

    def overlay_depth_frame(self, depth_img, rgb_img, alpha=0.8):
        img = rgb_img.copy()
        # print(depth_img.dtype)
        # print(rgb_img.dtype)
        cv2.addWeighted(depth_img, 0.5, img, 0.5, 0.5, img)

        return img

    def view_depth_overlay(self):
        i = 0

        while i < self.num_depth_frames:
            self.dev.playback.seek(self.rgb_stream, i)
            self.dev.playback.seek(self.depth_stream, i)
            rgb_frame = self.rgb_stream.read_frame()
            rgb_img = self.process_color_frame(rgb_frame)
            depth_frame = self.depth_stream.read_frame()
            depth_img = self.process_depth_frame(depth_frame)

            cali_depth = self.mapping_depth_to_color(depth_img)
            # colored_depth = self.apply_depth_colormap(cali_depth)

            # cali_depth = cali_depth.reshape((cali_depth.shape[0], cali_depth.shape[1], 1))
            # cali_depth = np.concatenate(
            #     (np.zeros((cali_depth.shape[0], cali_depth.shape[1], 1), dtype=np.uint8), cali_depth, cali_depth), axis=2)

            overlay_img = self.overlay_depth_frame(cali_depth, rgb_img)

            i += 1

            yield overlay_img


class DatasetCreator:
    """
    For converting the handcam dataset from oni to hdf5/numpy.


    """

    top_level_datasets_dir = "/local/home/luke/datasets"
    hdf5_path = os.path.join(top_level_datasets_dir, "handcam/handcam.hdf5")
    dataset_samples_path = os.path.join(top_level_datasets_dir, "handcam/samples")
    greenscreens_path = os.path.join(top_level_datasets_dir, "handcam/greenscreens")

    dataset_w = 320
    dataset_h = 240

    def __init__(self):
        pass

    def add_new_samples(self, make_new=False):
        if make_new:
            while True:
                response = raw_input(
                    "I'm going to overwrite the database. Is that really what you want? (Y/n): "
                )
                if response == "Y":
                    break
                elif response in ["n", "N"]:
                    print("Okay, quitting")
                    return False
                else:
                    print("Does not compute. Try again.")

            self.f = h5py.File(self.hdf5_path, "w")
            dt = h5py.special_dtype(vlen=bytes)
            self.sample_ids = self.f.create_dataset(
                "sample_ids",
                (1,),
                maxshape=(None,),
                dtype=dt,
                compression="gzip",
                shuffle=True,
                chunks=(1,),
            )
            self.sample_ids.attrs[
                u"cluttered_indicies"
            ] = u""  # need to specify the u in python2 or it makes problems in python3 later on.
            self.sample_ids.attrs[u"lighting_indicies"] = u""
            # TODO: add something to keep track of bad samples here
        else:
            self.f = h5py.File(self.hdf5_path, "r+")
            self.sample_ids = self.f["sample_ids"]

        # Add the greenscreens
        self.add_greenscreens()

        next_sample_index = 0  # start at 0
        if not make_new:
            next_sample_index = len(
                self.sample_ids
            )  # start with the next number in line

        sample_list = glob.glob(os.path.join(self.dataset_samples_path, "*", "*"))
        sample_list = [
            x.split("samples/")[-1] for x in sample_list
        ]  # 20180222/160565-grasp_1, etc.
        new_samples = []
        for sample_string in sample_list:
            if sample_string not in self.sample_ids:
                new_samples.append(sample_string)

        if len(new_samples) == 0:
            print("No new samples to add. Exiting dataset creation.")
            return True

        # Resize sample_ids to hold the required number of samples
        new_size = next_sample_index + len(new_samples)
        self.sample_ids.resize((new_size,))

        # Keep track of sample ids for greenscreen, clutter, lighting
        cluttered_indicies = self.sample_ids.attrs[u"cluttered_indicies"].split(",")
        lighting_indicies = self.sample_ids.attrs[u"lighting_indicies"].split(",")

        if cluttered_indicies == [u""]:
            cluttered_indicies = []
        if lighting_indicies == [u""]:
            lighting_indicies = []

        for sample_string in new_samples:
            # new sample, add it to dataset
            oni = OniSampleReader(
                os.path.join(self.dataset_samples_path, sample_string)
            )  # TODO: wrap with try/except and record bad samples
            print("Adding sample " + str(next_sample_index) + " " + sample_string)

            self.sample_ids[next_sample_index] = sample_string.decode("utf-8")

            # Each new sample gets a group, using an index
            new_group = self.f.create_group(str(next_sample_index).decode("utf-8"))
            new_group.attrs[u"label"] = oni.grasp_label.decode("utf-8")
            new_group.attrs[u"sample_name"] = sample_string.decode("utf-8")

            for key, value in oni.misc_attrs.iteritems():
                # Add attributes to this group for:
                #      lighting, clutter, green_screen, armReadyTime_ms, objectTouched_ms, lighting, clutter, greenScreen,
                #      handedness, cameraid, subjectid
                new_group.attrs[key] = value  # already encoded like u'string'

            if new_group.attrs[u"lighting"] == True:
                lighting_indicies.append(str(next_sample_index).decode("utf-8"))
            if new_group.attrs[u"clutter"] == True:
                cluttered_indicies.append(str(next_sample_index).decode("utf-8"))

            # Each new sample needs a dataset for video, accel, gyro and pose
            rgb, depth = oni.convert_to_np()
            new_group.create_dataset(
                "rgb",
                data=rgb,
                chunks=(1, self.dataset_h, self.dataset_w, 3),
                dtype=rgb.dtype,
                compression="gzip",
                shuffle=True,
            )
            new_group.create_dataset(
                "depth",
                data=depth,
                chunks=(1, self.dataset_h, self.dataset_w, 1),
                dtype=depth.dtype,
                compression="gzip",
                shuffle=True,
            )
            new_group.create_dataset("timestamps", data=oni.timestamps)
            new_group.create_dataset(
                "accel", data=oni.accel, compression="gzip", shuffle=True
            )
            new_group.create_dataset(
                "gyro", data=oni.gyro, compression="gzip", shuffle=True
            )
            new_group.create_dataset(
                "pose", data=oni.pose, compression="gzip", shuffle=True
            )

            next_sample_index += 1

        # update the attributes for the whole dataset
        self.sample_ids.attrs[u"cluttered_indicies"] = u",".join(cluttered_indicies)
        self.sample_ids.attrs[u"lighting_indicies"] = u",".join(lighting_indicies)

        self.f.flush()
        self.f.close()

    def add_greenscreens(self):
        print("Adding greenscreens")
        try:
            self.greenscreens = self.f["greenscreens"]
        except KeyError:
            # No green screens yet, create the new group.
            self.greenscreens = self.f.create_group("greenscreens")

        greenscreen_list = glob.glob(
            os.path.join(self.greenscreens_path, "*image.png")
        )  # don't want masks and images
        greenscreen_list = [
            x.split("/")[-1].split("-image")[0] for x in greenscreen_list
        ]  # 166-right, 167-right, etc.
        nothing_new = True
        for greenscreen in greenscreen_list:
            # print(greenscreen)
            if greenscreen not in self.greenscreens.keys():
                nothing_new = False
                print("Adding " + greenscreen)
                temp_group = self.greenscreens.create_group(greenscreen)

                image = cv2.imread(
                    os.path.join(self.greenscreens_path, greenscreen + "-image.png")
                )
                mask = cv2.imread(
                    os.path.join(self.greenscreens_path, greenscreen + "-mask.png")
                )

                temp_group.create_dataset("image", data=image)
                temp_group.create_dataset("mask", data=mask)

        if nothing_new:
            print("No new greenscreens found")

    # def delete_sample(self, sampleid):
    #     while True:
    #         response = raw_input("I'm going to overwrite the sample %s. Is that really what you want? (Y/n): " % sampleid)
    #         if response == 'Y':
    #             break
    #         elif response in ['n', 'N']:
    #             print('Okay, quitting')
    #             return False
    #         else:
    #             print("Does not compute. Try again.")
    #
    #     self.f = h5py.File(self.hdf5_path, 'r+')
    #     self.sample_ids = self.f['sample_ids']
    #
    #     del self.f[self.sample_ids[sampleid]]
