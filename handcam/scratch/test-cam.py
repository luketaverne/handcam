import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
import v4l2capture
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: ignore=E402


# Refernce for rgb stream:
# https://3dclub.orbbec3d.com/t/astra-pro-rgb-stream-using-openni/1015

#####
#
# Config options
#
#####
mode = "table"


#####
#
# Setup
#
#####

openni2.initialize("/home/luke/programming/OpenNI-Linux-x64-2.3/Redist")
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640, resolutionY=480, fps=30))

rgb_stream = dev.create_color_stream()
rgb_stream.start()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_YUYV, resolutionX=640, resolutionY=480, fps=30))

cap = cv2.VideoCapture("/dev/astra_mini_s")

video = v4l2capture.Video_device("/dev/astra_mini_s")
size_x, size_y = video.set_format(640,480,fourcc="MJPG")

print "device chose {0}x{1} res".format(size_x,size_y)


valid_modes = ["arm", "table"]

#####
#
# Sanity Checks
#
#####
assert(mode in valid_modes)

plt.figure()
plt.show(block=False)
i = 0
max_val = 65553


def rescale_depth(img):
    img = 255.0 * (img / 65535.0)
    # img = (img - 100) * (255.0/ (255.0 - 100.0))

    return img


def process_raw_frame(frame):
    frame_data = frame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img = rescale_depth(img)
    img = img.astype(np.uint8)
    img.shape = (1, 480, 640)
    img = np.concatenate((img, img, img), axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = np.squeeze(img)

    img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

    # print color_img.shape
    if mode == "table":
        img = cv2.flip(img, 0)

    return img


while i < 10000:
    frame = depth_stream.read_frame()
    depth_img = process_raw_frame(frame)

    ret, rgb_frame = cap.read()
    print ret

    cv2.imshow("rgb", rgb_frame)
    cv2.imshow("depth", depth_img)
    cv2.waitKey(34)
    i += 1

print("Max val found: ", max_val)
openni2.unload()
