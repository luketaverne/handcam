import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
import matplotlib

from handcam.ltt.datasets.handcam.OrbbecCamParams import OrbbecCamParams

matplotlib.use("TkAgg")
import scipy.misc
import matplotlib.pyplot as plt  # noqa: ignore=E402

openni2.initialize("/local/home/luke/programming/OpenNI-Linux-x64-2.3/Redist")
dev = openni2.Device.open_file(
    "/local/home/luke/datasets/handcam/samples/20180225/151719-grasp_4/video.oni"
)
# left-hand-white-screen.oni
dev.playback.set_speed(-1)

# Depth setup
depth_stream = dev.create_depth_stream()
depth_stream.start()
print(depth_stream.get_video_mode())
print(depth_stream.get_number_of_frames())
num_frames = depth_stream.get_number_of_frames()
# rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_YUYV, resolutionX=640, resolutionY=480, fps=30))

w = 320
h = 240


def process_depth_frame(frame):
    frame_data = frame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img.shape = (1, h, w)
    img = np.concatenate((img, img, img), axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = img[:, :, 0]

    img = np.fliplr(img)
    img = np.expand_dims(img, axis=2)

    return img


i = 0

while i < num_frames:
    dev.playback.seek(depth_stream, i)
    depth_frame = depth_stream.read_frame()
    depth_image = process_depth_frame(depth_frame)

    # plt.hist(np.ndarray.flatten(depth_image[depth_image != 0]), bins=100, range=(0, 65535))
    # plt.show()

    depth_im_cutoff_zerod = depth_image.copy()
    depth_im_cutoff_zerod[depth_image > 15000] = 0
    depth_im_cutoff_maxed = depth_image.copy()
    depth_im_cutoff_maxed[depth_image > 15000] = 65535

    out_im = np.concatenate(
        (depth_im_cutoff_zerod, depth_im_cutoff_zerod, depth_im_cutoff_maxed), axis=2
    )

    # out_im[0:10,0:10,0:3] = 0
    # out_im[0:10, 0:10, 2] = 65535

    scipy.misc.imsave("depth_im_replace_over_15000.png", out_im, format="png")

    cv2.imshow("depth", out_im)
    cv2.waitKey(34)
    i += 1
    while True:
        pass
openni2.unload()
