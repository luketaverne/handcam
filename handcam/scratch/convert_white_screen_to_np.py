import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
import matplotlib
from scipy.ndimage.interpolation import shift

from handcam.ltt.datasets.handcam.OrbbecCamParams import OrbbecCamParams

matplotlib.use("TkAgg")
import scipy.misc
import matplotlib.pyplot as plt  # noqa: ignore=E402

openni2.initialize("/local/home/luke/programming/OpenNI-Linux-x64-2.3/Redist")
left_or_right = "right2"
dev = openni2.Device.open_file(
    "/local/home/luke/datasets/handcam/calibration/"
    + left_or_right
    + "-hand-white-screen.oni"
)
# left-hand-white-screen.oni
dev.playback.set_speed(-1)

# Depth setup
rgb_stream = dev.create_color_stream()
rgb_stream.start()
print(rgb_stream.get_video_mode())
print(rgb_stream.get_number_of_frames())
num_frames = rgb_stream.get_number_of_frames()
# rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_YUYV, resolutionX=640, resolutionY=480, fps=30))

w = 640
h = 480


def process_rgb_frame(frame):
    frame_data = frame.get_buffer_as_uint8()
    img = np.frombuffer(frame_data, dtype=np.uint8)
    img.shape = (480, 640, 3)

    return img


i = 0

problem_indices = [
    43,
    44,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    60,
    61,
    89,
    92,
    93,
    94,
    95,
    114,
    128,
]
problem_indices += list(range(96, 127))  # 96-126
problem_indices += list(range(159, 194))  # 159-193

print(problem_indices)

while i < num_frames:
    thresh = 135
    # i = 118
    dev.playback.seek(rgb_stream, i)
    rgb_frame = rgb_stream.read_frame()
    rgb_img_o = process_rgb_frame(rgb_frame)

    # scipy.misc.imsave('rgb' + str(i), rgb_img, format="png")
    # scipy.misc.imsave('depth' + str(i), depth_img, format="png")
    rgb_img = cv2.cvtColor(rgb_img_o, cv2.COLOR_RGB2BGR)

    tmp = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    if left_or_right == "right2":

        _, alpha = cv2.threshold(tmp, thresh, 255, cv2.THRESH_BINARY_INV)

        alpha[:, 0:199] = 0  # empirically found

        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Here, the image is pretty good for "object" part of trimark

        # Now need to close the top of the image so we can do contour detection.
        alpha_closed = np.copy(alpha)
        alpha_closed[0, :] = 0

        # Now find the contours of the hand.
        _, contours, hierarchy = cv2.findContours(
            alpha_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        masked = cv2.bitwise_and(rgb_img, rgb_img, mask=alpha)

        cv2.drawContours(alpha_closed, contours, -1, (255 / 2, 255 / 2, 255 / 2), 10)

        # Replace border pixels
        min_y = np.where(alpha[1, :] > 0)[0][0]
        max_y = np.where(alpha[1, :] > 0)[0][-1]

        alpha_closed[0:10, min_y + 10 : max_y - 10] = 255

        if i in problem_indices:
            shift_amount = -25
            alpha_closed = shift(alpha_closed, [shift_amount, 0])
            alpha_closed[shift_amount:, :] = 0
            rgb_img_o = shift(rgb_img_o, [shift_amount, 0, 0])
            rgb_img_o[shift_amount:, :] = 255
        else:
            shift_amount = -10
            alpha_closed = shift(alpha_closed, [shift_amount, 0])
            alpha_closed[shift_amount:, :] = 0
            rgb_img_o = shift(rgb_img_o, [shift_amount, 0, 0])
            rgb_img_o[shift_amount:, :] = 255

        scipy.misc.imsave(str(i) + "-right2-trimap.png", alpha_closed, format="png")
        scipy.misc.imsave(str(i) + "-right2-image.png", rgb_img_o, format="png")
        # if i in problem_indices:
        # cv2.imshow("overlay", alpha)
        # cv2.waitKey(34)
        # while True:
        #     pass
        i += 1
    elif left_or_right == "left":
        _, alpha = cv2.threshold(tmp, 125, 255, cv2.THRESH_BINARY_INV)

        # alpha[:, 0:199] = 0  # empirically found

        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Here, the image is pretty good for "object" part of trimark

        # Now need to close the top of the image so we can do contour detection.
        alpha_closed = np.copy(alpha)
        alpha_closed[0, :] = 0

        # Now find the contours of the hand.
        _, contours, hierarchy = cv2.findContours(
            alpha_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        masked = cv2.bitwise_and(rgb_img, rgb_img, mask=alpha)

        cv2.drawContours(alpha_closed, contours, -1, (255 / 2, 255 / 2, 255 / 2), 10)

        # Replace border pixels
        min_y = np.where(alpha[1, :] > 0)[0][0]
        max_y = np.where(alpha[1, :] > 0)[0][-1]

        alpha_closed[0:10, min_y + 10 : max_y - 10] = 255

        # scipy.misc.imsave(str(i) + "-left-trimap.png", alpha_closed, format="png")
        # scipy.misc.imsave(str(i) + "-left-image.png", rgb_img_o, format="png")

        cv2.imshow("overlay", masked)
        cv2.waitKey(34)
        i += 1

# print("Max val found: ", max_val)
openni2.unload()
