import cv2
import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
import matplotlib

from handcam.ltt.datasets.handcam.OrbbecCamParams import OrbbecCamParams

matplotlib.use("TkAgg")
import scipy.misc
import matplotlib.pyplot as plt  # noqa: ignore=E402

# Refernce for rgb stream:
# https://3dclub.orbbec3d.com/t/astra-pro-rgb-stream-using-openni/1015

#####
#
# Config options
#
#####
mode = "table"

w = 640
h = 480

# Camera Params
fx_d = 578.938
fy_d = 578.938
cx_d = 318.496
cy_d = 251.533

k1_d = -0.094342
k2_d = 0.290512
p1_d = -0.299526
p2_d = -0.000318
k3_d = -0.000279

cam_matrix_d = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])

dist_d = np.array([k1_d, k2_d, p1_d, p2_d, k3_d])

newcameramtx_d, roi_d = cv2.getOptimalNewCameraMatrix(
    cam_matrix_d, dist_d, (w, h), 1, (w, h)
)

fx_rgb = 517.138
fy_rgb = 517.138
cx_rgb = 319.184
cy_rgb = 229.077

k1_rgb = 0.044356
k2_rgb = -0.174023
p1_rgb = 0.077324
p2_rgb = 0.001794
k3_rgb = -0.003853

cam_matrix_rgb = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])

dist_rgb = np.array([k1_rgb, k2_rgb, p1_rgb, p2_rgb, k3_rgb])

newcameramtx_rgb, roi_rgb = cv2.getOptimalNewCameraMatrix(
    cam_matrix_rgb, dist_rgb, (w, h), 1, (w, h)
)

RR = np.array([[1, -0.003, 0.002], [0.003, 1, 0.005], [-0.002, -0.005, 1]])
RR2d = np.array([[1, -0.003], [0.003, 1]])
# RR = np.array([
#     [-0.003, 1 , 0.002],
#     [1, 0.003,0.005],
#     [-0.005, -0.002,1]
# ])

TT = np.array([-25.097, 0.288, -1.118])
TT2d = np.array([-25.097, 0.288])

RandT2d = np.zeros((2, 3))
RandT2d[0:2, 0:2] = RR2d
RandT2d[0:2, 2] = TT2d.transpose()

homog = np.zeros((4, 4))
homog[0:3, 0:3] = RR
homog[0:3, 3] = TT.transpose()
homog[3, 3] = 1

homog2 = np.zeros((3, 3))
homog2 = np.matmul(cam_matrix_d, RR)
# homog2[0:2,0:2] = np.array([[1,0],[0,1]])
# homog2[0:2,2] = TT[0:2].transpose()
# homog2[2,2] = 1

# R_d,R_rgb,P_d,P_rgb,_,_,_ = cv2.stereoRectify(cam_matrix_d,dist_d,cam_matrix_rgb,dist_rgb,(w,h),RR,TT)
# map_d1,map_d2=cv2.initUndistortRectifyMap(cam_matrix_d,dist_d,R_d,P_d,(w,h),cv2.CV_16SC2)
# map_rgb1,map_rgb2=cv2.initUndistortRectifyMap(cam_matrix_rgb,dist_rgb,R_rgb,P_rgb,(w,h),cv2.CV_16SC2)

R_d, R_rgb, P_d, P_rgb, _, _, _ = cv2.stereoRectify(
    cam_matrix_d,
    dist_d,
    cam_matrix_rgb,
    dist_rgb,
    (w, h),
    RR,
    TT,
    None,
    None,
    None,
    None,
    None,
    cv2.CALIB_ZERO_DISPARITY,
)
print(P_d)
print(P_rgb)
map_d1, map_d2 = cv2.initUndistortRectifyMap(
    cam_matrix_d, dist_d, R_d, P_d, (w, h), cv2.CV_32FC1
)
map_rgb1, map_rgb2 = cv2.initUndistortRectifyMap(
    cam_matrix_rgb, dist_rgb, R_rgb, P_rgb, (w, h), cv2.CV_32FC1
)

###
#
# Trying with homography
#
###


#####
#
# Setup
#
#####

openni2.initialize("/local/home/luke/programming/OpenNI-Linux-x64-2.3/Redist")
dev = openni2.Device.open_file(
    "/local/home/luke/datasets/handcam/150287-1-grasp_6/video.oni"
)

# Diagnostics: make sure we have some valid device
print(dev.get_device_info())
print(dev.has_sensor(c_api.OniSensorType.ONI_SENSOR_DEPTH))
print(dev.has_sensor(c_api.OniSensorType.ONI_SENSOR_COLOR))

dev.set_property(101, 1)
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
dev.set_depth_color_sync_enabled(True)
dev.playback.set_speed(-1)

# Depth setup
depth_stream = dev.create_depth_stream()
depth_stream.start()
# depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640, resolutionY=480, fps=30))

rgb_stream = dev.create_color_stream()
rgb_stream.start()
print(rgb_stream.get_video_mode())
print(rgb_stream.get_number_of_frames())
print(depth_stream.get_video_mode())
print(depth_stream.get_number_of_frames())
num_frames = rgb_stream.get_number_of_frames()
# rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_YUYV, resolutionX=640, resolutionY=480, fps=30))

valid_modes = ["arm", "table"]

#####
#
# Sanity Checks
#
#####
assert mode in valid_modes

plt.figure()
plt.show(block=False)
i = 1
max_val = 65553


def rescale_depth(img):
    img = 255.0 * (img / 65535.0)
    # img = (img - 100) * (255.0/ (255.0 - 100.0))

    return img


def process_rgb_frame(frame):
    frame_data = frame.get_buffer_as_uint8()
    img = np.frombuffer(frame_data, dtype=np.uint8)
    #    img = rescale_depth(img)
    # img = img.astype(np.uint8)
    img.shape = (480, 640, 3)
    #    img = np.concatenate((img, img, img), axis=0)
    #    img = np.swapaxes(img, 0, 2)
    #    img = np.swapaxes(img, 0, 1)
    # img = np.squeeze(img)

    #    img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

    # print color_img.shape
    #    if mode == "table":
    #        img = cv2.flip(img, 0)
    #
    # img = cv2.undistort(img, cam_matrix_rgb, dist_rgb, None)
    return img


def process_depth_frame(frame):
    frame_data = frame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    # img = rescale_depth(img)
    img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    # img = 255 - img
    img.shape = (1, 480, 640)
    img = np.concatenate((img, img, img), axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = img[:, :, 0]
    # img = np.squeeze(img)

    # img = cv2.normalize(img,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

    # print color_img.shape
    #    if mode == "table":
    #        img = cv2.flip(img, 0)
    #
    # img = cv2.undistort(img, cam_matrix_d, dist_d, None)
    return img


def create_depth_overlay(depth_img, rgb_img):
    # b_channel, g_channel, r_channel = cv2.split(img)
    #
    # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50  # creating a dummy alpha channel image.
    #
    # img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    # rgb_img = cv2.undistort(rgb_img, cam_matrix_rgb, dist_rgb, None)
    # depth_img = cv2.undistort(depth_img, cam_matrix_d, dist_d, None)
    #
    # depth_img = cv2.warpPerspective(depth_img, homog2, (w, h))
    # depth_img = depth_img

    # depth_img = cv2.remap(depth_img,map_d1,map_d2,cv2.INTER_LINEAR)
    # rgb_img = cv2.remap(rgb_img, map_rgb1, map_rgb2, cv2.INTER_LINEAR)

    # depth_img = cv2.warpAffine(depth_img,RandT2d,(w,h))

    # depth_img = np.multiply(depth_img, RR)

    img = rgb_img.copy()

    # depth_img = cv2.warpPerspective(depth_img,RR,(w,h))

    alpha = 0.8

    cv2.addWeighted(depth_img, alpha, img, 1 - alpha, 0, img)

    return img


dev.playback.set_speed(-1)
# first = False
# while True:
#     if not first:
#         first = True
#         dev.playback.seek(rgb_stream, i)
#
#         depth_frame = depth_stream.read_frame()
#         depth_img = process_depth_frame(depth_frame)
#
#         plt.hist(depth_img[:,:,0].ravel(), bins=100,bottom=1)
#         plt.show()


while i < num_frames:
    dev.playback.seek(rgb_stream, i)
    rgb_frame = rgb_stream.read_frame()
    rgb_img = process_rgb_frame(rgb_frame)
    depth_frame = depth_stream.read_frame()
    depth_img = process_depth_frame(depth_frame)

    # scipy.misc.imsave('rgb' + str(i), rgb_img, format="png")
    # scipy.misc.imsave('depth' + str(i), depth_img, format="png")

    # ret, rgb_frame = cap.read()
    # print ret

    # cv2.imshow("rgb", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    # cv2.imshow("depth", depth_img)
    cv2.imshow("overlay", create_depth_overlay(depth_img, rgb_img))
    # cv2.imshow("depth", depth_img)
    cv2.waitKey(34)
    i += 1

# print("Max val found: ", max_val)
openni2.unload()
