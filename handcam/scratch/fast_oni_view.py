import read_oni_as_np
from handcam.ltt.datasets.handcam.OrbbecCamParams import OrbbecCamParams
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle

cam_params = OrbbecCamParams(1, (320, 240))
# print(type(cam_params.mat))
def overlay_depth_frame(depth_img, rgb_img, alpha=0.8):
    img = rgb_img.copy()
    # print(depth_img.dtype)
    # print(rgb_img.dtype)
    cv2.addWeighted(depth_img, 0.5, img, 0.5, 0.5, img, dtype=cv2.CV_8UC3)

    return img


from handcam.ltt.datasets.handcam.OniProcessingCpp import OniSampleReader

# oni_file = "/local/home/luke/programming/master-thesis/python/scratch/142592-grasp_3" # camera moving
# oni_file = "/local/home/luke/programming/master-thesis/python/scratch/145850-grasp_5" # sascha arm
oni_file = (
    "/local/home/luke/datasets/handcam/samples/20180314/213884-grasp_7"  # sascha arm
)

read_pickle = False

if read_pickle:
    with open(oni_file + ".pckl", "rb") as f:
        vid = pickle.load(f)

    for frame in range(vid.shape[0]):
        cv2.imshow("depth", vid[frame])
        cv2.waitKey(32)
        img = cv2.flip(vid[frame], 0)
        img = np.rot90(img)
        cv2.imwrite(oni_file + "/vid/" + str(frame) + ".png", img)
        print("next")

else:

    print("before oni")
    oni = OniSampleReader(oni_file)
    print("after oni")

    vid, frame_labels = oni.vid, oni.frame_labels
    print(frame_labels)
    print(frame_labels.shape)
    print(vid.shape)

    out_vid = np.empty((vid.shape[0], vid.shape[1], vid.shape[2], 3), np.uint8)

    for frame in range(vid.shape[0]):
        # cv2.imshow('rgb', np.asarray(vid[frame,...,0:3], dtype=np.uint8))
        rgb = np.asarray(vid[frame, ..., 0:3], dtype=np.uint8)
        depth = oni.getDepthHistogram(vid[frame, ..., 3])
        # print(rgb.dtype)
        # print(depth.dtype)
        img = overlay_depth_frame(depth, rgb)
        out_vid[frame] = img
        # img = cv2.bitwise_or(rgb, depth)
        cv2.imshow("depth", img)
        cv2.waitKey(32)
        print("next")
        # print(timestamps[frame])
        # plt.imshow(res[frame,...,3])
        # plt.show()

    # with open(oni_file + ".pckl", "wb") as f:
    #     pickle.dump(out_vid, f)
