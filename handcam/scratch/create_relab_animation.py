import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
from handcam.ltt.datasets.handcam.OniProcessingCpp import OniSampleReader
from random import shuffle
import scipy.misc


sample_dirs = glob.glob("/local/home/luke/datasets/handcam/samples/*/*/")
mosaic_dir = "/local/home/luke/datasets/handcam/mosaic"

shuffle(sample_dirs)

print("found %d oni files" % len(sample_dirs))

im_num = 0

for sample in sample_dirs:
    oni = OniSampleReader(sample)

    frame_chosen = np.random.randint(
        int(oni.vid.shape[0] * 0.15), int(oni.vid.shape[0] * 0.95)
    )

    rgb_frame = np.asarray(oni.vid[frame_chosen, ..., 0:3], dtype=np.uint8)

    rgb_frame = np.rot90(rgb_frame, 1, axes=(0, 1))

    scipy.misc.imsave(os.path.join(mosaic_dir, str(im_num) + ".png"), rgb_frame)

    im_num += 1

#
# for i in range(oni.vid.shape[0]):
#     cv2.imshow('frame', rgb[i])
#     cv2.waitKey(34)
# plt.imshow(oni.vid[i,...,0:3])
# print(oni.vid[i,...,0])
# plt.show()
