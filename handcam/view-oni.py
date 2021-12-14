import cv2
import numpy as np
from handcam.ltt.datasets.handcam.OniProcessing import OniSampleReader
from handcam.ltt.datasets.handcam.OniProcessing import DatasetCreator

creator = DatasetCreator()

creator.add_new_samples(make_new=False)

oni = OniSampleReader("/local/home/luke/datasets/handcam/samples/130905-grasp_2/")

overlay = oni.view_depth_overlay()

while True:
    cv2.imshow("overlay", overlay.next())
    cv2.waitKey(34)
    while True:
        pass
    pass
