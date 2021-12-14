import cv2
import numpy as np
from handcam.ltt.datasets.handcam.OniProcessing import OniSampleReader
from handcam.ltt.datasets.handcam.OniProcessing import DatasetCreator

creator = DatasetCreator()

creator.add_new_samples(make_new=False)
