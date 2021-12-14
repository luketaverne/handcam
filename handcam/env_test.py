def test():
    import numpy as np
    import timeit
    import itertools
    import cv2
    import tensorflow as tf

    from tensorflow.python.client import device_lib

    assert device_lib.list_local_devices()
