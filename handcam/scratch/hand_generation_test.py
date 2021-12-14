from handcam.ltt.datasets.handcam.HandCamDataHandler import Handler
from handcam.ltt.datasets.uw_rgbd.UWDataHandler import TrainGenerator
from handcam.ltt.datasets.uw_rgbd.UWDataHandler import Handler as UWHandler
from handcam.ltt.util.Preprocessing import DataAugmentation

import numpy as np
import cv2

handler = Handler()
uw_handler = UWHandler()

hand_generator = handler.hand_generator(32)
# old_train_generator = uw_handler.trainGenerator(32)

train_generator = TrainGenerator(
    f=uw_handler.f,
    train_indicies=uw_handler.train_indicies,
    num_classes=uw_handler.num_classes,
    batch_size=32,
)

data_augmentation = DataAugmentation(
    image_generator=train_generator,
    hand_generator=hand_generator,
    simple_crop=(224, 224),
    batch_size=32,
)

# print(type(next(np.asarray(hand_generator))))

out_im, out_mask, _ = next(data_augmentation)
# out = uw_handler.f['data'][0]
print(out_mask.shape)
print(out_mask.dtype)


# out_im = out[0,:,:,0:3]
# out_mask = out[0,...,3]

cv2.imshow("frame", out_im[0])
# cv2.imshow('frame',np.expand_dims(out_mask, axis=2))
cv2.waitKey(34)

while True:
    pass
