import subprocess

from handcam.ltt.util.TFTools import (
    _DatasetInitializerHook,
    shuffle_dataset,
    per_sequence_standardization,
    per_sequence_standardization_rgbd,
)

import tensorflow as tf
import glob
import sys
import numpy as np
import six
import os
import pickle
import datetime

# from handcam.ltt.network.model.Wide_ResNet import wide_resnet_tf_depth as resnet_model_rgbd
from handcam.ltt.network.model.Wide_ResNet import wide_resnet_tf as resnet_model
from handcam.ltt.network.model.RNNModel import LSTMModel as lstm_model
from handcam.ltt.util.Utils import AttrDict, handcam_gesture_spotting_acc

models_to_eval = {
    "/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgbd-imu/train/2018-09-05/23:33": None,
    "/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgbd-imu/train/2018-09-06/02:41": None,
    "/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgbd-imu/train/2018-09-06/05:56": None,
    "/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgbd-imu/train/2018-09-06/08:51": None,
    "/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgbd-imu/train/2018-09-06/11:48": None,
    "/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgbd-imu/train/2018-09-06/16:19": None,
    "/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgbd-imu/train/2018-09-06/18:59": None,
    "/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgbd-imu/train/2018-09-06/21:20": None,
    "/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgbd-imu/train/2018-09-07/00:59": None,
    "/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgbd-imu/train/2018-09-07/03:36": None
    # '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/depth/frozen_train/2018-08-20/19:40': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/depth/train/2018-08-20/22:25': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgbd/frozen_train/2018-08-20/21:32': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgbd/train/2018-08-21/00:01': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgb/frozen_train/2018-08-20/20:22': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/sequence_resnet-18/rgb/train/2018-08-20/23:09': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/single_frames_resnet-18/depth/train/2018-08-20/18:22': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/single_frames_resnet-18/rgbd/train/2018-08-20/19:08': None,
    # '/media/luke/hdd-3tb/models/handcam/split0/single_frames_resnet-18/rgb/train/2018-08-20/18:46': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/depth/frozen_train/2018-08-21/21:19': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/depth/train/2018-08-21/23:38': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgbd/frozen_train/2018-08-21/22:40': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgbd/train/2018-08-22/01:17': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgb/frozen_train/2018-08-21/21:57': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-18/rgb/train/2018-08-22/00:17': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-50/depth/frozen_train/2018-08-23/19:12': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-50/rgbd/frozen_train/2018-08-23/23:04': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/sequence_resnet-50/rgb/frozen_train/2018-08-23/20:16': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/single_frames_resnet-18/depth/train/2018-08-21/17:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/single_frames_resnet-18/rgbd/train/2018-08-21/18:44': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/single_frames_resnet-18/rgb/train/2018-08-21/18:14': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/single_frames_resnet-50/depth/train/2018-08-23/11:51': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/single_frames_resnet-50/rgbd/train/2018-08-23/13:36': None,
    # '/media/luke/hdd-3tb/models/handcam/split1/single_frames_resnet-50/rgb/train/2018-08-23/12:32': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/depth/frozen_train/2018-08-22/04:11': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/depth/train/2018-08-22/08:31': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgbd/frozen_train/2018-08-22/06:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgbd/train/2018-08-22/09:59': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgb/frozen_train/2018-08-22/04:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-18/rgb/train/2018-08-22/08:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-50/depth/frozen_train/2018-08-24/04:35': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-50/rgbd/frozen_train/2018-08-24/07:02': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/sequence_resnet-50/rgb/frozen_train/2018-08-24/05:26': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/single_frames_resnet-18/depth/train/2018-08-22/02:17': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/single_frames_resnet-18/rgbd/train/2018-08-22/03:35': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/single_frames_resnet-18/rgb/train/2018-08-22/02:57': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/single_frames_resnet-50/depth/train/2018-08-24/02:08': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/single_frames_resnet-50/rgbd/train/2018-08-24/03:37': None,
    # '/media/luke/hdd-3tb/models/handcam/split2/single_frames_resnet-50/rgb/train/2018-08-24/02:42': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/depth/frozen_train/2018-08-22/12:23': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/depth/train/2018-08-22/15:58': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgbd/frozen_train/2018-08-22/15:08': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgbd/train/2018-08-22/18:08': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgb/frozen_train/2018-08-22/13:59': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-18/rgb/train/2018-08-22/17:17': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-50/depth/frozen_train/2018-08-24/12:50': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-50/rgbd/frozen_train/2018-08-24/16:53': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/sequence_resnet-50/rgb/frozen_train/2018-08-24/14:32': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/single_frames_resnet-18/depth/train/2018-08-22/11:00': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/single_frames_resnet-18/rgbd/train/2018-08-22/11:48': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/single_frames_resnet-18/rgb/train/2018-08-22/11:16': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/single_frames_resnet-50/depth/train/2018-08-24/09:24': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/single_frames_resnet-50/rgbd/train/2018-08-24/11:23': None,
    # '/media/luke/hdd-3tb/models/handcam/split3/single_frames_resnet-50/rgb/train/2018-08-24/10:10': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/depth/frozen_train/2018-08-22/20:36': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/depth/train/2018-08-22/23:04': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgbd/frozen_train/2018-08-22/22:14': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgbd/train/2018-08-23/01:16': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgb/frozen_train/2018-08-22/21:36': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-18/rgb/train/2018-08-23/00:10': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-50/depth/frozen_train/2018-08-24/23:00': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-50/rgbd/frozen_train/2018-08-25/04:29': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/sequence_resnet-50/rgb/frozen_train/2018-08-25/01:49': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/single_frames_resnet-18/depth/train/2018-08-22/19:11': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/single_frames_resnet-18/rgbd/train/2018-08-22/19:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/single_frames_resnet-18/rgb/train/2018-08-22/19:30': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/single_frames_resnet-50/depth/train/2018-08-24/19:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/single_frames_resnet-50/rgbd/train/2018-08-24/21:43': None,
    # '/media/luke/hdd-3tb/models/handcam/split4/single_frames_resnet-50/rgb/train/2018-08-24/20:50': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/depth/frozen_train/2018-08-23/04:06': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/depth/train/2018-08-23/06:34': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgbd/frozen_train/2018-08-23/05:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgbd/train/2018-08-23/08:19': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgb/frozen_train/2018-08-23/05:13': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-18/rgb/train/2018-08-23/07:38': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-50/depth/frozen_train/2018-08-25/11:27': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-50/rgbd/frozen_train/2018-08-25/15:06': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/sequence_resnet-50/rgb/frozen_train/2018-08-25/13:11': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/single_frames_resnet-18/depth/train/2018-08-23/02:54': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/single_frames_resnet-18/rgbd/train/2018-08-23/03:39': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/single_frames_resnet-18/rgb/train/2018-08-23/03:22': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/single_frames_resnet-50/depth/train/2018-08-25/07:09': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/single_frames_resnet-50/rgbd/train/2018-08-25/09:37': None,
    # '/media/luke/hdd-3tb/models/handcam/split5/single_frames_resnet-50/rgb/train/2018-08-25/08:35': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/depth/frozen_train/2018-08-23/12:15': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/depth/train/2018-08-23/15:13': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgbd/frozen_train/2018-08-23/13:58': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgbd/train/2018-08-23/17:53': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgb/frozen_train/2018-08-23/12:50': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-18/rgb/train/2018-08-23/16:47': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-50/depth/frozen_train/2018-08-26/00:18': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-50/rgbd/frozen_train/2018-08-26/03:24': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/sequence_resnet-50/rgb/frozen_train/2018-08-26/01:13': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/single_frames_resnet-18/depth/train/2018-08-23/10:39': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/single_frames_resnet-18/rgbd/train/2018-08-23/11:38': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/single_frames_resnet-18/rgb/train/2018-08-23/11:06': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/single_frames_resnet-50/depth/train/2018-08-25/19:42': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/single_frames_resnet-50/rgbd/train/2018-08-25/21:27': None,
    # '/media/luke/hdd-3tb/models/handcam/split6/single_frames_resnet-50/rgb/train/2018-08-25/20:25': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/depth/frozen_train/2018-08-22/16:30': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/depth/train/2018-08-22/21:24': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgb/frozen_train/2018-08-22/18:23': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgb/train/2018-08-22/22:36': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgbd/frozen_train/2018-08-28/10:04': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-18/rgbd/train/2018-08-28/10:38': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-50/depth/frozen_train/2018-08-26/11:24': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-50/rgbd/frozen_train/2018-08-26/14:05': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/sequence_resnet-50/rgb/frozen_train/2018-08-26/12:13': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/single_frames_resnet-18/depth/train/2018-08-22/14:30': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/single_frames_resnet-18/rgb/train/2018-08-22/14:57': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/single_frames_resnet-18/rgbd/train/2018-08-28/09:21': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/single_frames_resnet-50/depth/train/2018-08-26/07:39': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/single_frames_resnet-50/rgbd/train/2018-08-26/09:30': None,
    # '/media/luke/hdd-3tb/models/handcam/split7/single_frames_resnet-50/rgb/train/2018-08-26/08:36': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/depth/frozen_train/2018-08-23/00:54': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/depth/train/2018-08-23/03:20': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgbd/frozen_train/2018-08-23/02:27': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgbd/train/2018-08-23/04:51': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgb/frozen_train/2018-08-23/01:27': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-18/rgb/train/2018-08-23/04:08': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-50/depth/frozen_train/2018-08-26/22:18': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-50/rgbd/frozen_train/2018-08-27/02:58': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/sequence_resnet-50/rgb/frozen_train/2018-08-26/23:59': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/single_frames_resnet-18/depth/train/2018-08-22/23:20': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/single_frames_resnet-18/rgbd/train/2018-08-23/00:30': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/single_frames_resnet-18/rgb/train/2018-08-22/23:55': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/single_frames_resnet-50/depth/train/2018-08-26/19:11': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/single_frames_resnet-50/rgbd/train/2018-08-26/21:03': None,
    # '/media/luke/hdd-3tb/models/handcam/split8/single_frames_resnet-50/rgb/train/2018-08-26/20:06': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/depth/frozen_train/2018-08-23/07:21': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/depth/train/2018-08-23/09:15': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgbd/frozen_train/2018-08-23/08:35': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgbd/train/2018-08-23/10:38': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgb/frozen_train/2018-08-23/07:58': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-18/rgb/train/2018-08-23/09:49': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-50/depth/frozen_train/2018-08-27/07:32': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-50/rgbd/frozen_train/2018-08-27/12:23': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/sequence_resnet-50/rgb/frozen_train/2018-08-27/09:29': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/single_frames_resnet-18/depth/train/2018-08-23/05:40': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/single_frames_resnet-18/rgbd/train/2018-08-23/06:58': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/single_frames_resnet-18/rgb/train/2018-08-23/06:25': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/single_frames_resnet-50/depth/train/2018-08-27/04:49': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/single_frames_resnet-50/rgbd/train/2018-08-27/06:13': None,
    # '/media/luke/hdd-3tb/models/handcam/split9/single_frames_resnet-50/rgb/train/2018-08-27/05:31': None
}

# dict[resnet_size][seq_or_single][modality][gesture_or_accuracy]
compiled_results_dict = {
    "resnet-50": {"single_frames": {}, "sequence_frozen": {}},
    "resnet-18": {"single_frames": {}, "sequence_frozen": {}, "sequence_end2end": {}},
}

for resnet_type in compiled_results_dict.keys():
    for model_type in compiled_results_dict[resnet_type].keys():
        for split_id in range(0, 10):
            compiled_results_dict[resnet_type][model_type]["split%d" % split_id] = {}
            if split_id == 0 and resnet_type == "resnet-50":
                if model_type == "single_frames":
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgb"] = {"accuracy": 0.9818, "gesture_spotting": 0.9945}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["depth"] = {"accuracy": 0.8604, "gesture_spotting": 0.9394}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgbd"] = {"accuracy": 0.9813, "gesture_spotting": 0.9924}
                else:
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgb"] = {"accuracy": 0.9540, "gesture_spotting": 0.9922}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["depth"] = {"accuracy": 0.9461, "gesture_spotting": 0.9864}
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ]["rgbd"] = {"accuracy": 0.9602, "gesture_spotting": 0.9919}
                # need to put the numbers from the paper in here, models are gone.
                pass
            else:
                for modality in ["depth", "rgb", "rgbd"]:
                    compiled_results_dict[resnet_type][model_type][
                        "split%d" % split_id
                    ][modality] = {"accuracy": None, "gesture_spotting": None}


# run for each model. Load the FLAGS.pckl file to set everything up the same way as for training
for model_path in models_to_eval.keys():
    checkpoint_id = models_to_eval[model_path]

    if (
        "/tmp/luke/handcam" not in model_path
        and "/media/luke/hdd-3tb" not in model_path
    ):
        model_path = os.path.join("/tmp/luke/handcam/", model_path)

    with open(os.path.join(model_path, "FLAGS.pckl"), "rb") as f:
        flags_dict = pickle.load(f)
        FLAGS = AttrDict(flags_dict)  # allow attribute access to FLAGS.

    # need to modify FLAGS a bit
    FLAGS.batch_size = 1

    # Sanity check FLAGS
    if FLAGS.input_modality not in ["rgb", "rgbd", "depth"]:
        raise (ValueError("input_modality must be one of: rgb, rgbd, depth."))
    else:
        print(FLAGS.input_modality)

    if FLAGS.model_type not in ["single_frames", "sequence"]:
        raise (ValueError("model_type must be one of: single_frames, sequence."))

    if FLAGS.mode not in ["train", "eval", "frozen_train"]:
        raise (ValueError("mode must be one of: train, eval, frozen_train"))

    if FLAGS.resnet_size not in [18, 50]:
        raise (ValueError("resnet size must be one of: 18, 50"))

    with open(os.path.join(model_path, "results.pckl"), "rb") as f:
        out_dict = pickle.load(f)

    print(
        "per frame: %.2f\tgesture spotting: %.2f"
        % (100 * out_dict["val_accuracy"], 100 * out_dict["gesture_spotting_accuracy"])
    )

    dict_resnet_type = "resnet-%d" % FLAGS.resnet_size
    dict_model_type = FLAGS.model_type
    if FLAGS.model_type != "single_frames":
        dict_model_type = (
            "sequence_frozen" if FLAGS.mode == "frozen_train" else "sequence_end2end"
        )

    dict_split_name = "split%d" % FLAGS.validation_split_num
    dict_modality = FLAGS.input_modality

    compiled_results_dict[dict_resnet_type][dict_model_type][dict_split_name][
        dict_modality
    ] = {
        "accuracy": out_dict["val_accuracy"],
        "gesture_spotting": out_dict["gesture_spotting_accuracy"],
    }


with open("/home/luke/github/master-thesis/python/all_validations_imu.pckl", "wb") as f:
    pickle.dump(compiled_results_dict, f)
