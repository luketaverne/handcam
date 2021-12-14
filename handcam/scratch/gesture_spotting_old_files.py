import numpy as np
import pickle
from handcam.ltt.util.Utils import handcam_gesture_spotting_acc

test_list = [
    "handcam_seq60_rgb_results.pckl",
    "handcam_seq60_depth_results.pckl",
    "handcam_seq60_rgbd_results.pckl",
    "handcam_seq60_rgboldloc_oldobj_clutter_results.pckl",
    "handcam_seq60_rgboldloc_newobj_single_results.pckl",
    "handcam_seq60_rgboldloc_newobj_clutter_results.pckl",
    "handcam_seq60_rgbnewloc_oldobj_single_results.pckl",
    "handcam_seq60_rgbnewloc_oldobj_clutter_results.pckl",
    "handcam_seq60_rgbnewloc_newobj_single_results.pckl",
    "handcam_seq60_rgbnewloc_newobj_clutter_results.pckl",
    "handcam_seq60_deptholdloc_oldobj_clutter_results.pckl",
    "handcam_seq60_deptholdloc_newobj_single_results.pckl",
    "handcam_seq60_deptholdloc_newobj_clutter_results.pckl",
    "handcam_seq60_depthnewloc_oldobj_single_results.pckl",
    "handcam_seq60_depthnewloc_oldobj_clutter_results.pckl",
    "handcam_seq60_depthnewloc_newobj_single_results.pckl",
    "handcam_seq60_depthnewloc_newobj_clutter_results.pckl",
    "handcam_seq60_rgbdoldloc_oldobj_clutter_results.pckl",
    "handcam_seq60_rgbdoldloc_newobj_single_results.pckl",
    "handcam_seq60_rgbdoldloc_newobj_clutter_results.pckl",
    "handcam_seq60_rgbdnewloc_oldobj_single_results.pckl",
    "handcam_seq60_rgbdnewloc_oldobj_clutter_results.pckl",
    "handcam_seq60_rgbdnewloc_newobj_single_results.pckl",
    "handcam_seq60_rgbdnewloc_newobj_clutter_results.pckl",
]


for item in test_list:
    with open(item, "rb") as f:
        results = pickle.load(f)

    acc = 100 * handcam_gesture_spotting_acc(results)

    print(item + "\n%.2f\n\n" % acc)
