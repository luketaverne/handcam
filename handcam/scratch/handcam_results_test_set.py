import pickle
import numpy as np
import os
from sklearn.metrics import jaccard_similarity_score, confusion_matrix
import sys
import matplotlib
import re
from textwrap import wrap
import itertools

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
import matplotlib

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

full_seq_nets = [
    "handcam_seq60_rgb",
    "handcam_seq60_depth",
    "handcam_seq60_rgbd",
]

seq_nets_to_eval =[
    "handcam_seq15_rgb",
    "handcam_seq15_depth",
    "handcam_seq15_rgbd"
] + full_seq_nets

root_dir = "/local/home/luke/datasets/handcam"
file_ending = "_results.pckl"

test_types = ["newloc_newobj_single", "newloc_newobj_clutter", "newloc_oldobj_clutter", "newloc_oldobj_single",
              "oldloc_newobj_single", "oldloc_newobj_clutter", "oldloc_oldobj_clutter"]

sys.stdout.write("\t")
for test_type in test_types:
    sys.stdout.write('%s\t' % test_type)

sys.stdout.write("\n")
sys.stdout.flush()

for name in seq_nets_to_eval:
    # print header first
    sys.stdout.write('%s\t' % name)

    for test_type in test_types:
        try:
            with open(os.path.join(root_dir, name + test_type + file_ending), "rb") as f:
                res_dict = pickle.load(f)
        except IOError as e:
            print("Can't find file for %s" % name)
            continue

        accuracy = res_dict['val_accuracy']

        sys.stdout.write("%0.2f\t" % (accuracy*100))
        # print(name + ":\t%0.2f%%" % (accuracy * 100))

    sys.stdout.write("\n")
    sys.stdout.flush()

# Make something to check the seq-60 for choosing the correct grasp type


# Jaccard index
# with open(os.path.join(root_dir, "handcam_seq60_rgbd" + file_ending), "rb") as f:
#     res_dict = pickle.load(f)
#
# preds = res_dict['preds'] # not one-hot
# labels = res_dict['labels'] # one-hot

sys.stdout.write("\n\n\nJaccard Scores\t")
for test_type in test_types:
    sys.stdout.write('%s\t\t' % test_type)

sys.stdout.write("\n")
sys.stdout.flush()


for name in full_seq_nets:
    # print header first
    sys.stdout.write('%s\t' % name)

    for test_type in test_types:
        try:
            with open(os.path.join(root_dir, name + test_type + file_ending), "rb") as f:
                res_dict = pickle.load(f)
        except IOError as e:
            print("Can't find file for %s" % name)
            continue

        preds = res_dict['preds']  # not one-hot
        labels = res_dict['labels']  # one-hot



        jaccard_scores = []

        for i in range(len(labels)):
            frame_preds = preds[i]
            frame_truth = np.argmax(labels[i], axis=1)

            jaccard_scores.append(jaccard_similarity_score(frame_truth, frame_preds))

        jaccard_mean = np.mean(jaccard_scores)
        jaccard_std = np.std(jaccard_scores)

        sys.stdout.write("%0.3f\t%0.3f\t" % (jaccard_mean, jaccard_std))
        # print(name + ":\t%0.2f%%" % (accuracy * 100))

    sys.stdout.write("\n")
    sys.stdout.flush()

for name in full_seq_nets:
    # print header first
    sys.stdout.write('%s\t' % name)

    for test_type in test_types:
        try:
            with open(os.path.join(root_dir, name + test_type + file_ending), "rb") as f:
                res_dict = pickle.load(f)
        except IOError as e:
            print("Can't find file for %s" % name)
            continue

        preds = res_dict['preds']  # not one-hot
        labels = res_dict['labels']  # one-hot
        normal_labels = []

        # if test_type == "newloc_newobj_clutter":
        #     for i in range(len(labels)):
        #         frame_truth = np.argmax(labels[i], axis=1)
        #         print("Preds: " + str(preds[i]))
        #         print("labels: " + str(frame_truth))
        #         print("")

            # normal_labels.append(frame_truth)

        # labels = np.argmax(labels, axis=1)

        # print("Preds: " + str(preds))
        # print("labels: " + str(normal_labels))

        # print(name + ":\t%0.2f%%" % (accuracy * 100))

    sys.stdout.write("\n")
    sys.stdout.flush()


for name in ["handcam_seq15_rgbd"]:
    for test_type in test_types:
        try:
            with open(os.path.join(root_dir, name + test_type + file_ending), "rb") as f:
                res_dict = pickle.load(f)
        except IOError as e:
            print("Can't find file for %s" % name)
            continue

        preds = res_dict['preds']  # not one-hot
        labels = res_dict['labels']  # one-hot

        normal_truth = []

        for i in range(len(labels)):
            normal_truth.append(np.argmax(labels[i][-1], axis=0))

        text_labels = ['grasp_1', 'grasp_2', 'grasp_3', 'grasp_4', 'grasp_5',
                                                              'grasp_6', 'grasp_7']

        cm = confusion_matrix(normal_truth, preds, labels=range(len(text_labels)))

        np.set_printoptions(precision=2)
        ###fig, ax = matplotlib.figure.Figure()

        fig = matplotlib.figure.Figure(figsize=(14, 14), dpi=320, facecolor='w', edgecolor='k')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(name + ": " + test_type)
        im = ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in text_labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=22)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=18, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=22)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=18, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=14, verticalalignment='center', color= "black")
        fig.set_tight_layout(True)

        fig.savefig(name + "_" + test_type + "_confusion_plt.png")

        del fig
