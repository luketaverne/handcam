import pickle
import numpy as np
import os
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix
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

single_nets_to_eval = [
    "handcam_wrn_rgb",
    "handcam_wrn_depth",
    "handcam_wrn_rgbd",
]

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

combined_nets = single_nets_to_eval + seq_nets_to_eval

for name in combined_nets:
    try:
        with open(os.path.join(root_dir, name + file_ending), "rb") as f:
            res_dict = pickle.load(f)
    except IOError as e:
        print("Can't find file for %s" % name)
        continue

    accuracy = res_dict['val_accuracy']

    print(name + ":\t%0.2f%%" % (accuracy * 100))

# Make something to check the seq-60 for choosing the correct grasp type


# Jaccard index
with open(os.path.join(root_dir, "handcam_seq60_rgbd" + file_ending), "rb") as f:
    res_dict = pickle.load(f)

preds = res_dict['preds'] # not one-hot
labels = res_dict['labels'] # one-hot

jaccard_scores = []

for full_seq_net_name in full_seq_nets:
    with open(os.path.join(root_dir, full_seq_net_name + file_ending), "rb") as f:
        res_dict = pickle.load(f)

    preds = res_dict['preds']  # not one-hot
    labels = res_dict['labels']  # one-hot

    for i in range(len(labels)):
        frame_preds = preds[i]
        frame_truth = np.argmax(labels[i], axis=1)

        jaccard_scores.append(jaccard_similarity_score(frame_truth, frame_preds))

    jaccard_mean = np.mean(jaccard_scores)
    jaccard_std = np.std(jaccard_scores)

    print(full_seq_net_name + " Jaccard:\t%0.3f +/- %0.3f" % (jaccard_mean, jaccard_std))
    # print("Jac std:\t%0.6f" % jaccard_std)

for name in (["handcam_seq15_rgb", "handcam_seq15_depth", "handcam_seq15_rgbd"] + single_nets_to_eval):
    with open(os.path.join(root_dir, name + file_ending), "rb") as f:
        res_dict = pickle.load(f)

    preds = res_dict['preds']  # one-hot
    labels = res_dict['labels']  # one-hot

    # print(labels[0])
    if name in single_nets_to_eval:
        print(np.argmax(preds[0], axis=0))
        print(np.argmax(labels[0], axis=0))


    normal_truth = []
    normal_preds = []

    for i in range(len(preds)):
        if name in single_nets_to_eval:
            normal_truth.append(np.argmax(labels[i], axis=0))
            normal_preds.append(np.argmax(preds[i], axis=0))
        else:
            normal_truth.append(np.argmax(labels[i][-1], axis=0))

    if name in single_nets_to_eval:
        # print(preds[])
        preds = normal_preds

    # print(normal_truth)
    # print(preds)
    text_labels = ['grasp_1', 'grasp_2', 'grasp_3', 'grasp_4', 'grasp_5',
                                                          'grasp_6', 'grasp_7']

    cm = confusion_matrix(normal_truth, preds, labels=range(len(text_labels)))

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(14, 14), dpi=320, facecolor='w', edgecolor='k')
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(name)
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

    fig.savefig(name + "_confusion_plt.png")