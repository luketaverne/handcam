import os, pickle
import numpy as np
import re
from textwrap import wrap
import itertools


from sklearn.metrics import jaccard_similarity_score, confusion_matrix
font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
import matplotlib

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

root_path = '/local/home/luke/datasets/handcam/handcam_results'
sample_name = 'handcam_seq60_rgbdoldloc_oldobj_clutter_results.pckl'

full_path = os.path.join(root_path, sample_name)
res_dict = {}

try:
    with open(full_path, "rb") as f:
        res_dict = pickle.load(f)
except IOError as e:
    print("Can't find file for %s" % sample_name)

# convert to non-one hot
# preds = np.asarray(res_dict['preds'])
# preds = res_dict['preds']
labels = []
preds = []

for label in res_dict['labels']:
    labels.extend(np.argmax(label, 1)) # [[6 6 6 3 3 3 6 6],[6 6 6 2 2 2 6],...]

for pred in res_dict['preds']:
    preds.extend(pred)

# labels = np.asarray(labels)

# print(labels)
# print(preds)

text_labels = ['grasp_1', 'grasp_2', 'grasp_3', 'grasp_4', 'grasp_5',
               'grasp_6', 'grasp_7']

cm = confusion_matrix(labels, preds, labels=range(len(text_labels)))

np.set_printoptions(precision=2)
###fig, ax = matplotlib.figure.Figure()

fig = matplotlib.figure.Figure(figsize=(14, 14), dpi=320, facecolor='w', edgecolor='k')
canvas = FigureCanvas(fig)
ax = fig.add_subplot(1, 1, 1)
ax.set_title(sample_name)
im = ax.imshow(cm, cmap='Oranges')

classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in text_labels]
classes = ['\n'.join(wrap(l, 40)) for l in classes]

tick_marks = np.arange(len(classes))

ax.set_xlabel('Predicted', fontsize=22)
ax.set_xticks(tick_marks)
c = ax.set_xticklabels(classes, fontsize=18, rotation=-90, ha='center')
ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_bottom()

ax.set_ylabel('True Label', fontsize=22)
ax.set_yticks(tick_marks)
ax.set_yticklabels(classes, fontsize=18, va='center')
ax.yaxis.set_label_position('left')
ax.yaxis.tick_left()

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=14,
            verticalalignment='center', color="black")
fig.set_tight_layout(True)

fig.savefig(sample_name.split(".pckl")[0] + "_confusion_plt.png")

del fig