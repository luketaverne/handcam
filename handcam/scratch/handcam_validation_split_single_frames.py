import glob
import os
import random
import pickle

random.seed(0)

split_percentage = 0.1

grasp_labels = ['grasp_1','grasp_2','grasp_3','grasp_4','grasp_5','grasp_6','grasp_7']

dataset_root = '/local/home/luke/datasets/handcam/single_frames/'
dataset_tfrecords = os.path.join(dataset_root, 'tfrecords')
train_filenames = []
validation_filenames = []

for grasp in grasp_labels:
    print('Working on ' + grasp)
    grasp_filenames = glob.glob(os.path.join(dataset_tfrecords, grasp, "*.tfrecord"))
    print("Found %d files." % len(grasp_filenames))

    random.shuffle(grasp_filenames)
    random.shuffle(grasp_filenames)
    random.shuffle(grasp_filenames)

    last_val_index = int(len(grasp_filenames) * split_percentage)

    validation_filenames.extend(grasp_filenames[0:last_val_index])
    train_filenames.extend(grasp_filenames[last_val_index:])


print("%d train files" % len(train_filenames))
print("%d validation files" % len(validation_filenames))

with open(os.path.join(dataset_root,'train_split0.pckl'), "wb") as f:
    pickle.dump(train_filenames, f)

with open(os.path.join(dataset_root,'validation_split0.pckl'), "wb") as f:
    pickle.dump(validation_filenames, f)

print('saved pickles')