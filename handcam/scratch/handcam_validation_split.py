import glob
import os
import random
import pickle

random.seed(0)

split_percentage = 0.1

grasp_labels = ['grasp_1','grasp_2','grasp_3','grasp_4','grasp_5','grasp_6','grasp_7']

dataset_root = '/local/home/luke/datasets/handcam/'
dataset_samples = os.path.join(dataset_root, 'samples')

for split_id in range(0,10):
    train_filenames = []
    validation_filenames = []

    for grasp in grasp_labels:
        print('Working on ' + grasp)
        grasp_filenames = glob.glob(os.path.join(dataset_samples, '*','*' + grasp))
        for i in range(len(grasp_filenames)):
            # Extract the id: 20180312/154902_grasp0
            grasp_filenames[i] = grasp_filenames[i].split("samples/")[-1]
        print("Found %d files." % len(grasp_filenames))

        random.shuffle(grasp_filenames)
        random.shuffle(grasp_filenames)
        random.shuffle(grasp_filenames)

        last_val_index = int(len(grasp_filenames) * split_percentage)

        validation_filenames.extend(grasp_filenames[0:last_val_index])
        train_filenames.extend(grasp_filenames[last_val_index:])

    print("%d train files" % len(train_filenames))
    print("%d validation files" % len(validation_filenames))

    with open(os.path.join(dataset_root,'train_split%d.pckl' % split_id), "wb") as f:
        pickle.dump(train_filenames, f)

    with open(os.path.join(dataset_root,'validation_split%d.pckl' % split_id), "wb") as f:
        pickle.dump(validation_filenames, f)

    print(validation_filenames)

    print('saved pickles for split_%d' % split_id)

