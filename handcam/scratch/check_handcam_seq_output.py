import numpy as np
import pickle
import os

model_dir = '/local/home/luke/datasets/handcam/'

predictions = []
sample_ids = []
sample_names = []

with open("/local/home/luke/predictions.pckl", "rb") as f:
    predictions = pickle.load(f)

with open("/local/home/luke/labels.pckl", "rb") as f:
    sample_ids = pickle.load(f)

with open("/local/home/luke/sample_names.pckl", "rb") as f:
    sample_names = pickle.load(f)

# check_id = 15
for check_id in range(0,100):
    print(sample_names[check_id])
    print("Pred: \n\t" + str(predictions[check_id]))
    print("Labels:\n\t" + str(np.argmax(sample_ids[check_id], axis=1)))