import pickle
import sys

import numpy as np

with open('/home/luke/github/master-thesis/python/all_validations.pckl', "rb") as f:
    validation_results = pickle.load(f)

with open('/home/luke/github/master-thesis/python/all_validations_imu.pckl', "rb") as f:
    imu_validation_results = pickle.load(f)

with open('/home/luke/github/master-thesis/python/all_test_cases.pckl', "rb") as f:
    test_results = pickle.load(f)

# results_dict[resnet_type][model_type][test_type]['split%d'][modality]['accuracy'/'gesture_spotting']

# validation sets format
#               rgb     depth   rgbd
# resnet-50
# resnet-18
start_split = 0
end_split = 10

def print_results_table(table_label, model_types, resnet_types, acc_or_gesture, test_types=None, with_IMU=False):
    # sanity checks
    for model_type in model_types:
        assert model_type in ['single_frames', 'sequence_frozen', 'sequence_end2end']

    for resnet_type in resnet_types:
        assert resnet_type in ['resnet-50', 'resnet-18']

    assert acc_or_gesture in ['accuracy', 'gesture_spotting']

    modalities = ['rgbd-imu'] if with_IMU else ['rgb', 'depth', 'rgbd']

    print(table_label)

    if test_types is not None:
        for test_type in test_types:
            assert test_type in ["oldloc_newobj_single", "newloc_oldobj_single", "newloc_newobj_single"]

        for test_type in test_types:
            for model_type in model_types:
                for resnet_type in resnet_types:
                    if model_type == 'sequence_end2end' and resnet_type == 'resnet-50':
                        continue
                    for modality in ['rgb', 'depth', 'rgbd']:
                        results = []

                        for split_id in range(start_split, end_split):
                            results.append(
                                test_results[resnet_type][model_type][test_type]['split%d' % split_id][modality][acc_or_gesture])

                        assert len(results) == end_split - start_split
                        out_mean = 100 * float(np.mean(results))
                        out_std = 100 * float(np.std(results))
                        sys.stdout.write("$%.2f \pm %.2f$" % (out_mean, out_std))

                        if not (resnet_type == 'resnet-18' and model_type == 'sequence_end2end' and modality == 'rgbd'):
                            sys.stdout.write(" & ")

            sys.stdout.write("\n\n")
            sys.stdout.flush()
    else:


        for model_type in model_types:
            for resnet_type in resnet_types:
                if model_type == 'sequence_end2end' and resnet_type == 'resnet-50':
                    continue
                for modality in ['rgb', 'depth', 'rgbd']:
                    results = []

                    for split_id in range(start_split, end_split):
                        results.append(validation_results[resnet_type][model_type]['split%d' % split_id][modality][acc_or_gesture])

                    assert len(results) == end_split - start_split
                    out_mean = 100*float(np.mean(results))
                    out_std = 100*float(np.std(results))
                    sys.stdout.write("$%.2f \pm %.2f$" % (out_mean, out_std))

                    if modality != 'rgbd':
                        sys.stdout.write(" & ")

                sys.stdout.write("\n\n")
                sys.stdout.flush()
    return

# rows first

print_results_table(table_label="Single frames accuracy:",
                    model_types=['single_frames'],
                    resnet_types=['resnet-50', 'resnet-18'],
                    acc_or_gesture='accuracy')

print_results_table(table_label="Single frames gesture spotting:",
                    model_types=['single_frames'],
                    resnet_types=['resnet-50', 'resnet-18'],
                    acc_or_gesture='gesture_spotting')

print_results_table(table_label="Sequence validation accuracy:",
                    model_types=['sequence_frozen', 'sequence_end2end'],
                    resnet_types=['resnet-50', 'resnet-18'],
                    acc_or_gesture='accuracy')

print_results_table(table_label="Sequence validation gesture spotting:",
                    model_types=['sequence_frozen', 'sequence_end2end'],
                    resnet_types=['resnet-50', 'resnet-18'],
                    acc_or_gesture='gesture_spotting')

# Now testing sets.
print_results_table(table_label="Test set accuracy:",
                    model_types=['sequence_end2end'],
                    resnet_types=['resnet-18'],
                    acc_or_gesture='accuracy',
                    test_types=["oldloc_newobj_single", "newloc_oldobj_single", "newloc_newobj_single"])

print_results_table(table_label="Test set gesture spotting:",
                    model_types=['sequence_end2end'],
                    resnet_types=['resnet-18'],
                    acc_or_gesture='gesture_spotting',
                    test_types=["oldloc_newobj_single", "newloc_oldobj_single", "newloc_newobj_single"])
