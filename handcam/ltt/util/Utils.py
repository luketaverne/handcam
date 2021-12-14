from time import sleep
import sys
import cv2
import numpy as np
import tensorflow as tf

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def write_progress_bar(current_step, total_steps):
    if current_step == 0:
        sys.stdout.write('\n') # two newlines, as the last one will be overwritten immediately
    sys.stdout.write("\r")
    percent = current_step/float(total_steps)
    progress = ""
    bar_len = 30
    arrow_placed = False
    for i in range(bar_len):
        if i < int(bar_len * percent):
            progress += "="
        elif not arrow_placed:
            progress += ">"
            arrow_placed = True
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%% (%s / %s frames)" % (progress, percent * 100, current_step, total_steps))
    sys.stdout.flush()

    if current_step == total_steps:
        print('\n')

def apply_alpha_matting(foreground, background, alpha):
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    depth = False

    if len(background.shape) == 2 or background.shape[-1] == 1:
        # background = np.expand_dims(background, axis=2)
        depth = True
        # foreground = np.zeros(shape=foreground.shape, dtype=foreground.dtype)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255
    if len(alpha.shape) == 2:
        alpha = np.expand_dims(alpha, axis=2)
    if alpha.shape[2] == 1 and not depth:
        alpha = np.asarray(np.concatenate((alpha,alpha,alpha), axis=2))

    # Multiply the foreground with the alpha matte
    # print(alpha.shape)
    # print(foreground.shape)
    # print(background.shape)
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    out_image = cv2.add(foreground, background)

    if depth:
        out_image = np.expand_dims(out_image, axis=2)

    return np.asarray(out_image, dtype=background.dtype)

def apply_alpha_matting_tf(foreground, background, alpha):
    # Convert uint8 to float
    foreground = tf.cast(foreground, tf.float32)
    background = tf.cast(background, tf.float32)

    depth = False

    if len(background.shape) == 2 or background.shape[-1] == 1:
        depth = True

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = tf.cast(alpha, tf.float32) / 255
    if len(alpha.shape) == 2:
        alpha = tf.expand_dims(alpha, axis=2)
    if alpha.shape[2] == 1 and not depth:
        alpha = tf.concat([alpha, alpha, alpha], axis=2)

    foreground = tf.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = tf.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    out_image = tf.add(foreground, background)

    if depth and out_image.shape[-1] != 1:
        out_image = tf.expand_dims(out_image, axis=2)

    if not depth:
        # Get rid of noise on rgb
        out_image = tf.cast(out_image, dtype=tf.uint8)

    return tf.cast(out_image, dtype=background.dtype)

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

def handcam_gesture_spotting_acc(results_dict):
    correct_spotted = 0

    test_predictions = results_dict['preds']
    test_correct_labels = results_dict['labels']
    is_sequence = True if len(test_predictions[0]) !=7 else False
    if is_sequence:
        frame_count = 0
        for sample_index in range(len(test_predictions)):
            for frame_index in range(len(test_predictions[sample_index])):
                frame_count+=1
                pred = test_predictions[sample_index][frame_index]
                label = np.argmax(test_correct_labels[sample_index][frame_index])
                if (pred == 5 and label == 5) or (pred != 5 and label != 5):
                    correct_spotted += 1

        spotting_accuracy = correct_spotted / float(frame_count)
        return spotting_accuracy
    else:
        for i in range(len(test_predictions)):
            if (np.argmax(test_correct_labels[i]) == 5 and np.argmax(test_predictions[i]) == 5) or ((np.argmax(test_correct_labels[i]) != 5 and np.argmax(test_predictions[i]) != 5)):
                correct_spotted += 1
        spotting_accuracy = correct_spotted / float(len(test_predictions))
        return spotting_accuracy