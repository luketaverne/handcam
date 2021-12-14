# Master Thesis Repository - Handcam Project
#### Luke T. Taverne
#### Sept 2017 - Apr 2018

## Note:
This is the entire repository for the master thesis, including scratch code and potentially unusable code. If you're getting start with the handcam dataset, I recommend looking at the `handcam/tensorflow/handcam` directory, in particular the `train_handcam_any.py` and `run_all_validations.py` files. Together with the [dataset](https://ait.ethz.ch/projects/2019/handcam/), I've bundled the network weights which were used to produce the results that are in the [paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=2FvqPwoAAAAJ&citation_for_view=2FvqPwoAAAAJ:2osOgNQ5qMEC), so please use this as a verification of the network setup.

You may use the `pyproject.toml` and `poetry.lock` files as a guideline for the project requirements, but they currently don't produce a working environment due to the old version of tensorflow used here & the newer hardware not being supported. The versions listed in this README should be considered definitive.


## Repository Structure
There are several folders containing the different aspects of the project:
- `android`
    - The data collection application and required dependencies.
- `config`
    - Contains some USB rules files for using the device with Linux (not important for thesis content)
- `cpp`
    - The C++ code for processing the oni videos, applying global matting for UW-hands.
- `datacapture`
    - Was the beginning of the C++ program for data collection on Linux Abandoned after the switch to Android, but maybe useful for future reference.
- `python`
    - Contains all code for processing data samples (Handcam and UW-hands), training with Tensorflow (and some in Keras which were also abandoned), evaluation and plotting scripts for everything including Handcam samples.
- `thirdparty`
    - References to thirdparty applications and libraries used in development.

## Software Versions
I will try to list below all key software used and which version it was. In general, everything was computed on `ait-server-03`. Not all of these are required for everything, for example some packages are just for making animated plots.

- Ubuntu Linux 16.04 (x86_64)
- Tensorflow `1.4.0`
- python `3.5.2`
    - numpy `1.14.1`
    - scipy `1.0.0`
    - matplotlib `2.1.0`
    - opencv `3.3.0`
    - imageio `2.1.2`
- cudnn `7.0`
- cuda `9.0.176`
- mkl-dnn `0.11.0`
- nvidia drivers `384.111` (using GeForce GTX 1080-ti cards)
- boost `1.66.0` (needed for the oni processing in C++)
- opencv `3.4.1` (needed for oni processing in C++)
- ffmpeg `2.8.14` (`3.4.2` on Mac)

## Setup of repository
Here I'll try to cover most of the important stuff you need to make the repo work.
### OpenNI2
This is important if you're going to work with the `*.oni` files directly. If you're just going to use the `TFRecord` files (which I highly recommend), then you can skip this part.
- Unzip the `2-Linux.zip` file, then get `OpenNI-Linux-x86-2.3.zip` from inside and unzip that. This is a special release of OpenNI2 modified by Orbbec to work with their camera. They also have a Github page that supposedly contains the same software, but I never got that to work.
- I moved `OpenNI-Linux-x86-2.3` to `~/programming/OpenNI-Linux-x86-2.3`. Change to this directory, then run `./install.sh`. If you're on the server, it will probably complain about root privileges because it's trying to install some USB rules that you don't need on the server. Open `install.sh` and comment out lines 26-29 and 36-39 (the check for root, and the copy of the usb rules into `/etc/udev/...`. Run it again.
- This should produce a file `OpenNIDevEnvironment` in the current directory, which contains definitions for `OPENNI2_INCLUDE` and `OPENNI2_REDIST`. You can either `source` this every time you want to use OpenNI things, or you can copy these definitions to your `.bashrc`.

That should be everything you need for running OpenNI2 for this project. If you install the `primesense` package for python2 (only! does not work with python3) with `pip2 install primesense`, then you can open the oni files in python. This is pretty slow if you're processing a dataset, but if you want to play with the files you'll be able to much quicker in python. You need to do:
```python
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np

# Openni and device setup
openni2.initialize("/path/to/openni2/redist")
dev = openni2.Device.open_file("/path/to/video.oni")
dev.playback.set_speed(-1) # -1 means no framerate, manually advance.

# Make depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()

# Make rgb stream
rgb_stream = dev.create_color_stream()
rgb_stream.start()

# Check number of frames if you want (will be different for depth and rgb)
print("Depth count: %s" % depth_stream.get_number_of_frames())
print("Rgb count: %s" % rgb_stream.get_number_of_frames())

# go to some frame
dev.playback.seek(depth_stream, some_frame_num)
dev.playback.seek(rgb_stream, some_frame_num)

# get the frame
depth_frame = depth_stream.read_frame()
rgb_frame = rgb_stream.read_frame()

# process the frames
depth_img = process_depth_frame(depth_frame)
rgb_img = process_color_frame(rgb_frame)

# depth_img and rgb_img are now in numpy format, have fun!


def process_depth_frame(depth_frame):
    # height and width are 240 and 320 if you're using my samples.
    depth_frame = depth_frame.get_buffer_as_uint16()
    depth_img = np.frombuffer(depth_frame, dtype=np.uint16)
    depth_img.shape = (1, FRAME_HEIGHT, FRAME_WIDTH)
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
    depth_img = np.swapaxes(depth_img, 0, 2)
    depth_img = np.swapaxes(depth_img, 0, 1)
    depth_img = depth_img[:, :, 0:1] # keep the trailing dimension.

    depth_img = np.fliplr(depth_img)

    return depth_img

 def process_color_frame(rgb_frame):
    # height and width are 240 and 320 if you're using my samples.
    rgb_frame = rgb_frame.get_buffer_as_uint8()
    rgb_img = np.frombuffer(rgb_frame, dtype=np.uint8)
    rgb_img.shape = (FRAME_HEIGHT, FRAME_WIDTH, 3)
    rgb_img = np.fliplr(rgb_img)

    return rgb_img

```
to import the library, get it set up, and convert the frames to numpy format. **Please note**: the depth frames are not calibrated/ aligned with RGB, you need to do this on your own (look in my python or c++ code for this), and in the python version of OpenNI2 you do not have access to the frame timestamps. Use caution when naïvely aligning rgb and depth using the frame timestamps, the framerates are not the same.

### Camera Setup
#### Filtering
The camera ships with the "enhanced filter" turned off. I didn't inspect the properties of the filter, but when it is off you get these annoying runs of blank rows and columns in the depth map, that dance around the image. The enhanced filter is controlled by a persistent software switch. An Orbbec employee sent me a Dropbox link to some Windows-only software to turn on the filter. Once the filter is turned on, it stays on until you reuse the tool to turn it off. I already did this for both of the RELab cameras. To use the tool:

- Unzip the `3-Windows.zip` file. This contains a pdf README and some other things. Unzip `Enhanced FilterTools... .zip`. Follow the instructions in the pdf to turn on the software filter (this is the enhanced filter, the hardware filter doesn't work very well and gives those lines mentioned earlier). You might need to install the sensor driver first (comes from the same folder).

#### Camera Parameters
The cameras are factory calibrated, you can read the parameters using some of the Orbbec tools:

- In `3-Windows.zip`, use `OpenNI2/OpenNI-Windows-x86-2.3.zip` and run the AstraViewer program with the camera attached via USB. Use the numerical keys to apply the depth mapping using the onboard camera parameters, and the parameters will be printed in the command window.

### Android things
The Android application things were designed to be used with Eclipse, which I could not for the life of me figure out how to get it to compile. I managed to port their example Android app over to Android Studio and it works pretty well. You should import the `android/DataCaptureProject` as an existing project in Android studio and then everything should work fine. The Orbbec demo lives in `android/NiViewerAndroid` if you want to investigate the original project structure/ try to figure out how to port it over yourself.

### Boost (`1.66.0 rc2`)
Make sure to build with C++11 support, specify `with-python` and `link=shared` when running the `bootstrap.sh` command, and **very important** you need to add a `-a` flag to most (all?) of the boost build steps. Otherwise it will compile but you will not be able to use the `boost-python` libraries without crashes/segfaults. Use python3 when specifying the python executable and libraries, and you should either use the python library from your `virtualenv`, or use a global one that has at least numpy installed.


### OpenCV 3.2
Compile OpenCV 3.4.1 with python support. There are plenty of resources for this online. Again, make sure to use C++11 support.

## "How do I ... ?"
### Create the TFRecord files
For Handcam:
- Single frames:
    - use `tensorflow/handcam/prepare_handcam_tfrecords_images.py`. You need the `cpp/read_oni_sample_as_np` extension installed and working.
- Sequences:
    - use `tensorflow/handcam/prepare_handcam_tfrecords.py`. You need the `cpp/read_oni_sample_as_np` extension installed and working.

For UW-hands:
- use `python/tensorflow/uw/prepare_UW_tfrecords_split_fast.py`. You'll need to have the `cpp/data_aug_uw` extension installed.
    - **NOTE**: This seems to have a memory leak, as it takes up hundreds of GBs of RAM before it finishes. It will work on the server as long as there is space for it. You can try to fix this (monitor RAM usage with `free -h`).

### Train a model
For now, all of the model trainings are in separate files. Ideally this would be done instead with a configuration file and good logs.

Handcam (`python/tensorflow/handcam/`):
- Single frames:
    - `train_handcamWRN_rgb.py`, `train_handcamWRN_depth.py`, `train_handcamWRN_rgbd.py`
    
- Seq-15 ("reaching phase" in the report)
    - `train_handcamWRN_fixedsequence_rgb.py`, `train_handcamWRN_fixedsequence_depth.py`, `train_handcamWRN_fixedsequence_rgbd.py`
- Seq-60 ("full sequence" in the report)
    - `train_handcamWRN_fixedsequence_all_rgb.py`, `train_handcamWRN_fixedsequence_all_depth.py`, `train_handcamWRN_fixedsequence_all_rgbd.py`
  
UW-hands (`python/tensorflow/uw/`):
- `train_UW_hands_rgb.py`, `train_UW_hands_depth.py`, `train_UW_hands_rgbd.py`

### Evaluate a model
Handcam (`python/tensorflow/handcam/`):
- `run_all_validations.py` for validation set, `run_all_test_samples.py` for the test set.

UW-hands (`python/tensorflow/uw/`):
- I didn't write restore and evaluate code for this. I made the model evaluate on the full validation set every ~30 steps during training, and applied stopping criteria to training. Then I looked in Tensorboard to get the results. Follow the Handcam examples if you want to restore this for some reason.

### Interpret the results of evaluating the model
Handcam:
- After running the validation script mentioned above, run `python/scratch/handcam_results.py` or `python/scratch/handcam_results_test_set.py`. It will print the results to the screen and make some confusion matrix files.

UW-hands (`python/tensorflow/uw/`):
- Again, nothing to interpret.

### Plot a Handcam sample
- Use `python/scratch/create_sample_plots.py`. The end of the file has the code for making the video. Feel free to clean this up, I left this is a development/experiment file so you can see the whole process for how I figured out how to combine the video and plots.

## "What about ... ?"

### The new Astra SDK from Orbbec
You can try to use this, I've added it to the `android` section and I'm able to compile it as part of the Android app, but I couldn't find any built-in methods for recording the frames, accessing frame timestamps, or any working example of how to use the SDK (currently you have to decompile the bytecode and read the class files to figure out what to do). Maybe it will be easier than OpenNI in the future?

