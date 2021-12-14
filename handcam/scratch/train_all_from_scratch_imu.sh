#!/bin/bash
#split0
split_num=0
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 1
split_num=1
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 2
split_num=2
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 3
split_num=3
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 4
split_num=4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 5
split_num=5
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 6
split_num=6
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 7
split_num=7
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 8
split_num=8
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 9
split_num=9
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --with_naive_IMU=True --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /media/luke/hdd-3tb/models/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4

