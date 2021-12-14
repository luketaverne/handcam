#!/bin/bash
# split 1
#split_num=1
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
## split 2
#split_num=2
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
## split 3
#split_num=3
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
## split 4
#split_num=4
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
## split 5
#split_num=5
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 6
split_num=6
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
# split 7
#split_num=7
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
## split 8
#split_num=8
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
## split 9
#split_num=9
#cp -R /home/luke/ait-server-03/datasets/handcam/single_frames/tfrecords/split${split_num} /home/luke/datasets/handcam/single_frames/tfrecords/
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality depth
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgb
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type single_frames --input_modality rgbd
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode frozen_train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality depth --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/depth/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgb --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgb/train/*/*/ | head -n 1` --batch_size 4
#python tensorflow/handcam/train_handcam_any.py --validation_split_num $split_num --model_type sequence --mode train --input_modality rgbd --resnet_weights_dir `ls -td -- /tmp/luke/handcam/split${split_num}/single_frames_resnet-18/rgbd/train/*/*/ | head -n 1` --batch_size 4
