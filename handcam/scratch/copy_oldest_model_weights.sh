#!/usr/bin/env bash
new_path=/media/luke/hdd-3tb/models/handcam-paper/
old_path=/media/luke/hdd-3tb/models/handcam/

model_paths=`ls -td -- /media/luke/hdd-3tb/models/handcam/split*/*/*/*/*/*/`
# is array of: /media/luke/hdd-3tb/models/handcam/split3/single_frames_resnet-18/depth/train/2018-08-22/11:00/

#echo $model_paths
for i in $model_paths
do
    root_out_new="$new_path${i:35}" # split0/single_frames_resnet-18/depth/train/2018-08-20/18:22/
    # need to copy summary/*, FLAGS.pckl, model-###*, *.pckl
    model_ckpt_data=`ls -td -- ${i}model-*.meta | head -n 1 | grep -Eo '[0-9]+.meta$'`
#    model_ckpt_data=`exec ls ${i}model-*.meta | sed 's/\([0-9]\+\).*/\1/g' | sort -n | head -1`
    ckpt_number="${model_ckpt_data::-5}" # just the number left

    mkdir -p ${root_out_new}summary/
#    echo $i
    cp -R ${i}summary/* ${root_out_new}summary/
    cp ${i}*.pckl ${root_out_new}
    cp ${i}checkpoint ${root_out_new}
    cp ${i}model-${ckpt_number}* ${root_out_new}

done