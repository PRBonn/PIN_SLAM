#!/bin/bash

root_path=./data/TUM_rgbd

sequence_name=rgbd_dataset_freiburg2_xyz

base_path=${root_path}/${sequence_name}

command="python3 ./dataset/converter/tum_to_pin_format.py
        --input_root ${base_path}
        --output_root ${base_path}
        --intrinsic_file ${base_path}/cam_params.json
        --vis_on True
        --down_sample False"

echo "Convert TUM RGBD dataset to our format"
eval $command
echo "Done."