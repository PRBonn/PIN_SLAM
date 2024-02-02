#!/bin/bash

root_path=./data/Replica

sequence_name=room0

base_path=${root_path}/${sequence_name}

command="python3 ./dataset/converter/replica_to_pin_format.py
        --input_root ${base_path}
        --output_root ${base_path}
        --intrinsic_file ${root_path}/cam_params.json
        --vis_on False
        --down_sample True"

echo "Convert Replica RGBD dataset to our format"
eval $command
echo "Done."