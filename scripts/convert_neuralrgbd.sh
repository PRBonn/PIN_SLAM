#!/bin/bash

root_path=./data/neural_rgbd_data

sequence_name=icl_living_room
#sequence_name=breakfast_room
#sequence_name=staircase
#sequence_name=green_room
#sequence_name=complete_kitchen
#sequence_name=whiteroom
#sequence_name=grey_white_room
#sequence_name=morning_apartment
#sequence_name=kitchen
#sequence_name=thin_geometry

base_path=${root_path}/${sequence_name}

command="python3 ./dataset/converter/neuralrgbd_to_pin_format.py
        --input_root ${base_path}
        --output_root ${base_path}
        --intrinsic_file ${base_path}/focal.txt
        --vis_on False
        --down_sample False"

echo "Convert Neural RGBD dataset to our format"
eval $command
echo "Done."