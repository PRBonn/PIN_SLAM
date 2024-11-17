#!/bin/bash

xhost local:root && docker run -it --rm -e SDL_VIDEODRIVER=x11 -e DISPLAY=$DISPLAY --env='DISPLAY' --ipc host --privileged --network host -p 8080:8081  --gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw  \
-v /your_data_storage_directory_here:/storage/  \
pinslam:localbuild xfce4-terminal --title=PIN-SLAM
