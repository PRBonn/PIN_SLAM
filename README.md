<p align="center">

  <h1 align="center">📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency</h1>

  <p align="center">
    <a href="https://github.com/PRBonn/PIN_SLAM#run-pin-slam"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/PIN_SLAM#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://arxiv.org/pdf/2401.09101v1.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/PRBonn/PIN_SLAM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>
  
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/"><strong>Yue Pan</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/xingguang-zhong/"><strong>Xingguang Zhong</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/louis-wiesmann/"><strong>Louis Wiesmann</strong></a>
    .
    <a href=""><strong>Thorbjörn Posewsky</strong></a>
    .
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong></a>
  <h3 align="center"><a href="https://arxiv.org/pdf/2401.09101v1.pdf">Preprint</a> | Video</a></h3>
  <div align="center"></div>
</p>

**TL;DR: PIN-SLAM is a full-fledged implicit neural LiDAR SLAM system including odometry, loop closure detection, and globally consistent mapping**


![pin_slam_teaser](https://github.com/PRBonn/PIN_SLAM/assets/34207278/b5ab4c89-cdbe-464e-afbe-eb432b42fccc)

*Globally consistent point-based implicit neural (PIN) map built with PIN-SLAM in Bonn. The high-fidelity mesh can be reconstructed from the neural point map.*

----

![pin_slam_loop_compare](https://github.com/PRBonn/PIN_SLAM/assets/34207278/7dadd438-5a46-451a-9add-c9c08dcae277)

*Comparison of (a) the inconsistent mesh with duplicated structures reconstructed by PIN LiDAR odometry, and (b) the globally consistent mesh reconstructed by PIN-SLAM.*


----


| Globally Consistent Mapping | Various Scenarios | RGB-D SLAM Extension |
| :-: | :-: | :-: |
| <video src='https://github.com/PRBonn/PIN_SLAM/assets/34207278/b157f24c-0220-4ac4-8cf3-2247aeedfc2e'> | <video src='https://github.com/PRBonn/PIN_SLAM/assets/34207278/0906f7cd-aebe-4fb7-9ad4-514d089329bd'> | <video src='https://github.com/PRBonn/PIN_SLAM/assets/34207278/4519f4a8-3f62-42a1-897e-d9feb66bfcd0'> |


<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run-pin-slam">How to run PIN-SLAM</a>
    </li>
    <li>
      <a href="#visualizer-instructions">Visualizer instructions</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
    <li>
      <a href="#related-projects">Related projects</a>
    </li>
  </ol>
</details>


## Abstract

<details>
  <summary>[Details (click to expand)]</summary>
Accurate and robust localization and mapping are
essential components for most autonomous robots. In this paper,
we propose a SLAM system for building globally consistent maps,
called PIN-SLAM, that is based on an elastic and compact
point-based implicit neural map representation. Taking range
measurements as input, our approach alternates between incremental learning of the local implicit signed distance field
and the pose estimation given the current local map using a
correspondence-free, point-to-implicit model registration. Our
implicit map is based on sparse optimizable neural points,
which are inherently elastic and deformable with the global pose
adjustment when closing a loop. Loops are also detected using the
neural point features. Extensive experiments validate that PIN-SLAM is robust to various environments and versatile to different
range sensors such as LiDAR and RGB-D cameras. PIN-SLAM
achieves pose estimation accuracy better or on par with the state-of-the-art LiDAR odometry or SLAM systems and outperforms
the recent neural implicit SLAM approaches while maintaining
a more consistent, and highly compact implicit map that can be
reconstructed as accurate and complete meshes. Finally, thanks to
the voxel hashing for efficient neural points indexing and the fast
implicit map-based registration without closest point association,
PIN-SLAM can run at the sensor frame rate on a moderate GPU.
</details>



## Installation

### Platform requirement
* Ubuntu OS (tested on 20.04)

* With GPU (recommended) or CPU only (run much slower)

* GPU memory requirement (> 6 GB recommended)

* Windows/MacOS with CPU-only mode

### 1. Set up conda environment

```
conda create --name pin python=3.8
conda activate pin
```

### 2. Install the key requirement PyTorch

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

The commands depend on your CUDA version. You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).

### 3. Install other dependency

```
pip3 install open3d==0.17 scikit-image gtsam wandb tqdm rich roma natsort pyquaternion pypose evo laspy rospkg 
```

Note that `rospkg` is optional. You can install it if you would like to use PIN-SLAM with ROS.


## Run PIN-SLAM

### Clone the repository

```
git clone git@github.com:PRBonn/PIN_SLAM.git
cd PIN_SLAM
```

### Sanity test

For a sanity test, do the following to download an example part (first 100 frames) of the KITTI dataset (seq 00):

```
sh ./scripts/download_kitti_example.sh
```

And then run:

```
python3 pin_slam.py ./config/lidar_slam/run_demo.yaml
```

<details>
  <summary>[Details (click to expand)]</summary>
  
Use `run_demo_no_vis.yaml` if you are running on a server without an X service. 
Use `run_demo_sem.yaml` if you want to conduct metric-semantic SLAM using semantic segmentation labels.

You can visualize the SLAM process in PIN-SLAM visualizer and check the results in the `./experiments` folder.
</details>


### Run on your datasets

For an arbitrary data sequence, you can run:
```
python3 pin_slam.py path_to_your_config_file.yaml
```

<details>
  <summary>[Details (click to expand)]</summary>

Generally speaking, you only need to edit in the config file the 
`pc_path`, which is the path to the folder containing the point cloud (`.bin`, `.ply`, `.pcd` or `.las` format) for each frame. 
For ROS bag, you can use `./scripts/rosbag2ply.py` to extract the point cloud in `.ply` format.

For pose estimation evaluation, you may also provide the path `pose_path` to the reference pose file and optionally the path `calib_path` to the extrinsic calibration file. Note the pose file and calibration file should be in the [KITTI odometry data format](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

The SLAM results and logs will be output in the `output_root` folder specified in the config file. 

You may check [here](https://github.com/PRBonn/PIN_SLAM/blob/main/eval/README.md) for the results that can be obtained with this repository on a couple of popular datasets. 

The training logs can be monitored via [Weights & Bias](wandb.ai) online if you turn on the `wandb_vis_on` option in the config file. If it's your first time using Weights & Bias, you will be requested to register and log in to your wandb account.

</details>

### ROS 1 Support

If you are not using PIN-SLAM as a part of a ROS package, you can avoid the catkin stuff and simply run:

```
python3 pin_slam_ros.py [path_to_your_config_file] [point_cloud_topic_name] [(optional)point_timestamp_field_name]
```

<details>
  <summary>[Details (click to expand)]</summary>

For example:

```
python3 pin_slam_ros.py ./config/lidar_slam/run_ros_general.yaml /os_cloud_node/points time
```

After playing the ROS bag or launching the sensor you can then visualize the results in Rviz by:

```
rviz -d ./config/pin_slam_ros.rviz 
```

You may use the ROS service `save_results` and `save_mesh` to save the results and mesh in the `output_root` folder.

The process will stop and the results and logs will be saved in the `output_root` folder if no new messages are received for more than 30 seconds.

If you are running without a powerful GPU, PIN-SLAM may not run at the sensor frame rate. You need to play the rosbag with a lower rate to run PIN-SLAM properly.

You can also put `pin_slam_ros.py` into a ROS package for `rosrun` or `roslaunch`.

</details>


### Inspect the results after SLAM

After the SLAM process, you can reconstruct mesh from the PIN map within an arbitrary bounding box with an arbitrary resolution by running:

```
python3 vis_pin_map.py [path/to/your/result/folder] [marching_cubes_resolution_m] [(cropped)_map_file.ply] [output_mesh_file.ply] [mesh_min_nn]
```

<details>
  <summary>[Details (click to expand)]</summary>

The bounding box of `(cropped)_map_file.ply` will be used for the bounding box for mesh reconstruction. `mesh_min_nn` controls the trade-off between completeness and accuracy. The smaller number (for example `6`) will lead to a more complete mesh with more guessed artifacts. The larger number (for example `15`) will lead to a less complete but more accurate mesh.

For example, for the case of the sanity test, run:

```
python3 vis_pin_map.py ./experiments/sanity_test_* 0.2 neural_points.ply mesh_20cm.ply 8
```
</details>


## Visualizer Instructions

We provide a PIN-SLAM visualizer based on [lidar-visualizer](https://github.com/PRBonn/lidar-visualizer) to monitor the SLAM process.

The keyboard callbacks are listed below.

<details>
  <summary>[Details (click to expand)]</summary>

| Button |                                          Function                                          |
|:------:|:------------------------------------------------------------------------------------------:|
|  Space |                                        pause/resume                                        |
| ESC/Q  |                           exit                                                             |
|   G    |                     switch between the global/local map visualization                      |
|   E    |                     switch between the ego/map viewpoint                                   |
|   F    |                     toggle on/off the current point cloud  visualization                   |
|   M    |                         toggle on/off the mesh visualization                               |
|   A    |                 toggle on/off the current frame axis & sensor model visualization          |
|   P    |                 toggle on/off the neural points map visualization                          |
|   D    |               toggle on/off the training data pool visualization                           |
|   I    |               toggle on/off the SDF horizontal slice visualization                         |
|   T    |              toggle on/off PIN SLAM trajectory visualization                               |
|   Y    |              toggle on/off the ground truth trajectory visualization                       |
|   U    |              toggle on/off PIN odometry trajectory visualization                           |
|   R    |                           re-center the view point                                         |
|   Z    |              3D screenshot, save the currently visualized entities in the log folder       |
|   B    |                  toggle on/off back face rendering                                         |
|   W    |                  toggle on/off mesh wireframe                                              |
| Ctrl+9 |                                Set mesh color as normal direction                          |
|   5    |   switch between point cloud for mapping and for registration (with point-wise weight)     |
|   7    |                                      switch between black and white background             |
|   /    |   switch among different neural point color mode, 0: geometric feature, 1: color feature, 2: timestamp, 3: stability, 4: random             |
|  <     |  decrease mesh nearest neighbor threshold (more complete and more artifacts)               |
|  >     |  increase mesh nearest neighbor threshold (less complete but more accurate)                |
|  \[/\] |  decrease/increase mesh marching cubes voxel size                                          |
|  ↑/↓   |  move up/down the horizontal SDF slice                                                     |
|  +/-   |                  increase/decrease point size                                              |

</details>

## Contact
If you have any questions, please contact:

- Yue Pan {[yue.pan@igg.uni-bonn.de]()}


## Related Projects

[SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping): Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations

[LocNDF](https://github.com/PRBonn/LocNDF): Neural Distance Field Mapping for Robot Localization

[KISS-ICP](https://github.com/PRBonn/kiss-icp): A LiDAR odometry pipeline that just works