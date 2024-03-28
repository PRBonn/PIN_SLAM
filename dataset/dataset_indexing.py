#!/usr/bin/env python3
# @file      dataset_indexing.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import os

from utils.config import Config

def set_dataset_path(config: Config, dataset_name: str, seq: str):
    if dataset_name == "kitti":
        config.name = "kitti_" + seq
        base_path = config.pc_path.rsplit('/', 3)[0]
        config.pc_path = os.path.join(base_path, 'sequences', seq, "velodyne")  # input point cloud folder
        pose_file_name = seq + '.txt'
        config.pose_path = os.path.join(base_path, 'poses', pose_file_name)   # input pose file
        config.calib_path = os.path.join(base_path, 'sequences', seq, "calib.txt")  # input calib file (to sensor frame)
        config.label_path = os.path.join(base_path, 'sequences', seq, "labels") # input point-wise label path, for semantic mapping (optional)
    elif dataset_name == "mulran":
        config.name = "mulran_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "Ouster")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        config.calib_path = os.path.join(base_path, seq, "calib.txt")  # input calib file (to sensor frame)
    elif dataset_name == "kitti_carla":
        config.name = "kitti_carla_" + seq
        base_path = config.pc_path.rsplit('/', 3)[0]
        config.pc_path = os.path.join(base_path, seq, "generated", "frames")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "generated", "poses.txt")   # input pose file
        config.calib_path = os.path.join(base_path, seq, "generated", "calib.txt")  # input calib file (to sensor frame)
    elif dataset_name == "ncd":
        config.name = "ncd_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "bin")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        config.calib_path = os.path.join(base_path, seq, "calib.txt")  # input calib file (to sensor frame)
    elif dataset_name == "ncd128":
        config.name = "ncd128_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "ply")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
    elif dataset_name == "ipbcar":
        config.name = "ipbcar_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "ouster")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        config.calib_path = os.path.join(base_path, seq, "calib.txt")  # input calib file (to sensor frame)
    elif dataset_name == "ntu":
        config.name = "ntu_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "pointcloud_l1")  # input point cloud folder
    elif dataset_name == "hilti":
        config.name = "hilti_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "pointcloud")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
    elif dataset_name == "eth_dynamic":
        config.name = "eth_dynamic_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "ply")  # input point cloud folder
    elif dataset_name == "m2dgr":
        config.name = "m2dgr_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "points")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
    elif dataset_name == "replica":
        config.name = "replica_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "rgbd_down_ply")  # input point cloud folder
        #config.pc_path = os.path.join(base_path, seq, "rgbd_ply")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file     
    elif dataset_name == "neuralrgbd":
        config.name = "neuralrgbd_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "rgbd_ply")  # input point cloud folder
        # config.pc_path = os.path.join(base_path, seq, "rgbd_gt_ply")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses_pin.txt")   # input pose file     
    elif dataset_name == "tum":
        config.name = "tum_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "rgbd_ply")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file    
    elif dataset_name == "scannet":
        config.name = "scannet_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "rgbd_ply")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file    
