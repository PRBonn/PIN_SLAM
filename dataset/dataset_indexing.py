#!/usr/bin/env python3
# @file      dataset_indexing.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import os

from utils.config import Config

def set_dataset_path(config: Config, dataset_name: str = '', seq: str = ''):
    
    config.name = config.name + '_' + dataset_name + '_' + seq.replace("/", "")
    
    if config.use_kiss_dataloader:
        config.data_loader_name = dataset_name
        config.data_loader_seq = seq
        print('KISS-ICP data loaders used')
        from kiss_icp.datasets import available_dataloaders 
        print('Available dataloaders:', available_dataloaders())

    else:
        if dataset_name == "kitti":
            base_path = config.pc_path.rsplit('/', 3)[0]
            config.pc_path = os.path.join(base_path, 'sequences', seq, "velodyne")  # input point cloud folder
            pose_file_name = seq + '.txt'
            config.pose_path = os.path.join(base_path, 'poses', pose_file_name)   # input pose file
            config.calib_path = os.path.join(base_path, 'sequences', seq, "calib.txt")  # input calib file (to sensor frame)
            config.label_path = os.path.join(base_path, 'sequences', seq, "labels") # input point-wise label path, for semantic mapping (optional)
            config.kitti_correction_on = True
            config.correction_deg = 0.195
        elif dataset_name == "mulran":
            config.name = config.name + "_mulran_" + seq
            base_path = config.pc_path.rsplit('/', 2)[0]
            config.pc_path = os.path.join(base_path, seq, "Ouster")  # input point cloud folder
            config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        elif dataset_name == "kitti_carla":
            config.name = config.name + "_kitti_carla_" + seq
            base_path = config.pc_path.rsplit('/', 3)[0]
            config.pc_path = os.path.join(base_path, seq, "generated", "frames")  # input point cloud folder
            config.pose_path = os.path.join(base_path, seq, "generated", "poses.txt")   # input pose file
            config.calib_path = os.path.join(base_path, seq, "generated", "calib.txt")  # input calib file (to sensor frame)
        elif dataset_name == "ncd":
            config.name = config.name + "_ncd_" + seq
            base_path = config.pc_path.rsplit('/', 2)[0]
            config.pc_path = os.path.join(base_path, seq, "bin")  # input point cloud folder
            config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
            config.calib_path = os.path.join(base_path, seq, "calib.txt")  # input calib file (to sensor frame)
        elif dataset_name == "ncd128":
            config.name = config.name + "_ncd128_" + seq
            base_path = config.pc_path.rsplit('/', 2)[0]
            config.pc_path = os.path.join(base_path, seq, "ply")  # input point cloud folder
            config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        elif dataset_name == "ipbcar":
            config.name = config.name + "_ipbcar_" + seq
            base_path = config.pc_path.rsplit('/', 2)[0]
            config.pc_path = os.path.join(base_path, seq, "ouster")  # input point cloud folder
            config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
            config.calib_path = os.path.join(base_path, seq, "calib.txt")  # input calib file (to sensor frame)
        elif dataset_name == "hilti":
            config.name = config.name + "_hilti_" + seq
            base_path = config.pc_path.rsplit('/', 2)[0]
            config.pc_path = os.path.join(base_path, seq, "ply")  # input point cloud folder
            # config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        elif dataset_name == "m2dgr":
            config.name = config.name + "_m2dgr_" + seq
            base_path = config.pc_path.rsplit('/', 2)[0]
            config.pc_path = os.path.join(base_path, seq, "points")  # input point cloud folder
            config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        elif dataset_name == "replica":
            config.name = config.name + "_replica_" + seq
            base_path = config.pc_path.rsplit('/', 2)[0]
            config.pc_path = os.path.join(base_path, seq, "rgbd_down_ply")  # input point cloud folder
            #config.pc_path = os.path.join(base_path, seq, "rgbd_ply")  # input point cloud folder
            config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file     
