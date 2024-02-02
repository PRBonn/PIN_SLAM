#!/usr/bin/env python3
# @file      neuralrgbd_to_pin_format.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import open3d as o3d
from tqdm import tqdm
import argparse
import os
import numpy as np
import json
import sys
import glob
import re

sys.path.append('./') # the script is launched in the root folder, required for import visualizer
from utils.visualizer import MapVisualizer

# icl living room data is a bit different
def neuralrgbd_to_pin_format(args):

    if args.vis_on:
        vis = MapVisualizer()

    color_paths = [os.path.join(args.input_root, 'images', f) for f in sorted(os.listdir(os.path.join(args.input_root, 'images')), key=alphanum_key) if f.endswith('png')]
    # depth_paths = [os.path.join(args.input_root, 'depth_filtered', f) for f in sorted(os.listdir(os.path.join(args.input_root, 'depth_filtered')), key=alphanum_key) if f.endswith('png')]
    depth_paths = [os.path.join(args.input_root, 'depth', f) for f in sorted(os.listdir(os.path.join(args.input_root, 'depth')), key=alphanum_key) if f.endswith('png')]
    
    # get pose
    all_gt_poses, valid_gt_poses = load_poses(os.path.join(args.input_root, 'poses.txt'))

    pose_kitti_format_path = os.path.join(args.output_root, "poses_pin.txt")

    write_poses_kitti_format(all_gt_poses, pose_kitti_format_path)

    if args.down_sample:
        ply_path = os.path.join(args.output_root, "rgbd_down_ply")
    else:
        # ply_path = os.path.join(args.output_root, "rgbd_ply")
        ply_path = os.path.join(args.output_root, "rgbd_gt_ply")
    
    os.makedirs(ply_path, exist_ok=True)

    H = 480
    W = 640

    intrinsic = o3d.camera.PinholeCameraIntrinsic()

    focal = load_focal_length(args.intrinsic_file)
    print("Focal length:", focal)

    intrinsic.set_intrinsics(height=H,
                            width=W,
                            fx=focal,
                            fy=focal,
                            cx=(W-1.)/2.,
                            cy=(H-1.)/2.)

    depth_scale = 1000.
    # use this extrinsic matrix to rotate the image since frames captured with RealSense camera are upside down # really?
    extrinsic = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # extrinsic = np.eye(4)
    
    print("Camera Intrinsics:")
    print(intrinsic.intrinsic_matrix)                         

    map_pcd = o3d.geometry.PointCloud()

    frame_count = 0
    for color_path, depth_path in tqdm(zip(color_paths, depth_paths)):
        
        frame_id_str = f'{frame_count:06d}'
        cur_filename = frame_id_str+".ply"
        cur_path = os.path.join(ply_path, cur_filename)

        print(frame_id_str)
        print(color_path)
        print(depth_path)
        
        im_color = o3d.io.read_image(color_path)
        im_depth = o3d.io.read_image(depth_path) 
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(im_color, im_depth, depth_scale=depth_scale, 
                                                                     depth_trunc=args.max_depth_m, convert_rgb_to_intensity=False)

        im_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsic, extrinsic)

        if args.down_sample:
            im_pcd = im_pcd.random_down_sample(sampling_ratio=0.2) # TODO change here

        o3d.io.write_point_cloud(cur_path, im_pcd)      

        if args.vis_on:
            im_pcd = im_pcd.transform(all_gt_poses[frame_count])
            map_pcd += im_pcd
            map_pcd = map_pcd.voxel_down_sample(0.05)
            vis.update_pointcloud(map_pcd)

        frame_count+=1
    
    print("The Neural RGBD dataset in our format has been saved at %s", args.output_root)

    if args.vis_on:
        vis.stop()

def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())

def load_poses(path):
    file = open(path, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if 'nan' in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = np.array([[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]])
            poses.append(pose_floats)

    return poses, valid

def write_poses_kitti_format(poses_mat, posefile):
    poses_vec = []
    for pose_mat in poses_mat:
        pose_vec= pose_mat.flatten()[0:12]
        poses_vec.append(pose_vec)
    np.savetxt(posefile, poses_vec, delimiter=' ')

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', help="base path for the input data")
    parser.add_argument('--output_root', help="path for outputing the kitti format data")
    parser.add_argument('--intrinsic_file', help="path to the camera intrinsic json file", default=None)
    parser.add_argument('--max_depth_m', type=float, default=10.0, help="maximum depth to be used")
    parser.add_argument('--vis_on', type=str2bool, nargs='?', default=False)
    parser.add_argument('--down_sample', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    
    neuralrgbd_to_pin_format(args)