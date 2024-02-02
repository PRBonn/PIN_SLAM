#!/usr/bin/env python3
# @file      replica_to_pin_format.py
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

sys.path.append('./') # the script is launched in the root folder, required for import visualizer
from utils.visualizer import MapVisualizer

def replica_to_pin_format(args):

    if args.vis_on:
        vis = MapVisualizer()

    color_paths = sorted(glob.glob(f'{args.input_root}/results/frame*.jpg'))
    depth_paths = sorted(glob.glob(f'{args.input_root}/results/depth*.png'))
    poses = load_poses(os.path.join(args.input_root, 'traj.txt'), len(color_paths))

    # get pose
    pose_kitti_format_path = os.path.join(args.output_root, "poses.txt")

    write_poses_kitti_format(poses, pose_kitti_format_path)

    if args.down_sample:
        ply_path = os.path.join(args.output_root, "rgbd_down_ply")
    else:
        ply_path = os.path.join(args.output_root, "rgbd_ply")
    
    os.makedirs(ply_path, exist_ok=True)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    with open(args.intrinsic_file, 'r') as infile: # load intrinsic json file
        cam = json.load(infile)["camera"]
        intrinsic.set_intrinsics(height=cam["h"],
                                 width=cam["w"],
                                 fx=cam["fx"],
                                 fy=cam["fy"],
                                 cx=cam["cx"],
                                 cy=cam["cy"])

    depth_scale = cam["scale"]
    extrinsic = np.eye(4)

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

        im_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsic)

        if args.down_sample:
            im_pcd = im_pcd.random_down_sample(sampling_ratio=0.1) # TODO change here

        o3d.io.write_point_cloud(cur_path, im_pcd)      

        if args.vis_on:
            im_pcd = im_pcd.transform(poses[frame_count])
            if not args.down_sample:
                im_pcd = im_pcd.random_down_sample(sampling_ratio=0.1)
            map_pcd += im_pcd
            map_pcd = map_pcd.voxel_down_sample(0.03)
            vis.update_pointcloud(map_pcd)

        frame_count+=1
    
    print("The Replica RGBD dataset in our format has been saved at %s", args.output_root)

    if args.vis_on:
        vis.stop()

def load_poses(path, frame_num):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(frame_num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        # c2w[:3, 1] *= -1
        # c2w[:3, 2] *= -1
        # c2w[:3, 3] *= self.sc_factor
        poses.append(c2w)

    return poses

def write_poses_kitti_format(poses_mat, posefile):
    poses_vec = []
    for pose_mat in poses_mat:
        pose_vec= pose_mat.flatten()[0:12]
        poses_vec.append(pose_vec)
    np.savetxt(posefile, poses_vec, delimiter=' ')


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
    
    replica_to_pin_format(args)