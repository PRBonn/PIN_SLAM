#!/usr/bin/env python3
# @file      tum_to_pin_format.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import open3d as o3d
from tqdm import tqdm
import argparse
import os
import numpy as np
import json
import sys

sys.path.append('./') # the script is launched in the root folder, required for import visualizer
from utils.visualizer import MapVisualizer

def tum_to_pin_format(args):

    if args.vis_on:
        vis = MapVisualizer()

    # get pose
    pose_kitti_format_path = os.path.join(args.output_root, "poses.txt")
    
    color_paths, depth_paths, poses = loadtum(args.input_root, switch_axis=args.switch_axis)

    write_poses_kitti_format(poses, pose_kitti_format_path)

    if args.down_sample:
        ply_path = os.path.join(args.output_root, "rgbd_down_ply")
    else:
        ply_path = os.path.join(args.output_root, "rgbd_ply")
    
    os.makedirs(ply_path, exist_ok=True)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    if args.intrinsic_file is None: # default one
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        # fx = 525, fy = 525, cx = 319.5, cy = 239.5   as the default value
    else:           
        with open(args.intrinsic_file, 'r') as infile: # load intrinsic json file
            cam = json.load(infile)["camera"]
            intrinsic.set_intrinsics(height=cam["h"],
                                     width=cam["w"],
                                     fx=cam["fx"],
                                     fy=cam["fy"],
                                     cx=cam["cx"],
                                     cy=cam["cy"])

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
        im_rgbd = o3d.geometry.RGBDImage.create_from_tum_format(im_color, im_depth, convert_rgb_to_intensity=False)

        im_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsic)

        if args.down_sample:
            im_pcd = im_pcd.random_down_sample(sampling_ratio=0.2) # TODO change here

        o3d.io.write_point_cloud(cur_path, im_pcd)      

        if args.vis_on:
            im_pcd = im_pcd.transform(poses[frame_count])
            # down_pcd = down_pcd.random_down_sample(sampling_ratio=0.1)
            map_pcd += im_pcd
            map_pcd = map_pcd.voxel_down_sample(0.02)
            vis.update_pointcloud(map_pcd)

        frame_count+=1
    
    print("The TUM RGBD dataset in our format has been saved at %s", args.output_root)

    if args.vis_on:
        vis.stop()

def loadtum(datapath, frame_rate=-1, switch_axis=False):
    """ read video data in tum-rgbd format """
    if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
        pose_list = os.path.join(datapath, 'groundtruth.txt')
    elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
        pose_list = os.path.join(datapath, 'pose.txt')

    image_list = os.path.join(datapath, 'rgb.txt')
    depth_list = os.path.join(datapath, 'depth.txt')

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:, 1:].astype(np.float64)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_depth = depth_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)
    associations = associate_frames(
        tstamp_image, tstamp_depth, tstamp_pose)

    indicies = [0]
    for i in range(1, len(associations)):
        t0 = tstamp_image[associations[indicies[-1]][0]]
        t1 = tstamp_image[associations[i][0]]
        if t1 - t0 > 1.0 / frame_rate:
            indicies += [i]

    images, poses, depths = [], [], []
    for ix in indicies:
        (i, j, k) = associations[ix]
        images += [os.path.join(datapath, image_data[i, 1])]
        depths += [os.path.join(datapath, depth_data[j, 1])]
        c2w = pose_matrix_from_quaternion(pose_vecs[k])
        # if switch_axis:
        #     c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
        poses += [c2w]

    return images, depths, poses # as file path and pose


def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    from scipy.spatial.transform import Rotation
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix() # rotation
    pose[:3, 3] = pvec[:3] # translation
    return pose
    
def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j = np.argmin(np.abs(tstamp_depth - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))

            if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                    (np.abs(tstamp_pose[k] - t) < max_dt):
                associations.append((i, j, k))

    return associations
    
def parse_list(filepath, skiprows=0):
    """ read list data """
    data = np.loadtxt(filepath, delimiter=' ',
                        dtype=np.unicode_, skiprows=skiprows)
    return data

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
    parser.add_argument('--crop_edge', type=int, default=0.0, help="edge pixels for cropping")
    parser.add_argument('--vis_on', type=str2bool, nargs='?', default=False)
    parser.add_argument('--down_sample', type=str2bool, nargs='?', default=False)
    parser.add_argument('--switch_axis', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    
    tum_to_pin_format(args)