#!/usr/bin/env python3
# @file      slam_dataset.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import os
import sys
import numpy as np
from numpy.linalg import inv
import math
import torch
from torch.utils.data import Dataset
import contextlib
import open3d as o3d
from tqdm import tqdm
from rich import print
import csv
from typing import List
import matplotlib.cm as cm
import wandb

from utils.config import Config
from utils.tools import get_time, voxel_down_sample_torch, deskewing, transform_torch, plot_timing_detail, tranmat_close_to_identity
from utils.semantic_kitti_utils import *
from eval.eval_traj_utils import *

# TODO: write a new dataloader for RGB-D inputs, not always converting them to KITTI Lidar format

class SLAMDataset(Dataset):
    def __init__(self, config: Config) -> None:

        super().__init__()

        self.config = config
        self.silence = config.silence
        self.dtype = config.dtype
        self.device = config.device

        # point cloud files
        if config.pc_path != "":
            from natsort import natsorted 
            self.pc_filenames = natsorted(os.listdir(config.pc_path)) # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
            self.total_pc_count = len(self.pc_filenames)

        # pose related
        self.gt_pose_provided = True
        if config.pose_path == '':
            self.gt_pose_provided = False

        self.odom_poses = None
        if config.track_on:
            self.odom_poses = []
            
        self.pgo_poses = None
        if config.pgo_on:
            self.pgo_poses = []

        self.gt_poses = None
        if self.gt_pose_provided:
            self.gt_poses = []

        self.poses_w = None
        self.poses_w_closed = None

        self.travel_dist = []
        self.time_table = []

        self.poses_ref = [np.eye(4)] # only used when gt_pose_provided

        self.calib = {}
        self.calib['Tr'] = np.eye(4) # as default if calib file is not provided # as T_lidar<-camera
        
        if self.gt_pose_provided:
            if config.calib_path != '':
                self.calib = read_kitti_format_calib(config.calib_path)
            # TODO: this should be updated, select the pose with correct format, tum format may not endwith csv
            if config.pose_path.endswith('txt'):
                poses_uncalib = read_kitti_format_poses(config.pose_path)
                if config.closed_pose_path is not None and config.use_gt_loop:
                    poses_closed_uncalib = read_kitti_format_poses(config.closed_pose_path)
            elif config.pose_path.endswith('csv'):
                poses_uncalib = read_tum_format_poses_csv(config.pose_path)
            else: 
                sys.exit("Wrong pose file format. Please use either *.txt or *.csv")

            # apply calibration
            # actually from camera frame to LiDAR frame, lidar pose in world frame 
            self.poses_w = apply_kitti_format_calib(poses_uncalib, inv(self.calib['Tr'])) 
            if config.closed_pose_path is not None and config.use_gt_loop:
                self.poses_w_closed = apply_kitti_format_calib(poses_closed_uncalib, inv(self.calib['Tr'])) 

            # pose in the reference frame (might be the first frame used)
            self.poses_ref = self.poses_w  # initialize size
            if len(self.poses_w) != self.total_pc_count:
                sys.exit("Number of the pose and point cloud are not identical")
            if self.total_pc_count > 2000:
                config.local_map_context = True

            # get the pose in the reference frame
            begin_flag = False
            begin_pose_inv = np.eye(4)
            
            for frame_id in range(self.total_pc_count):
                if not begin_flag:  # the first frame used
                    begin_flag = True
                    if config.first_frame_ref: # use the first frame as the reference (identity)
                        begin_pose_inv = inv(self.poses_w[frame_id])  # T_rw      
                self.poses_ref[frame_id] = begin_pose_inv @ self.poses_w[frame_id]
        # or we directly use the world frame as reference
 
        self.processed_frame: int = 0
        self.shift_ts: float = 0.0
        self.lose_track: bool = False # the odometry lose track or not (for robustness)
        self.consecutive_lose_track_frame: int = 0
        self.color_available: bool = False
        self.intensity_available: bool = False
        self.color_scale: float = 255.
        self.last_pose_ref = np.eye(4)
        self.last_odom_tran = np.eye(4)
        self.cur_pose_ref = np.eye(4)
        # count the consecutive stop frame of the robot
        self.stop_count: int = 0
        self.stop_status = False

        if self.config.kitti_correction_on:
            self.last_odom_tran[0,3] = self.config.max_range*1e-2 # inital guess for booting on x aixs
            self.color_scale = 1.
        

        # current frame point cloud (for visualization)
        self.cur_frame_o3d = o3d.geometry.PointCloud()
        # current frame bounding box in the world coordinate system
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()
        # merged downsampled point cloud (for visualization)
        self.map_down_o3d = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()

        self.static_mask = None

        # current frame's data
        self.cur_point_cloud_torch = None
        self.cur_point_ts_torch = None
        self.cur_sem_labels_torch = None
        self.cur_sem_labels_full = None

        # source data for registration
        self.cur_source_points = None
        self.cur_source_normals = None
        self.cur_source_colors = None


    def read_frame_ros(self, msg, ts_field_name = "time", ts_col=3):

        # ros related
        import rospy
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs import point_cloud2

        # ts_col represents the column id for timestamp
        self.cur_pose_ref = np.eye(4)
        self.cur_pose_torch = torch.tensor(self.cur_pose_ref, device=self.device, dtype=self.dtype)

        pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z", ts_field_name), skip_nans=True)

        # convert the point cloud data to a numpy array
        data = np.array(list(pc_data))

        # print(data)

        # how to read the timestamp information
        if ts_col > data.shape[1]-1:
            point_ts = None
        else:
            point_ts = data[:, ts_col]

            if self.processed_frame == 0:
                self.shift_ts = point_ts[0]

            point_ts = point_ts - self.shift_ts

        # print(point_ts)
        
        point_cloud = data[:,:3]
        
        if point_ts is None:
            print("The point cloud message does not contain the time stamp field:", ts_field_name)

        self.cur_point_cloud_torch = torch.tensor(point_cloud, device=self.device, dtype=self.dtype)

        if self.config.deskew:
            self.get_point_ts(point_ts)
            

    def read_frame(self, frame_id):
        
        # load gt pose if available
        if self.gt_pose_provided:
            self.cur_pose_ref = self.poses_ref[frame_id]
            self.gt_poses.append(self.cur_pose_ref)
        else: # or initialize with identity
            self.cur_pose_ref = np.eye(4)
        self.cur_pose_torch = torch.tensor(self.cur_pose_ref, device=self.device, dtype=self.dtype)

        point_ts = None

        # load point cloud (support *pcd, *ply and kitti *bin format)
        frame_filename = os.path.join(self.config.pc_path, self.pc_filenames[frame_id])
        if not self.silence:
            print(frame_filename)
        if not self.config.semantic_on: 
            point_cloud, point_ts = read_point_cloud(frame_filename, self.config.color_channel) #  [N, 3], [N, 4] or [N, 6], may contain color or intensity 
            if self.config.color_channel > 0:
                point_cloud[:,-self.config.color_channel:]/=self.color_scale
            self.cur_sem_labels_torch = None
        else:
            label_filename = os.path.join(self.config.label_path, self.pc_filenames[frame_id].replace('bin','label'))
            point_cloud, sem_labels, sem_labels_reduced = read_semantic_point_label(frame_filename, label_filename) # [N, 4] , [N], [N]
            self.cur_sem_labels_torch = torch.tensor(sem_labels_reduced, device=self.device, dtype=torch.long) # reduced labels (20 classes)
            self.cur_sem_labels_full = torch.tensor(sem_labels, device=self.device, dtype=torch.long) # full labels (>20 classes)
        
        self.cur_point_cloud_torch = torch.tensor(point_cloud, device=self.device, dtype=self.dtype)

        if self.config.deskew:
            self.get_point_ts(point_ts)
        
        # print(self.cur_point_ts_torch)
    
    # point-wise timestamp is now only used for motion undistortion (deskewing)
    def get_point_ts(self, point_ts = None):
        if self.config.deskew:
            if point_ts is not None and self.config.valid_ts_in_points: 
                if not self.silence:
                    print('Pointwise timestamp available')
                self.cur_point_ts_torch = torch.tensor(point_ts, device=self.device, dtype=self.dtype)
            else:
                if self.cur_point_cloud_torch.shape[0] == 64 * 1024:  # for Ouster 64-beam LiDAR
                    if not self.silence:
                        print("Ouster-64 point cloud deskewed")
                    self.cur_point_ts_torch = (torch.floor(torch.arange(64 * 1024) / 64) / 1024).reshape(-1, 1).to(self.cur_point_cloud_torch)
                elif self.cur_point_cloud_torch.shape[0] == 128 * 1024:  # for Ouster 128-beam LiDAR
                    if not self.silence:
                        print("Ouster-128 point cloud deskewed")
                    self.cur_point_ts_torch = (torch.floor(torch.arange(128 * 1024) / 128) / 1024).reshape(-1, 1).to(self.cur_point_cloud_torch)
                else:
                    yaw = -torch.atan2(self.cur_point_cloud_torch[:,1], self.cur_point_cloud_torch[:,0])  # y, x -> rad (clockwise)
                    if self.config.lidar_type_guess == "velodyne":
                        # for velodyne LiDAR (from -x axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 1.0) # [0,1]
                        if not self.silence:
                            print("Velodyne point cloud deskewed")
                    else:
                        # for Hesai LiDAR (from +y axis, clockwise)
                        self.cur_point_ts_torch = 0.5 * (yaw / math.pi + 0.5) # [-0.25,0.75]
                        self.cur_point_ts_torch[self.cur_point_ts_torch < 0] += 1.0 # [0,1]
                        if not self.silence:
                            print("HESAI point cloud deskewed")


    def preprocess_frame(self, frame_id=0):

        # T1 = get_time()
        # poses related
        cur_pose_init_guess = self.cur_pose_ref
        if self.processed_frame == 0: # initialize the first frame, no tracking yet
            if self.config.track_on:
                self.odom_poses.append(self.cur_pose_ref)
            if self.config.pgo_on:
                self.pgo_poses.append(self.cur_pose_ref)
            if self.gt_pose_provided and frame_id > 0: # not start with the first frame
                self.last_odom_tran = inv(self.poses_ref[frame_id-1]) @ self.cur_pose_ref # T_last<-cur
            self.travel_dist.append(0.)
            self.last_pose_ref = self.cur_pose_ref
        elif self.processed_frame > 0: 
            # pose initial guess
            last_translation = np.linalg.norm(self.last_odom_tran[:3,3])
            # if self.config.uniform_motion_on and not self.lose_track and last_translation > 0.2 * self.config.voxel_size_m: # apply uniform motion model here
            if self.config.uniform_motion_on and not self.lose_track: # apply uniform motion model here
                cur_pose_init_guess = self.last_pose_ref @ self.last_odom_tran # T_world<-cur = T_world<-last @ T_last<-cur
            else: # static initial guess
                cur_pose_init_guess = self.last_pose_ref

            if not self.config.track_on and self.gt_pose_provided:
                cur_pose_init_guess = self.poses_ref[frame_id]
            
            # pose initial guess tensor
            self.cur_pose_guess_torch = torch.tensor(cur_pose_init_guess, dtype=torch.float64, device=self.device)   

        if self.config.adaptive_range_on:
            pc_max_bound, _ = torch.max(self.cur_point_cloud_torch[:, :3], dim=0)
            pc_min_bound, _ = torch.min(self.cur_point_cloud_torch[:, :3], dim=0)

            min_x_range = min(torch.abs(pc_max_bound[0]),  torch.abs(pc_min_bound[0]))
            min_y_range = min(torch.abs(pc_max_bound[1]),  torch.abs(pc_min_bound[1]))
            max_x_y_min_range = max(min_x_range, min_y_range)

            crop_max_range = min(self.config.max_range, 2.*max_x_y_min_range)
        else:
            crop_max_range = self.config.max_range
        
        # adaptive
        train_voxel_m = (crop_max_range/self.config.max_range) * self.config.vox_down_m
        source_voxel_m = (crop_max_range/self.config.max_range) * self.config.source_vox_down_m

        # down sampling (together with the color and semantic entities)
        original_count = self.cur_point_cloud_torch.shape[0]
        if original_count < 10: # deal with missing data (invalid frame)
            print("[bold red]Not enough input point cloud, skip this frame[/bold red]") 
            if self.config.track_on:
                self.odom_poses.append(cur_pose_init_guess)
            if self.config.pgo_on:
                self.pgo_poses.append(cur_pose_init_guess)
            return False

        if self.config.rand_downsample:
            kept_count = int(original_count*self.config.rand_down_r)
            idx = torch.randint(0, original_count, (kept_count,), device=self.device)
        else:
            idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:,:3], train_voxel_m)
        self.cur_point_cloud_torch = self.cur_point_cloud_torch[idx]
        if self.cur_point_ts_torch is not None:
            self.cur_point_ts_torch = self.cur_point_ts_torch[idx]
        if self.cur_sem_labels_torch is not None:
            self.cur_sem_labels_torch = self.cur_sem_labels_torch[idx]
            self.cur_sem_labels_full = self.cur_sem_labels_full[idx]
        
        # T2 = get_time()

        # preprocessing, filtering
        if self.cur_sem_labels_torch is not None:
            self.cur_point_cloud_torch, self.cur_sem_labels_torch = filter_sem_kitti(self.cur_point_cloud_torch, self.cur_sem_labels_torch, self.cur_sem_labels_full,
                                                                                     True, self.config.filter_moving_object) 
        else:
            self.cur_point_cloud_torch, self.cur_point_ts_torch = crop_frame(self.cur_point_cloud_torch, self.cur_point_ts_torch, 
                                                                             self.config.min_z, self.config.max_z, 
                                                                             self.config.min_range, crop_max_range)

        if self.config.kitti_correction_on:
            self.cur_point_cloud_torch = intrinsic_correct(self.cur_point_cloud_torch, self.config.correction_deg)

        # T3 = get_time()

        # prepare for the registration
        if self.processed_frame > 0: 

            cur_source_torch = self.cur_point_cloud_torch.clone() # used for registration
            
            # source point voxel downsampling (for registration)
            idx = voxel_down_sample_torch(cur_source_torch[:,:3], source_voxel_m)
            cur_source_torch = cur_source_torch[idx]
            self.cur_source_points = cur_source_torch[:,:3]
            if self.config.color_on:
                self.cur_source_colors = cur_source_torch[:,3:]
            
            if self.cur_point_ts_torch is not None:
                cur_ts = self.cur_point_ts_torch.clone()
                cur_source_ts = cur_ts[idx]
            else:
                cur_source_ts = None

            # deskewing (motion undistortion) for source point cloud
            if self.config.deskew and not self.lose_track:
                self.cur_source_points = deskewing(self.cur_source_points, cur_source_ts, 
                                                   torch.tensor(self.last_odom_tran, device=self.device, dtype=self.dtype)) # T_last<-cur
                
            # print("# Source point for registeration : ", cur_source_torch.shape[0])
    
        # T4 = get_time()
        return True
    
    def update_odom_pose(self, cur_pose_torch: torch.tensor):
        # needed to be at least the second frame

        self.cur_pose_torch = cur_pose_torch.detach() # need to be out of the computation graph, used for mapping

        self.cur_pose_ref = self.cur_pose_torch.cpu().numpy()    
    
        self.last_odom_tran = inv(self.last_pose_ref) @ self.cur_pose_ref # T_last<-cur

        if tranmat_close_to_identity(self.last_odom_tran, 1e-3, self.config.voxel_size_m*0.1):
            self.stop_count += 1
        else:
            self.stop_count = 0
        
        if self.stop_count > self.config.stop_frame_thre:
            self.stop_status = True
            if not self.silence:
                print("Robot stopped")
        else:
            self.stop_status = False

        if self.config.pgo_on: # initialization the pgo pose
            self.pgo_poses.append(self.cur_pose_ref) 

        if self.odom_poses is not None:
            cur_odom_pose = self.odom_poses[-1] @ self.last_odom_tran # T_world<-cur
            self.odom_poses.append(cur_odom_pose)

        if len(self.travel_dist) > 0:
            cur_frame_travel_dist = np.linalg.norm(self.last_odom_tran[:3,3])
            if cur_frame_travel_dist > self.config.surface_sample_range_m * 40.0: # too large translation in one frame --> lose track
                self.lose_track = True 
                sys.exit("Too large translation in one frame, system failed") # FIXME
               
            accu_travel_dist = self.travel_dist[-1] + cur_frame_travel_dist
            self.travel_dist.append(accu_travel_dist)
            if not self.silence:
                print("Accumulated travel distance (m): %f" % accu_travel_dist)
        else: 
            sys.exit("This function needs to be used from at least the second frame")
        
        self.last_pose_ref = self.cur_pose_ref # update for the next frame

        # deskewing (motion undistortion using the estimated transformation) for the sampled points for mapping
        if self.config.deskew and not self.lose_track:
            self.cur_point_cloud_torch = deskewing(self.cur_point_cloud_torch, self.cur_point_ts_torch, 
                                         torch.tensor(self.last_odom_tran, device=self.device, dtype=self.dtype)) # T_last<-cur
        
        if self.lose_track:
            self.consecutive_lose_track_frame += 1
        else:
            self.consecutive_lose_track_frame = 0
        
        if self.consecutive_lose_track_frame > 10:
            sys.exit("Lose track for a long time, system failed") # FIXME

    def update_poses_after_pgo(self, pgo_cur_pose, pgo_poses):
        self.cur_pose_ref = pgo_cur_pose
        self.last_pose_ref = pgo_cur_pose # update for next frame
        self.pgo_poses = pgo_poses # update pgo pose

    def update_o3d_map(self):

        frame_down_torch = self.cur_point_cloud_torch # no futher downsample

        frame_o3d = o3d.geometry.PointCloud()
        frame_points_np = frame_down_torch[:,:3].detach().cpu().numpy().astype(np.float64)
    
        frame_o3d.points = o3d.utility.Vector3dVector(frame_points_np)

        # visualize or not
        # uncomment to visualize the dynamic mask
        if self.config.dynamic_filter_on and self.static_mask is not None:
            static_mask = self.static_mask.detach().cpu().numpy()
            frame_colors_np = np.ones_like(frame_points_np) * 0.7
            frame_colors_np[~static_mask,1:] = 0.0
            frame_colors_np[~static_mask,0] = 1.0
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_colors_np.astype(np.float64))

        frame_o3d = frame_o3d.transform(self.cur_pose_ref)

        if self.config.color_channel > 0:
            frame_colors_np = frame_down_torch[:,3:].detach().cpu().numpy().astype(np.float64)
            if self.config.color_channel == 1:
                frame_colors_np = np.repeat(frame_colors_np.reshape(-1, 1),3,axis=1) 
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_colors_np)
        elif self.cur_sem_labels_torch is not None:
            frame_label_torch = self.cur_sem_labels_torch
            frame_label_np = frame_label_torch.detach().cpu().numpy()
            frame_label_color = [sem_kitti_color_map[sem_label] for sem_label in frame_label_np]
            frame_label_color_np = np.asarray(frame_label_color, dtype=np.float64)/255.0
            frame_o3d.colors = o3d.utility.Vector3dVector(frame_label_color_np)

        self.cur_frame_o3d = frame_o3d 
        if self.cur_frame_o3d.has_points():
            self.cur_bbx = self.cur_frame_o3d.get_axis_aligned_bounding_box()

        cur_max_z = self.cur_bbx.get_max_bound()[-1]
        cur_min_z = self.cur_bbx.get_min_bound()[-1]

        bbx_center = self.cur_pose_ref[:3,3]
        bbx_min = np.array([bbx_center[0]-self.config.max_range, bbx_center[1]-self.config.max_range, cur_min_z])
        bbx_max = np.array([bbx_center[0]+self.config.max_range, bbx_center[1]+self.config.max_range, cur_max_z])

        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)

        # use the downsampled neural points here (done outside the class)

    def write_results_log(self, cur_frame: int, run_path: str):
        log_folder = 'log'
        frame_str = str(cur_frame)
        if self.config.track_on:
            write_traj_as_o3d(self.odom_poses, os.path.join(run_path, log_folder, frame_str+"_odom_poses.ply"))
        if self.config.pgo_on:
            write_traj_as_o3d(self.pgo_poses, os.path.join(run_path, log_folder, frame_str+"_slam_poses.ply"))
        if self.gt_pose_provided:
            write_traj_as_o3d(self.gt_poses, os.path.join(run_path, log_folder, frame_str+"_gt_poses.ply"))
                                   
    def write_results(self, run_path: str):
        odom_poses_out = apply_kitti_format_calib(self.odom_poses, self.calib['Tr'])
        write_kitti_format_poses(os.path.join(run_path, "odom_poses_"), odom_poses_out)
        write_tum_format_poses(os.path.join(run_path, "odom_poses_"), odom_poses_out)
        write_traj_as_o3d(self.odom_poses, os.path.join(run_path, "odom_poses.ply"))

        if self.config.pgo_on:
            slam_poses_out = apply_kitti_format_calib(self.pgo_poses, self.calib['Tr'])
            write_kitti_format_poses(os.path.join(run_path, "slam_poses_"), slam_poses_out)
            write_tum_format_poses(os.path.join(run_path, "slam_poses_"), slam_poses_out)
            write_traj_as_o3d(self.pgo_poses, os.path.join(run_path, "slam_poses.ply"))
        
        if self.gt_pose_provided:
            write_traj_as_o3d(self.gt_poses, os.path.join(run_path, "gt_poses.ply"))

        time_table = np.array(self.time_table)
        mean_time_s = np.sum(time_table)/self.processed_frame*1.0
        if not self.silence:
            print("Consuming time per frame        (s):", f"{mean_time_s:.3f}")
            print("Calculated over %d frames" % self.processed_frame)
        np.save(os.path.join(run_path, "time_table.npy"), time_table) # save detailed time table
        if self.config.o3d_vis_on:
            plot_timing_detail(time_table, os.path.join(run_path, "time_details.png"), self.config.pgo_on)

        pose_eval = None
        
        # evaluation report
        if self.gt_pose_provided and len(self.gt_poses) == len(self.odom_poses):
            print("Odometry evaluation:")
            avg_tra, avg_rot = relative_error(self.gt_poses, self.odom_poses)
            ate_rot, ate_trans, align_mat = absolute_error(self.gt_poses, self.odom_poses, self.config.eval_traj_align)
            if avg_tra == 0: # for rgbd dataset (shorter sequence)
                print("Absoulte Trajectory Error      (cm):", f"{ate_trans*100.0:.3f}")
            else:
                print("Average Translation Error       (%):", f"{avg_tra:.3f}")
                print("Average Rotational Error (deg/100m):", f"{avg_rot*100.0:.3f}")
                print("Absoulte Trajectory Error       (m):", f"{ate_trans:.3f}")

            if self.config.wandb_vis_on:
                wandb_log_content = {'Average Translation Error [%]': avg_tra, 'Average Rotational Error [deg/m]': avg_rot,
                                     'Absoulte Trajectory Error [m]': ate_trans, 'Absoulte Rotational Error [deg]': ate_rot,
                                     'Consuming time per frame [s]': mean_time_s} 
                wandb.log(wandb_log_content)

            if self.config.pgo_on and len(self.gt_poses) == len(self.pgo_poses):
                print("SLAM evaluation:")
                avg_tra_slam, avg_rot_slam = relative_error(self.gt_poses, self.pgo_poses)
                ate_rot_slam, ate_trans_slam, align_mat_slam = absolute_error(self.gt_poses, self.pgo_poses, self.config.eval_traj_align)
                if avg_tra_slam == 0: # for rgbd dataset (shorter sequence)
                    print("Absoulte Trajectory Error      (cm):", f"{ate_trans_slam*100.0:.3f}")
                else:
                    print("Average Translation Error       (%):", f"{avg_tra_slam:.3f}")
                    print("Average Rotational Error (deg/100m):", f"{avg_rot_slam*100.0:.3f}")
                    print("Absoulte Trajectory Error       (m):", f"{ate_trans_slam:.3f}")

                if self.config.wandb_vis_on:
                    wandb_log_content = {'SLAM Average Translation Error [%]': avg_tra_slam, 'SLAM Average Rotational Error [deg/m]': avg_rot_slam, 'SLAM Absoulte Trajectory Error [m]': ate_trans_slam, 'SLAM Absoulte Rotational Error [deg]': ate_rot_slam} 
                    wandb.log(wandb_log_content)
            
            csv_columns = ['Average Translation Error [%]', 'Average Rotational Error [deg/m]', 'Absoulte Trajectory Error [m]', 'Absoulte Rotational Error [deg]', "Consuming time per frame [s]", "Frame count"]
            pose_eval = [{csv_columns[0]: avg_tra, csv_columns[1]: avg_rot, csv_columns[2]: ate_trans, csv_columns[3]: ate_rot, csv_columns[4]: mean_time_s, csv_columns[5]: int(self.processed_frame)}]
            if self.config.pgo_on:
                slam_eval_dict = {csv_columns[0]: avg_tra_slam, csv_columns[1]: avg_rot_slam, csv_columns[2]: ate_trans_slam, csv_columns[3]: ate_rot_slam, csv_columns[4]: mean_time_s, csv_columns[5]: int(self.processed_frame)}
                pose_eval.append(slam_eval_dict)
            output_csv_path = os.path.join(run_path, "pose_eval.csv")
            try:
                with open(output_csv_path, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in pose_eval:
                        writer.writerow(data)
            except IOError:
                print("I/O error")
            
            if self.config.o3d_vis_on: # x service issue for remote server
                output_traj_plot_path_2d = os.path.join(run_path, "traj_plot_2d.png")
                output_traj_plot_path_3d = os.path.join(run_path, "traj_plot_3d.png")
                # trajectory not aligned yet in the plot
                if self.config.pgo_on:
                    plot_trajectories(output_traj_plot_path_2d, self.pgo_poses, self.gt_poses, self.odom_poses, plot_3d=False) 
                    plot_trajectories(output_traj_plot_path_3d, self.pgo_poses, self.gt_poses, self.odom_poses, plot_3d=True)   
                else:
                    plot_trajectories(output_traj_plot_path_2d, self.odom_poses, self.gt_poses, plot_3d=False)
                    plot_trajectories(output_traj_plot_path_3d, self.odom_poses, self.gt_poses, plot_3d=True)

        return pose_eval
    
    def write_merged_point_cloud(self, run_path: str):
        
        print("Begin to replay the dataset ...")

        o3d_device = o3d.core.Device("CPU:0")
        o3d_dtype = o3d.core.float32
        map_out_o3d = o3d.t.geometry.PointCloud(o3d_device)
        map_points_np = np.empty((0, 3))
        map_intensity_np = np.empty(0)
        map_color_np = np.empty((0, 3))

        use_frame_id = 0
        for frame_id in tqdm(range(self.total_pc_count)): # frame id as the idx of the frame in the data folder without skipping
            if (frame_id < self.config.begin_frame or frame_id > self.config.end_frame or frame_id % self.config.every_frame != 0):
                continue

            self.read_frame(frame_id)

            if self.config.kitti_correction_on:
                self.cur_point_cloud_torch = intrinsic_correct(self.cur_point_cloud_torch, self.config.correction_deg)

            if self.config.deskew and use_frame_id < self.processed_frame-1:
                if self.config.track_on:
                    tran_in_frame = np.linalg.inv(self.odom_poses[use_frame_id+1]) @ self.odom_poses[use_frame_id]
                elif self.gt_pose_provided:
                    tran_in_frame = np.linalg.inv(self.gt_poses[use_frame_id+1]) @ self.gt_poses[use_frame_id]
                self.cur_point_cloud_torch = deskewing(self.cur_point_cloud_torch, self.cur_point_ts_torch, 
                                                       torch.tensor(tran_in_frame, device=self.device, dtype=torch.float64)) # T_last<-cur
            
            down_vox_m = self.config.vox_down_m
            idx = voxel_down_sample_torch(self.cur_point_cloud_torch[:,:3], down_vox_m)
            
            frame_down_torch = self.cur_point_cloud_torch[idx]

            frame_down_torch, _ = crop_frame(frame_down_torch, None, self.config.min_z, self.config.max_z, self.config.min_range, self.config.max_range)
                                                       
            if self.config.pgo_on:
                cur_pose_torch = torch.tensor(self.pgo_poses[use_frame_id], device=self.device, dtype=torch.float64)
            elif self.config.track_on:
                cur_pose_torch = torch.tensor(self.odom_poses[use_frame_id], device=self.device, dtype=torch.float64)
            elif self.gt_pose_provided:
                cur_pose_torch = torch.tensor(self.gt_poses[use_frame_id], device=self.device, dtype=torch.float64)
            frame_down_torch[:,:3] = transform_torch(frame_down_torch[:,:3], cur_pose_torch) 

            frame_points_np = frame_down_torch[:,:3].detach().cpu().numpy()
            map_points_np = np.concatenate((map_points_np, frame_points_np), axis=0)
            if self.config.color_channel == 1:
                frame_intensity_np = frame_down_torch[:,3].detach().cpu().numpy()
                map_intensity_np = np.concatenate((map_intensity_np, frame_intensity_np), axis=0)
            elif self.config.color_channel == 3:
                frame_color_np = frame_down_torch[:,3:].detach().cpu().numpy()
                map_color_np = np.concatenate((map_color_np, frame_color_np), axis=0)
            
            use_frame_id += 1
        
        print("Replay done")

        map_out_o3d.point["positions"] =  o3d.core.Tensor(map_points_np, o3d_dtype, o3d_device)
        if self.config.color_channel == 1:
            map_out_o3d.point["intensity"] =  o3d.core.Tensor(np.expand_dims(map_intensity_np, axis=1), o3d_dtype, o3d_device)
        elif self.config.color_channel == 3:
            map_out_o3d.point["colors"] =  o3d.core.Tensor(map_color_np, o3d_dtype, o3d_device)
        
        # print("Estimate normal")
        # map_out_o3d.estimate_normals(max_nn=20)

        if run_path is not None:
            print("Output merged point cloud map")
            o3d.t.io.write_point_cloud(os.path.join(run_path, "map", "merged_point_cloud.ply"), map_out_o3d)


def read_point_cloud(filename: str, color_channel: int = 0, bin_channel_count: int = 4) -> np.ndarray:

    # read point cloud from either (*.ply, *.pcd, *.las) or (kitti *.bin) format
    if ".bin" in filename:
        # we also read the intensity channel here
        data_loaded = np.fromfile(filename, dtype=np.float32)

        # print(data_loaded)
        # for NCLT, it's a bit different from KITTI format, check: http://robots.engin.umich.edu/nclt/python/read_vel_sync.py
        # for KITTI, bin_channel_count = 4
        # for Boreas, bin_channel_count = 6 # (x,y,z,i,r,ts)
        points = data_loaded.reshape((-1, bin_channel_count))
        # print(points)
        ts = None
        if bin_channel_count == 6:
            ts = points[:, -1]
        
    elif ".ply" in filename:
        pc_load = o3d.t.io.read_point_cloud(filename)
        pc_load = {k: v.numpy() for k,v in pc_load.point.items() }
        
        keys = list(pc_load.keys())
        # print("available attributes:", keys)

        points = pc_load['positions']

        if 't' in keys:
            ts = pc_load['t'] * 1e-8
        elif 'timestamp' in keys:
            ts = pc_load['timestamp']
        else:
            ts = None

        if 'colors' in keys and color_channel == 3:           
            colors = pc_load['colors'] # if they are available
            points = np.hstack((points, colors))
        elif 'intensity' in keys and color_channel == 1:           
            intensity = pc_load['intensity'] # if they are available
            # print(intensity)
            points = np.hstack((points, intensity))
    elif ".pcd" in filename: # currently cannot be readed by o3d.t.io
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points, dtype=np.float64)
        ts = None
    elif ".las" in filename: # use laspy
        import laspy
        with laspy.open(filename) as fh:     
            las = fh.read()
            x = (las.points.X * las.header.scale[0] + las.header.offset[0])
            y = (las.points.Y * las.header.scale[1] + las.header.offset[1])
            z = (las.points.Z * las.header.scale[2] + las.header.offset[2])
            points = np.array([x, y, z], dtype=np.float64).T
            if color_channel == 1:
                intensity = np.array(las.points.intensity).reshape(-1, 1)
                # print(intensity)
                points = np.hstack((points, intensity))
            ts = None # TODO, also read the point-wise timestamp for las point cloud
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *pcd, *ply, *las and *bin)")

    # print("Loaded ", np.shape(points)[0], " points")

    return points, ts # as np

# now we only support semantic kitti format dataset
def read_semantic_point_label(bin_filename: str, label_filename: str, color_on: bool = False):

    # read point cloud (kitti *.bin format)
    if ".bin" in bin_filename:
        # we also read the intensity channel here
        points = np.fromfile(bin_filename, dtype=np.float32).reshape(-1, 4)
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *bin)")

    # read point cloud labels (*.label format)
    if ".label" in label_filename:
        labels = np.fromfile(label_filename, dtype=np.uint32).reshape(-1)
    else:
        sys.exit("The format of the imported point labels is wrong (support only *label)")

    labels = labels & 0xFFFF # only take the semantic part 

    # get the reduced label [0-20] 
    labels_reduced = np.vectorize(sem_map_function)(labels).astype(np.int32)  # fast version

    # original label [0-255]
    labels = np.array(labels, dtype=np.int32)

    return points, labels, labels_reduced # as np

def read_kitti_format_calib(filename: str):
    """ 
        read calibration file (with the kitti format)
        returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)

    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib

def read_kitti_format_poses(filename: str) -> List[np.ndarray]:
    """ 
        read pose file (with the kitti format)
        returns -> list, transformation before calibration transformation
    """
    pose_file = open(filename)

    poses = []

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(pose)

    pose_file.close()
    return poses   

def read_tum_format_poses_csv(filename: str) -> List[np.ndarray]:
    # now it supports csv file only (TODO)
    from pyquaternion import Quaternion

    poses = []
    with open(filename, mode="r") as f:
        reader = csv.reader(f)
        # get header and change timestamp label name
        header = next(reader)
        header[0] = "ts"
        # Convert string odometry to numpy transfor matrices
        for row in reader:
            odom = {l: row[i] for i, l in enumerate(header)}
            # Translarion and rotation quaternion as numpy arrays
            trans = np.array([float(odom[l]) for l in ["tx", "ty", "tz"]])
            quat_ijkw = np.array([float(odom[l]) for l in ["qx", "qy", "qz", "qw"]])
            quat = Quaternion(quat_ijkw[3], quat_ijkw[0], quat_ijkw[1], quat_ijkw[2]) # quaternion needs to use the w, i, j, k order , you need to switch as bit
            rot = quat.rotation_matrix
            # Build numpy transform matrix
            odom_tf = np.eye(4)
            odom_tf[0:3, 3] = trans
            odom_tf[0:3, 0:3] = rot
            # Add transform to timestamp indexed dictionary
            # odom_tfs[odom["ts"]] = odom_tf
            poses.append(odom_tf)

    return poses

# copyright: Nacho et al. KISS-ICP
def write_kitti_format_poses(filename: str, poses: List[np.ndarray]):
    def _to_kitti_format(poses: np.ndarray) -> np.ndarray:
        return np.array([np.concatenate((pose[0], pose[1], pose[2])) for pose in poses])

    np.savetxt(fname=f"{filename}_kitti.txt", X=_to_kitti_format(poses))

# copyright: Nacho et al. KISS-ICP
def write_tum_format_poses(filename: str, poses: List[np.ndarray], timestamps = None):
    from pyquaternion import Quaternion
    def _to_tum_format(poses, timestamps = None):
        tum_data = []
        with contextlib.suppress(ValueError):
            for idx in range(len(poses)):
                tx, ty, tz = poses[idx][:3, -1].flatten()
                qw, qx, qy, qz = Quaternion(matrix=poses[idx], atol=0.01).elements
                if timestamps is None:
                    tum_data.append([idx, tx, ty, tz, qx, qy, qz, qw]) # index as the ts
                else:
                    tum_data.append([float(timestamps[idx]), tx, ty, tz, qx, qy, qz, qw])
        return np.array(tum_data).astype(np.float64)

    np.savetxt(fname=f"{filename}_tum.txt", X=_to_tum_format(poses, timestamps), fmt="%.4f")

def apply_kitti_format_calib(poses: List[np.ndarray], calib_T_cl) -> List[np.ndarray]:
    """Converts from Velodyne to Camera Frame (# T_camera<-lidar)""" 
    poses_calib = []
    if calib_T_cl is not None:
        for pose in poses:
            poses_calib.append(calib_T_cl @ pose @ inv(calib_T_cl))
    return poses_calib 
    
# torch version
def crop_frame(points: torch.tensor, ts: torch.tensor, 
               min_z_th=-3.0, max_z_th=100.0, min_range=2.75, max_range=100.0):
    dist = torch.norm(points[:,:3], dim=1)
    filtered_idx = (dist > min_range) & (dist < max_range) & (points[:, 2] > min_z_th) & (points[:, 2] < max_z_th)
    points = points[filtered_idx]
    if ts is not None:
        ts = ts[filtered_idx]
    return points, ts

# torch version
def intrinsic_correct(points: torch.tensor, correct_deg=0.):

    # # This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
    # # the original idea and part of the implementation is taking from CT-ICP(Although IMLS-SLAM
    # # Originally introduced the calibration factor)
    # We set the correct_deg = 0.195 deg for KITTI odom dataset, inline with MULLS #issue 11
    if correct_deg == 0.:
        return points

    dist = torch.norm(points[:,:3], dim=1)
    kitti_var_vertical_ang = correct_deg / 180. * math.pi
    v_ang = torch.asin(points[:, 2] / dist)
    v_ang_c = v_ang + kitti_var_vertical_ang
    hor_scale = torch.cos(v_ang_c) / torch.cos(v_ang)
    points[:, 0] *= hor_scale
    points[:, 1] *= hor_scale
    points[:, 2] = dist * torch.sin(v_ang_c)

    return points

# now only work for semantic kitti format dataset # torch version
def filter_sem_kitti(points: torch.tensor, sem_labels_reduced: torch.tensor, sem_labels: torch.tensor,
                     filter_outlier = True, filter_moving = False):
    
    # sem_labels_reduced is the reduced labels for mapping (20 classes for semantic kitti)
    # sem_labels is the original semantic label (0-255 for semantic kitti)
    
    if filter_outlier: # filter the outliers according to semantic labels
        inlier_mask = (sem_labels > 1) # not outlier
    else:
        inlier_mask = (sem_labels >= 0) # all

    if filter_moving:
        static_mask = sem_labels < 100 # only for semantic KITTI dataset
        inlier_mask = inlier_mask & static_mask

    points = points[inlier_mask]
    sem_labels_reduced = sem_labels_reduced[inlier_mask]

    return points, sem_labels_reduced

def write_traj_as_o3d(poses: List[np.ndarray], path):

    o3d_pcd = o3d.geometry.PointCloud()
    poses_np = np.array(poses, dtype=np.float64)
    o3d_pcd.points = o3d.utility.Vector3dVector(poses_np[:,:3,3])

    ts_np = np.linspace(0, 1, len(poses))
    color_map = cm.get_cmap('jet')
    ts_color = color_map(ts_np)[:, :3].astype(np.float64)
    o3d_pcd.colors = o3d.utility.Vector3dVector(ts_color)

    if path is not None:
        o3d.io.write_point_cloud(path, o3d_pcd)

    return o3d_pcd