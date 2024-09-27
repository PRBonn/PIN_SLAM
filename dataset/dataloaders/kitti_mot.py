# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to mse, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE msE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import importlib
import os
from collections import namedtuple

import cv2
import numpy as np
import open3d as o3d

# https://www.cvlibs.net/datasets/kitti/setup.php
# https://www.cvlibs.net/datasets/kitti/eval_tracking.php # KITTI MOT dataset

# TODO

class KITTIMOTDataset:
    def __init__(self, data_dir, sequence: str, *_, **__):
        
        self.sequence_id = str(sequence).zfill(4)
        # include the data dir as kitti_mot/training/
        # self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)

        self.data_split = "training" # training or testing
        
        self.velodyne_dir = os.path.join(data_dir, "data_tracking_velodyne", self.data_split, "velodyne", self.sequence_id) 
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "/*.bin"))
        scan_count = len(self.scan_files)
        # print(scan_count)

        # img related
        self.load_img = False # default
        self.use_only_colorized_points = True

        # cam 2 (color)
        self.img2_dir = os.path.join(data_dir, "data_tracking_image_2", self.data_split, "image_02", self.sequence_id) 
        self.img2_files = sorted(glob.glob(self.img2_dir + "/*.png"))
        img2_count = len(self.img2_files)
        if img2_count == scan_count:
            self.image_available = True
        else:
            self.image_available = False

        # cam 3 (color)
        self.img3_dir = os.path.join(data_dir, "data_tracking_image_3", self.data_split, "image_03", self.sequence_id) 
        self.img3_files = sorted(glob.glob(self.img3_dir + "/*.png"))
        img3_count = len(self.img3_files)

        # calib files
        calib_file_path = os.path.join(data_dir, "data_tracking_calib", self.data_split, "calib", self.sequence_id+".txt")
        calib_mats = self.tracking_calib_from_txt(calib_file_path) 
        K_mat2 = calib_mats["K2"]
        K_mat3 = calib_mats["K3"]
        T_c2_l = calib_mats["T_c2_l"] 
        T_c3_l = calib_mats["T_c3_l"] 

        self.main_cam_name = "cam2" # cam2 as main cam

        if self.image_available: # now we use cam2 (left color)
            H, W = 375, 1242
            
            self.T_c_l_mats = {self.main_cam_name: T_c2_l} # use rectified frame or not?
            self.K_mats = {self.main_cam_name: K_mat2}
            self.cam_widths = {self.main_cam_name: W}
            self.cam_heights = {self.main_cam_name: H}

            self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
            self.intrinsic.set_intrinsics(
                                        height=H,
                                        width=W,
                                        fx=K_mat2[0,0],
                                        fy=K_mat2[1,1],
                                        cx=K_mat2[0,2],
                                        cy=K_mat2[1,2])

            self.extrinsic = T_c2_l

        # get poses in IMU frame by loading oxts data
        oxts_file_path = os.path.join(data_dir, "data_tracking_oxts", self.data_split, "oxts", self.sequence_id+".txt")
        poses_imu_w_tracking, _, _ = self.get_poses_calibration(data_dir, oxts_file_path)  # (n_frames, 4, 4) imu pose

        # GT poses in LiDAR frame
        Tr_lidar_imu = calib_mats["T_l_i"]
        Tr_imu_lidar = np.linalg.inv(Tr_lidar_imu)
        self.gt_poses = Tr_lidar_imu @ poses_imu_w_tracking @ Tr_imu_lidar 


    def __getitem__(self, idx):
        
        points = self.scans(idx)
        point_ts = self.get_timestamps(points)

        if self.load_img and self.image_available:
            # print("load img")
            img = self.read_img(self.img2_files[idx])
        
            points_rgb = np.ones_like(points)

            # project to the image plane to get the corresponding color
            points_rgb, depth_map = self.project_points_to_cam(points, points_rgb, img, 
                self.T_c_l_mats[self.main_cam_name], self.K_mats[self.main_cam_name])

            if self.use_only_colorized_points:
                with_rgb_mask = (points_rgb[:, 3] == 0)
                points = points[with_rgb_mask]
                points_rgb = points_rgb[with_rgb_mask]
                point_ts = point_ts[with_rgb_mask]

            # we skip the intensity here for now (and also the color mask)
            points = np.hstack((points[:,:3], points_rgb[:,:3]))

            img = np.concatenate((img, np.expand_dims(depth_map, axis=-1)), axis=-1) # 4 channels
            img_dict = {self.main_cam_name: img}

            frame_data = {"points": points, "point_ts": point_ts, "img": img_dict}
        else:
            frame_data = {"points": points, "point_ts": point_ts}

        return frame_data

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
        return points # N, 4
    
    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    # velodyne lidar
    @staticmethod
    def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        timestamps = 0.5 * (yaw / np.pi + 1.0)
        return timestamps

    def load_poses(self, poses_file):
        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return poses
    

    def tracking_calib_from_txt(self, calibration_path):
        # borrow from https://github.com/fudan-zvg/PVG/blob/main/scene/kittimot_loader.py
        """
        Extract tracking calibration information from a KITTI tracking calibration file.

        This function reads a KITTI tracking calibration file and extracts the relevant
        calibration information, including projection matrices and transformation matrices
        for camera, LiDAR, and IMU coordinate systems.

        Args:
            calibration_path (str): Path to the KITTI tracking calibration file.

        Returns:
            dict: A dictionary containing the following calibration information:
                P0, P1, P2, P3 (np.array): 3x4 projection matrices for the cameras. (already the rectified ones)
                Tr_cam2camrect (np.array): 4x4 transformation matrix from camera to rectified camera coordinates.
                Tr_velo2cam (np.array): 4x4 transformation matrix from LiDAR to camera coordinates.
                Tr_imu2velo (np.array): 4x4 transformation matrix from IMU to LiDAR coordinates.
        """
        # Read the calibration file
        f = open(calibration_path)
        calib_str = f.read().splitlines()

        def kitti_string_to_float(str):
            return float(str.split("e")[0]) * 10 ** int(str.split("e")[1])

        # Process the calibration data
        calibs = []
        for calibration in calib_str:
            calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

        # Extract the projection matrices
        P0 = np.reshape(calibs[0], [3, 4])
        P1 = np.reshape(calibs[1], [3, 4])
        P2 = np.reshape(calibs[2], [3, 4])
        P3 = np.reshape(calibs[3], [3, 4])

        K2 = P2[:3,:3]
        K3 = P3[:3,:3]
        t2 = P2[:, 3]
        t3 = P3[:, 3]

        T_c2_r = np.eye(4)
        T_c2_r[:3, 3] = np.dot(np.linalg.inv(K2), t2)

        T_c3_r = np.eye(4)
        T_c3_r[:3, 3] = np.dot(np.linalg.inv(K3), t3)

        # Extract the transformation matrix for camera to rectified camera coordinates
        T_r_c = np.eye(4)
        R_r_c = np.reshape(calibs[4], [3, 3])
        T_r_c[:3, :3] = R_r_c

        # Extract the transformation matrices for LiDAR to camera and IMU to LiDAR coordinates
        T_c_l = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        T_l_i = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

        T_c2_l = T_c2_r @ T_r_c @ T_c_l
        T_c3_l = T_c3_r @ T_r_c @ T_c_l

        return {
            "K2": K2,
            "K3": K3,
            "T_c2_l": T_c2_l,
            "T_c3_l": T_c3_l,
            "T_r_c": T_r_c,
            "T_l_i": T_l_i,
        }
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

        points = self.intrinsic_correct(points) # FIXME: only for kitti

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        # project to image space
        u, v, depth= self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        img_height, img_width, _ = np.shape(img)

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width))
        depth_img = np.zeros((img_height, img_width, 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
        # visualize points within 30 meters
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid] = depth[mask]

        # print(np.shape(points_rgb))

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb, depth_map
    
    def persepective_cam2image(self, points, K_mat):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(K_mat[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth
    
    def get_poses_calibration(self, basedir, oxts_path_tracking=None, selected_frames=None):
        # reference: https://github.com/fudan-zvg/PVG/blob/main/scene/kittimot_loader.py

        """
        Extract poses and calibration information from the KITTI dataset.

        This function processes the OXTS data (GPS/IMU) and extracts the
        pose information (translation and rotation) for each frame. It also
        retrieves the calibration information (transformation matrices and focal length)
        required for further processing.

        Args:
            basedir (str): The base directory containing the KITTI dataset.
            oxts_path_tracking (str, optional): Path to the OXTS data file for tracking sequences.
                If not provided, the function will look for OXTS data in the basedir.
            selected_frames (list, optional): A list of frame indices to process.
                If not provided, all frames in the dataset will be processed.

        Returns:
            tuple: A tuple containing the following elements:
                poses (np.array): An array of 4x4 pose matrices representing the vehicle's
                    position and orientation for each frame (IMU pose).
                calibrations (dict): A dictionary containing the transformation matrices
                    and focal length obtained from the calibration files.
                focal (float): The focal length of the left camera.
        """

        def oxts_to_pose(oxts):
            """
            OXTS (Oxford Technical Solutions) data typically refers to the data generated by an Inertial and GPS Navigation System (INS/GPS) that is used to provide accurate position, orientation, and velocity information for a moving platform, such as a vehicle. In the context of the KITTI dataset, OXTS data is used to provide the ground truth for the vehicle's trajectory and 6 degrees of freedom (6-DoF) motion, which is essential for evaluating and benchmarking various computer vision and robotics algorithms, such as visual odometry, SLAM, and object detection.

            The OXTS data contains several important measurements:

            1. Latitude, longitude, and altitude: These are the global coordinates of the moving platform.
            2. Roll, pitch, and yaw (heading): These are the orientation angles of the platform, usually given in Euler angles.
            3. Velocity (north, east, and down): These are the linear velocities of the platform in the local navigation frame.
            4. Accelerations (ax, ay, az): These are the linear accelerations in the platform's body frame.
            5. Angular rates (wx, wy, wz): These are the angular rates (also known as angular velocities) of the platform in its body frame.

            In the KITTI dataset, the OXTS data is stored as plain text files with each line corresponding to a timestamp. Each line in the file contains the aforementioned measurements, which are used to compute the ground truth trajectory and 6-DoF motion of the vehicle. This information can be further used for calibration, data synchronization, and performance evaluation of various algorithms.
            """
            poses = []

            def latlon_to_mercator(lat, lon, s):
                """
                Converts latitude and longitude coordinates to Mercator coordinates (x, y) using the given scale factor.

                The Mercator projection is a widely used cylindrical map projection that represents the Earth's surface
                as a flat, rectangular grid, distorting the size of geographical features in higher latitudes.
                This function uses the scale factor 's' to control the amount of distortion in the projection.

                Args:
                    lat (float): Latitude in degrees, range: -90 to 90.
                    lon (float): Longitude in degrees, range: -180 to 180.
                    s (float): Scale factor, typically the cosine of the reference latitude.

                Returns:
                    list: A list containing the Mercator coordinates [x, y] in meters.
                """
                r = 6378137.0  # the Earth's equatorial radius in meters
                x = s * r * ((np.pi * lon) / 180)
                y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
                return [x, y]
            
            def get_rotation(roll, pitch, heading):
                s_heading = np.sin(heading)
                c_heading = np.cos(heading)
                rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

                s_pitch = np.sin(pitch)
                c_pitch = np.cos(pitch)
                rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

                s_roll = np.sin(roll)
                c_roll = np.cos(roll)
                rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

                rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

                return rot
            
            def invert_transformation(rot, t):
                t = np.matmul(-rot.T, t)
                inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
                return np.concatenate([inv_translation, np.array([[0.0, 0.0, 0.0, 1.0]])])

            # Compute the initial scale and pose based on the selected frames
            if selected_frames is None:
                lat0 = oxts[0][0]
                scale = np.cos(lat0 * np.pi / 180)
                pose_0_inv = None
            else:
                oxts0 = oxts[selected_frames[0][0]]
                lat0 = oxts0[0]
                scale = np.cos(lat0 * np.pi / 180)

                pose_i = np.eye(4)

                [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
                z = oxts0[2]
                translation = np.array([x, y, z])
                rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
                pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

            # Iterate through the OXTS data and compute the corresponding pose matrices
            for oxts_val in oxts:
                pose_i = np.zeros([4, 4])
                pose_i[3, 3] = 1

                [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
                z = oxts_val[2]
                translation = np.array([x, y, z])

                roll = oxts_val[3]
                pitch = oxts_val[4]
                heading = oxts_val[5]
                rotation = get_rotation(roll, pitch, heading)  # (3,3)

                pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)  # (4, 4)
                if pose_0_inv is None:
                    pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

                pose_i = np.matmul(pose_0_inv, pose_i)
                poses.append(pose_i)

            return np.array(poses)

        # If there is no tracking path specified, use the default path
        if oxts_path_tracking is None:
            oxts_path = os.path.join(basedir, "oxts/data")
            oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
            calibration_path = os.path.dirname(basedir)

            calibrations = calib_from_txt(calibration_path)

            focal = calibrations[4]

            poses = oxts_to_pose(oxts)

        # If a tracking path is specified, use it to load OXTS data and compute the poses
        else:
            oxts_tracking = np.loadtxt(oxts_path_tracking)
            poses = oxts_to_pose(oxts_tracking)  # (n_frames, 4, 4)
            calibrations = None
            focal = None
            # Set velodyne close to z = 0
            # poses[:, 2, 3] -= 0.8

        # Return the poses, calibrations, and focal length
        return poses, calibrations, focal

    # only for kitti
    def intrinsic_correct(self, points, correct_deg=0.195):
        corrected_points = np.copy(points)
        dist = np.linalg.norm(points[:, :3], axis=1)
        kitti_var_vertical_ang = correct_deg / 180.0 * np.pi
        v_ang = np.arcsin(points[:, 2] / dist)
        v_ang_c = v_ang + kitti_var_vertical_ang
        hor_scale = np.cos(v_ang_c) / np.cos(v_ang)
        corrected_points[:, 0] *= hor_scale
        corrected_points[:, 1] *= hor_scale
        corrected_points[:, 2] = dist * np.sin(v_ang_c)
        return corrected_points