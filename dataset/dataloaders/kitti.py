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
import sys

import numpy as np

# https://www.cvlibs.net/datasets/kitti/setup.php

class KITTIOdometryDataset:
    def __init__(self, data_dir, sequence: str, *_, **__):

        self.sequence_id = str(sequence).zfill(2)
        self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
        
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        scan_count = len(self.scan_files)

        # point cloud semantic labels (from semantic kitti) # TODO
        self.sem_labels_dir = os.path.join(self.kitti_sequence_dir, "labels/")
        self.sem_files = sorted(glob.glob(self.sem_labels_dir + "*.label"))
        sem_file_count = len(self.sem_files)
        if sem_file_count == scan_count:
            self.sem_available = True
        else:
            self.sem_available = False

        self.load_img: bool = False
        self.use_only_colorized_points: bool = True

        # cam 2 (color)
        self.img2_dir = os.path.join(self.kitti_sequence_dir, "image_2/")
        self.img2_files = sorted(glob.glob(self.img2_dir + "*.png"))
        img2_count = len(self.img2_files)
        if img2_count == scan_count:
            self.image_available = True
            try:
                self.cv2 = importlib.import_module("cv2")
            except ModuleNotFoundError: # TODO
                print(
                    'img files requires opencv-python and is not installed on your system run "pip install opencv-python"'
                )
                sys.exit(1)
        else:
            self.image_available = False

        # cam 3 (color)
        self.img3_dir = os.path.join(self.kitti_sequence_dir, "image_3/")
        self.img3_files = sorted(glob.glob(self.img3_dir + "*.png"))

        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))

        calib_data = self._load_calib() # load all calib first

        if self.image_available: # now we use cam2 (left color)
            self.left_cam_name = "cam2"
            self.T_c_l_mats = {self.left_cam_name: calib_data['T_cam2_velo']}
            self.K_mats = {self.left_cam_name: calib_data["K_cam2"]}

        # Load GT Poses (if available)
        if int(sequence) < 11:
            self.poses_fn = os.path.join(data_dir, f"poses/{self.sequence_id}.txt")
            if not os.path.exists(self.poses_fn):
                self.poses_fn = os.path.join(self.kitti_sequence_dir, f"poses.txt")
            self.gt_poses = self.load_poses(self.poses_fn)

    def __getitem__(self, idx):
        
        points = self.scans(idx)
        point_ts = self.get_timestamps(points)

        if self.image_available and self.load_img:
            img = self.read_img(self.img2_files[idx])
            img_dict = {self.left_cam_name: img}

            points_rgb = np.ones_like(points)

            # project to the image plane to get the corresponding color
            points_rgb = self.project_points_to_cam(points, points_rgb, img, self.T_c_l_mats[self.left_cam_name], self.K_mats[self.left_cam_name])

            if self.use_only_colorized_points:
                with_rgb_mask = (points_rgb[:, 3] == 0)
                points = points[with_rgb_mask]
                points_rgb = points_rgb[with_rgb_mask]
                point_ts = point_ts[with_rgb_mask]

            # we skip the intensity here for now (and also the color mask)
            points = np.hstack((points[:,:3], points_rgb[:,:3]))

            frame_data = {"points": points, "point_ts": point_ts, "img": img_dict}
        else:
            frame_data = {"points": points, "point_ts": point_ts}

        return frame_data

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame"""
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        return Tr @ poses @ np.linalg.inv(Tr)

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
        return points # N, 4
    
    def read_img(self, img_file: str):
        img = self.cv2.imread(img_file)
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
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
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right

        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return _lidar_pose_gt(poses)

    def get_frames_timestamps(self) -> np.ndarray:
        timestamps = np.loadtxt(os.path.join(self.kitti_sequence_dir, "times.txt")).reshape(-1, 1)
        return timestamps

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict
    
    # partially from kitti360 dev kit
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

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
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        # depth mask
        min_depth = 0.5
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid] = depth[mask]

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb
    
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
    
    # from pykitti
    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        filedata = self.calibration

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        return data
