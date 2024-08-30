# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import importlib
import os
from collections import namedtuple
from pathlib import Path
import sys

import numpy as np

# Reference:
# https://www.cvlibs.net/datasets/kitti-360/documentation.php

class KITTI360Dataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):

        try:
            self.cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError:
            print(
                'img files requires opencv-python and is not installed on your system run "pip install opencv-python"'
            )
            sys.exit(1)

        self.sequence_id = str(sequence).zfill(2)

        seq_str = f"2013_05_28_drive_{str(sequence).zfill(4)}_sync/"

        self.load_img: bool = False
        self.use_only_colorized_points: bool = True

        lidar_folder = "data_3d_raw"
        img_folder = "data_2d_raw"
        pose_folder = "data_poses"

        self.lidar_root_dir = os.path.realpath(data_dir / lidar_folder / seq_str)
        self.img_root_dir = os.path.realpath(data_dir / img_folder / seq_str)
        self.pose_root_dir = os.path.realpath(data_dir / pose_folder / seq_str)
        self.calib_path = os.path.realpath(data_dir / "calibration")

        self._load_calib() # load all calib first

        self.T_l_co = self.Tr_lidar_cam0 # tran from original camera to lidar
        self.T_co_l = np.linalg.inv(self.T_l_co) 

        self.velodyne_dir = os.path.join(self.lidar_root_dir, "velodyne_points/data/")
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))

        # img_type = "data_rgb" # 512, 1392
        img_type = "data_rect" # 376, 1408

        self.T_c_l_mats = {}
        self.K_mats = {}

        # left cam
        self.cam0_dir = os.path.join(self.img_root_dir, f"image_00/{img_type}/")
        self.img0_files = sorted(glob.glob(self.cam0_dir + "*.png"))

        self.K_mat_left = self.calib_intrinsic['P_rect_00'] # for rectified cam 0

        self.T_cr_co = np.eye(4)
        self.T_cr_co[:3,:3] = self.calib_intrinsic['R_rect_00'] # from camera to rectified camera frame # 4,4

        self.T_c_l = np.matmul(self.T_cr_co, self.T_co_l) # tran from lidar to rectified camera frame
        self.T_l_c = np.linalg.inv(self.T_c_l)

        self.T_c_l_mats["cam_left_rect"] = self.T_c_l
        self.K_mats["cam_left_rect"] = self.K_mat_left

        # right cam
        self.cam1_dir = os.path.join(self.img_root_dir, f"image_00/{img_type}/")
        self.img1_files = sorted(glob.glob(self.cam1_dir + "*.png"))

        # GNSSINS stuff
        self.oxts_dir = os.path.join(self.pose_root_dir, "oxts/data/")
        self.oxts_files = sorted(glob.glob(self.oxts_dir + "*.txt"))
        self.oxts, self.imu_poses = self.load_oxts_packets_and_poses(self.oxts_files) # gt poses in IMU frame

        # GT poses in LiDAR frame
        self.gt_poses = self.Tr_lidar_imu @ self.imu_poses @ self.Tr_imu_lidar 

    def __len__(self):
        return len(self.scan_files)
    
    def __getitem__(self, idx):

        points, point_ts = self.read_point_cloud(self.scan_files[idx])

        if not self.load_img:
            frame_data = {"points": points, "point_ts": point_ts}
            return frame_data
            
        # now we use only the left cam
        img = self.read_img(self.img0_files[idx])
        cam_name = "cam_left_rect"
        img_dict = {cam_name: img}
        
        points_rgb = np.ones_like(points)

        # project to the image plane to get the corresponding color
        points_rgb = self.project_points_to_cam(points, points_rgb, img, self.T_c_l_mats[cam_name], self.K_mats[cam_name])

        if self.use_only_colorized_points:
            with_rgb_mask = (points_rgb[:, 3] == 0)
            points = points[with_rgb_mask]
            points_rgb = points_rgb[with_rgb_mask]
            point_ts = point_ts[with_rgb_mask]

        # we skip the intensity here for now (and also the color mask)
        points = np.hstack((points[:,:3], points_rgb[:,:3]))

        frame_data = {"points": points, "point_ts": point_ts, "img": img_dict}

        return frame_data

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64) 

        return points, self.get_timestamps(points)
    
    def read_img(self, img_file: str):
        img = self.cv2.imread(img_file)
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)

        return img
    
    # reference: kitti 360 dev-kit (Liao et al.)
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

    def get_linear_velocity(self, idx):
        packet = self.oxts[idx].packet
        return np.array([packet.vf, packet.vl, packet.vu])

    def get_angular_velocity(self, idx):
        packet = self.oxts[idx].packet
        return np.array([packet.wf, packet.wl, packet.wu])

    def get_velocities(self, idx):
        return self.get_linear_velocity(idx), self.get_angular_velocity(idx)

    # velodyne lidar
    @staticmethod
    def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        timestamps = 0.5 * (yaw / np.pi + 1.0)
        return timestamps

    ### some codes are adapted from kitti360Scripts/devkits
    # Utils to load transformation to camera pose to system pose

    def _load_calib_rigid(self, filename):
        # check file
        if not os.path.isfile(filename):
            raise RuntimeError('%s does not exist!' % filename)

        lastrow = np.array([0,0,0,1]).reshape(1,4)
        return np.concatenate((np.loadtxt(filename).reshape(3,4), lastrow))
    
    def _load_calib_cam_to_imu(self, filename):
        # check file
        if not os.path.isfile(filename):
            raise RuntimeError('%s does not exist!' % filename)
        
        # open file
        fid = open(filename,'r')
        
        # read variables
        Tr = {}
        cameras = ['image_00', 'image_01', 'image_02', 'image_03']
        lastrow = np.array([0,0,0,1]).reshape(1,4)
        for camera in cameras:
            Tr[camera] = np.concatenate((self._read_variable(fid, camera, 3, 4), lastrow))
        
        # close file
        fid.close()
        return Tr

    def _load_perspective_intrinsic(self, filename):
        # check file
        if not os.path.isfile(filename):
            raise RuntimeError('%s does not exist!' % filename)
        
        # open file
        fid = open(filename,'r')

        # read variables
        Tr = {}
        intrinsics = ['P_rect_00', 'R_rect_00', 'P_rect_01', 'R_rect_01']
        lastrow = np.array([0,0,0,1]).reshape(1,4)
        for intrinsic in intrinsics:
            if intrinsic.startswith('P_rect'):
                Tr[intrinsic] = np.concatenate((self._read_variable(fid, intrinsic, 3, 4), lastrow))
            else:
                Tr[intrinsic] = self._read_variable(fid, intrinsic, 3, 3)

        # close file
        fid.close()

        return Tr
    
    def _read_variable(self,fid,name,M,N):
        # rewind
        fid.seek(0,0)
        
        # search for variable identifier
        line = 1
        success = 0
        while line:
            line = fid.readline()
            if line.startswith(name):
                success = 1
                break

        # return if variable identifier not found
        if success==0:
            return None
        
        # fill matrix
        line = line.replace('%s:' % name, '')
        line = line.split()
        assert(len(line) == M*N)
        line = [float(x) for x in line]
        mat = np.array(line).reshape(M, N)

        return mat

    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""

        file_pers_intrinsic = os.path.join(self.calib_path, 'perspective.txt')
        self.calib_intrinsic = self._load_perspective_intrinsic(file_pers_intrinsic)

        file_cam_extrinsic = os.path.join(self.calib_path, 'calib_cam_to_pose.txt')
        self.calib_extrinsic = self._load_calib_cam_to_imu(file_cam_extrinsic) # Tr_imu_cam0, Tr_imu_cam1, Tr_imu_cam2, Tr_imu_cam3

        file_cam0_to_lidar = os.path.join(self.calib_path, 'calib_cam_to_velo.txt')
        self.Tr_lidar_cam0 = self._load_calib_rigid(file_cam0_to_lidar)

        self.Tr_lidar_imu = self.Tr_lidar_cam0 @ np.linalg.inv(self.calib_extrinsic["image_00"])
        
        self.Tr_imu_lidar = np.linalg.inv(self.Tr_lidar_imu)


    ### FROM THIS POINT EVERYTHING IS COPY PASTED FROM PYKITTI
    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transforation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
    
    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet."""

        def rotx(t):
            """Rotation about the x-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        def roty(t):
            """Rotation about the y-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def rotz(t):
            """Rotation about the z-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        er = 6378137.0  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + packet.lat) * np.pi / 360.0))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t
    
    def postprocess_oxts_poses(poses_in):
        """ convert coordinate system from
        #   x=forward, y=right, z=down 
        # to
        #   x=forward, y=left, z=up
        """

        R = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        
        poses  = []
        
        for i in range(len(poses_in)):
            # if there is no data => no pose
            if not len(poses_in[i]):
                poses.append([])
                continue
            P = poses_in[i]
            poses.append(np.matmul(R, P.T).T )
        
        return poses

    def load_oxts_packets_and_poses(self, oxts_files):
        """Generator to read OXTS ground truth data.

        Poses are given in an East-North-Up coordinate system
        whose origin is the first GPS position.

        GPS/IMU 3D localization unit
        ============================

        The GPS/IMU information is given in a single small text file which is
        written for each synchronized frame. Each text file contains 30 values
        which are:

          - lat:     latitude of the oxts-unit (deg)
          - lon:     longitude of the oxts-unit (deg)
          - alt:     altitude of the oxts-unit (m)
          - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
          - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
          - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
          - vn:      velocity towards north (m/s)
          - ve:      velocity towards east (m/s)
          - vf:      forward velocity, i.e. parallel to earth-surface (m/s)
          - vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
          - vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
          - ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
          - ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
          - az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
          - af:      forward acceleration (m/s^2)
          - al:      leftward acceleration (m/s^2)
          - au:      upward acceleration (m/s^2)
          - wx:      angular rate around x (rad/s)
          - wy:      angular rate around y (rad/s)
          - wz:      angular rate around z (rad/s)
          - wf:      angular rate around forward axis (rad/s)
          - wl:      angular rate around leftward axis (rad/s)
          - wu:      angular rate around upward axis (rad/s)
          - posacc:  velocity accuracy (north/east in m)
          - velacc:  velocity accuracy (north/east in m/s)
          - navstat: navigation status
          - numsats: number of satellites tracked by primary GPS receiver
          - posmode: position mode of primary GPS receiver
          - velmode: velocity mode of primary GPS receiver
          - orimode: orientation mode of primary GPS receiver

        To read the text file and interpret them properly an example is given in
        the matlab folder: First, use oxts = loadOxtsliteData('2011_xx_xx_drive_xxxx')
        to read in the GPS/IMU data. Next, use pose = convertOxtsToPose(oxts) to
        transform the oxts data into local euclidean poses, specified by 4x4 rigid
        transformation matrices. For more details see the comments in those files.

        """
        # Per dataformat.txt
        OxtsPacket = namedtuple(
            "OxtsPacket",
            "lat, lon, alt, "
            + "roll, pitch, yaw, "
            + "vn, ve, vf, vl, vu, "
            + "ax, ay, az, af, al, au, "
            + "wx, wy, wz, wf, wl, wu, "
            + "pos_accuracy, vel_accuracy, "
            + "navstat, numsats, "
            + "posmode, velmode, orimode",
        )

        # Bundle into an easy-to-access structure
        OxtsData = namedtuple("OxtsData", "packet, T_w_imu")
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []
        T_w_imu_poses = []

        for filename in oxts_files:
            with open(filename, "r") as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    packet = OxtsPacket(*line)

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.0)

                    R, t = self.pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = self.transform_from_rot_trans(R, t)
                    T_w_imu_poses.append(T_w_imu)

                    # print(T_w_imu)

                    oxts.append(OxtsData(packet, T_w_imu))

        # imu frame definition is different from original KITTI_raw
        # convert coordinate system from
        #   x=forward, y=right, z=down 
        # to
        #   x=forward, y=left, z=up
        tran_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        T_w_imu_poses = T_w_imu_poses @ tran_mat

        # # Start from identity
        first_pose = T_w_imu_poses[0]
        T_w_imu_poses = np.linalg.inv(first_pose) @ T_w_imu_poses
        
        return oxts, T_w_imu_poses
