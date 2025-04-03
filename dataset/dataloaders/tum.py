# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# Copyright (c) 2024 Yue Pan
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
import importlib
import os
from pathlib import Path

import numpy as np

class TUMDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        sequence_dir = os.path.join(data_dir, sequence)

        self.rgb_frames, self.depth_frames, self.gt_poses = self.loadtum(sequence_dir)

        self.intrinsic = self.o3d.camera.PinholeCameraIntrinsic()
        H, W = 480, 640

        if "freiburg1" in sequence:
            fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
        elif "freiburg2" in sequence:
            fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
        elif "freiburg3" in sequence:
            fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
        else: # default
            fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5

        self.intrinsic.set_intrinsics(height=H,
                                     width=W,
                                     fx=fx,
                                     fy=fy,
                                     cx=cx,
                                     cy=cy)
        
        self.K_mat = np.eye(3)
        self.K_mat[0,0]=fx
        self.K_mat[1,1]=fy
        self.K_mat[0,2]=cx
        self.K_mat[1,2]=cy

        self.K_mats = {"cam": self.K_mat}

        self.T_l_c = np.eye(4)
        self.T_c_l = np.linalg.inv(self.T_l_c)

        self.T_c_l_mats = {"cam": self.T_c_l}
        
        self.down_sample_on = False
        self.rand_down_rate = 0.1

        self.load_img = False

    def __len__(self):
        return len(self.depth_frames)

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        def parse_list(filepath, skiprows=0):
            """ read list data """
            data = np.loadtxt(filepath, delimiter=' ',
                                dtype=np.unicode_, skiprows=skiprows)
            return data

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
        
        def pose_matrix_from_quaternion(pvec):
            """ convert 4x4 pose matrix to (t, q) """
            from scipy.spatial.transform import Rotation
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix() # rotation
            pose[:3, 3] = pvec[:3] # translation
            return pose

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
            poses += [c2w]

        poses = np.array(poses)

        return images, depths, poses

    def __getitem__(self, idx):
        
        im_color = self.o3d.io.read_image(self.rgb_frames[idx])
        im_depth = self.o3d.io.read_image(self.depth_frames[idx]) 
        rgbd_image = self.o3d.geometry.RGBDImage.create_from_tum_format(im_color,
            im_depth, convert_rgb_to_intensity=False)

        pcd = self.o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self.intrinsic
        )
        if self.down_sample_on:
            pcd = pcd.random_down_sample(sampling_ratio=self.rand_down_rate)

        points_xyz = np.array(pcd.points, dtype=np.float64)
        points_rgb = np.array(pcd.colors, dtype=np.float64)
        points_xyzrgb = np.hstack((points_xyz, points_rgb))

        frame_data = {"points": points_xyzrgb}

        if self.load_img:
            rgb_image = np.array(self.rgb_frames[idx])
            rgb_image_dict = {"cam": rgb_image}
            frame_data["img"] = rgb_image_dict

        return frame_data 