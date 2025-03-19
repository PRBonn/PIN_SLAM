#!/usr/bin/env python3
# @file      gui_utils.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]

# This GUI is built on top of the great work of MonoGS (https://github.com/muskie82/MonoGS/blob/main/gui/gui_utils.py) 

import queue

from utils.tools import feature_pca_torch
from model.neural_points import NeuralPoints


class VisPacket:
    def __init__(
        self,
        frame_id = None,
        finish=False,
        current_pointcloud_xyz=None,
        current_pointcloud_rgb=None,
        mesh_verts=None,
        mesh_faces=None,
        mesh_verts_rgb=None,
        odom_poses=None,
        gt_poses=None,
        slam_poses=None,
        travel_dist=None,
        slam_finished=False,
    ):
        self.has_neural_points = False

        self.neural_points_data = None

        self.frame_id = frame_id

        self.add_scan(current_pointcloud_xyz, current_pointcloud_rgb)

        self.add_mesh(mesh_verts, mesh_faces, mesh_verts_rgb)

        self.add_traj(odom_poses, gt_poses, slam_poses)

        self.sdf_slice_xyz = None
        self.sdf_slice_rgb = None

        self.sdf_pool_xyz = None
        self.sdf_pool_rgb = None

        self.travel_dist = travel_dist
        self.slam_finished = slam_finished

        self.finish = finish

    # the sorrounding map is also added here
    def add_neural_points_data(self, neural_points: NeuralPoints, only_local_map: bool = True, 
                               pca_color_on: bool = True):
        
        if neural_points is not None:
            self.has_neural_points = True
            self.neural_points_data = {}
            self.neural_points_data["count"] = neural_points.count()
            self.neural_points_data["local_count"] = neural_points.local_count()
            self.neural_points_data["map_memory_mb"] = neural_points.cur_memory_mb
            self.neural_points_data["resolution"] = neural_points.resolution
            
            if only_local_map:
                self.neural_points_data["position"] = neural_points.local_neural_points
                self.neural_points_data["orientation"] = neural_points.local_point_orientations
                self.neural_points_data["geo_feature"] = neural_points.local_geo_features.detach()
                if neural_points.color_on:
                    self.neural_points_data["color_feature"] = neural_points.local_color_features.detach()
                self.neural_points_data["ts"] = neural_points.local_point_ts_update
                self.neural_points_data["stability"] = neural_points.local_point_certainties

                if pca_color_on:
                    local_geo_feature_3d, _ = feature_pca_torch((self.neural_points_data["geo_feature"])[:-1], principal_components=neural_points.geo_feature_pca, down_rate=17)
                    self.neural_points_data["color_pca_geo"] = local_geo_feature_3d

                    if neural_points.color_on:
                        local_color_feature_3d, _ = feature_pca_torch((self.neural_points_data["color_feature"])[:-1], principal_components=neural_points.color_feature_pca, down_rate=17)
                        self.neural_points_data["color_pca_color"] = local_color_feature_3d

            else:
                self.neural_points_data["position"] = neural_points.neural_points
                self.neural_points_data["orientation"] = neural_points.point_orientations
                self.neural_points_data["geo_feature"] = neural_points.geo_features
                if neural_points.color_on:
                    self.neural_points_data["color_feature"] = neural_points.color_features
                self.neural_points_data["ts"] = neural_points.point_ts_update
                self.neural_points_data["stability"] = neural_points.point_certainties
                if neural_points.local_mask is not None:
                    self.neural_points_data["local_mask"] = neural_points.local_mask[:-1]

                if pca_color_on:
                    geo_feature_3d, _ = feature_pca_torch(neural_points.geo_features[:-1], principal_components=neural_points.geo_feature_pca, down_rate=97)
                    self.neural_points_data["color_pca_geo"] = geo_feature_3d

                    if neural_points.color_on:
                        color_feature_3d, _ = feature_pca_torch(neural_points.color_features[:-1], principal_components=neural_points.color_feature_pca, down_rate=97)
                        self.neural_points_data["color_pca_color"] = color_feature_3d


    def add_scan(self, current_pointcloud_xyz=None, current_pointcloud_rgb=None):
        self.current_pointcloud_xyz = current_pointcloud_xyz
        self.current_pointcloud_rgb = current_pointcloud_rgb

        # TODO: add normal later

    def add_sdf_slice(self, sdf_slice_xyz=None, sdf_slice_rgb=None):
        self.sdf_slice_xyz = sdf_slice_xyz
        self.sdf_slice_rgb = sdf_slice_rgb

    def add_sdf_training_pool(self, sdf_pool_xyz=None, sdf_pool_rgb=None):
        self.sdf_pool_xyz = sdf_pool_xyz
        self.sdf_pool_rgb = sdf_pool_rgb

    def add_mesh(self, mesh_verts=None, mesh_faces=None, mesh_verts_rgb=None):
        self.mesh_verts = mesh_verts
        self.mesh_faces = mesh_faces
        self.mesh_verts_rgb = mesh_verts_rgb

    def add_traj(self, odom_poses=None, gt_poses=None, slam_poses=None, loop_edges=None):
        
        self.odom_poses = odom_poses
        self.gt_poses = gt_poses
        self.slam_poses = slam_poses

        if slam_poses is None:
            self.slam_poses = odom_poses

        self.loop_edges = loop_edges


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.empty():
                break
    return message


class ControlPacket:
    flag_pause = False
    flag_vis = True
    flag_mesh = False
    flag_sdf = False
    flag_global = False
    flag_source = False
    mc_res_m = 0.2
    mesh_min_nn = 10
    mesh_freq_frame = 50
    sdf_freq_frame = 50
    sdf_slice_height = 0.2
    sdf_res_m = 0.2
    cur_frame_id = 0

class ParamsGUI:
    def __init__(
        self,
        q_main2vis=None,
        q_vis2main=None,
        config=None,
        local_map_default_on: bool = True,
        robot_default_on: bool = True,
        mesh_default_on: bool = False,
        sdf_default_on: bool = False,   
        neural_point_map_default_on: bool = False,
        neural_point_color_default_mode: int = 1, # 1: geo feature pca, 2: photo feature pca, 3: time, 4: height
        neural_point_vis_down_rate: int = 1,
    ):
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.config = config

        self.robot_default_on = robot_default_on
        self.neural_point_map_default_on = neural_point_map_default_on
        self.mesh_default_on = mesh_default_on
        self.sdf_default_on = sdf_default_on
        self.local_map_default_on = local_map_default_on
        self.neural_point_color_default_mode = neural_point_color_default_mode
        self.neural_point_vis_down_rate = neural_point_vis_down_rate
        
