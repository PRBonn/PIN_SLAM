#!/usr/bin/env python3
# @file      visualizer.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

# Adapted from Nacho's awesome lidar visualizer (https://github.com/PRBonn/lidar-visualizer)
# This is deprecated, now we use the GUI in gui/slam_gui.py

import os
from functools import partial
from typing import Callable, List

import numpy as np
import open3d as o3d

from utils.config import Config

YELLOW = np.array([1, 0.706, 0])
RED = np.array([255, 0, 0]) / 255.0
PURPLE = np.array([238, 130, 238]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
GOLDEN = np.array([1.0, 0.843, 0.0])
GREEN = np.array([0, 128, 0]) / 255.0
BLUE = np.array([0, 0, 128]) / 255.0
LIGHTBLUE = np.array([0.00, 0.65, 0.93])


class MapVisualizer:
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(self, config: Config = None):

        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = True
        self.reset_bounding_box = True
        self.config = config

        self.cur_frame_id: int = 0

        # Create data
        self.scan = o3d.geometry.PointCloud()
        self.frame_axis = o3d.geometry.TriangleMesh()
        self.sensor_cad = o3d.geometry.TriangleMesh()
        self.mesh = o3d.geometry.TriangleMesh()
        self.sdf = o3d.geometry.PointCloud()
        self.neural_points = o3d.geometry.PointCloud()
        self.data_pool = o3d.geometry.PointCloud()

        self.odom_traj_pcd = o3d.geometry.PointCloud()
        self.pgo_traj_pcd = o3d.geometry.PointCloud()
        self.gt_traj_pcd = o3d.geometry.PointCloud()

        self.odom_traj = o3d.geometry.LineSet()
        self.pgo_traj = o3d.geometry.LineSet()
        self.gt_traj = o3d.geometry.LineSet()

        self.pgo_edges = o3d.geometry.LineSet()

        self.log_path = "./"
        self.sdf_slice_height = 0.0
        self.mc_res_m = 0.1
        self.mesh_min_nn = 10
        self.keep_local_mesh = True

        self.frame_axis_len = 0.5

        if config is not None:
            self.log_path = os.path.join(config.run_path, "log")
            self.frame_axis_len = config.vis_frame_axis_len
            self.sdf_slice_height = config.sdf_slice_height
            if self.config.sensor_cad_path is not None:
                self.sensor_cad = o3d.io.read_triangle_mesh(config.sensor_cad_path)
                self.sensor_cad.compute_vertex_normals()
            self.mc_res_m = config.mc_res_m
            self.mesh_min_nn = config.mesh_min_nn
            self.keep_local_mesh = config.keep_local_mesh

        self.before_pgo = True
        self.last_pose = np.eye(4)

        # Initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

        # Visualization options
        self.render_mesh: bool = True
        self.render_pointcloud: bool = True
        self.render_frame_axis: bool = True
        self.render_trajectory: bool = True
        self.render_gt_trajectory: bool = False
        self.render_odom_trajectory: bool = (
            True  # when pgo is on, visualize the odom or not
        )
        self.render_neural_points: bool = False
        self.render_data_pool: bool = False
        self.render_sdf: bool = False
        self.render_pgo: bool = self.render_trajectory

        self.sdf_slice_height_step: float = 0.1

        self.vis_pc_color: bool = True
        self.pc_uniform_color: bool = False

        self.vis_only_cur_samples: bool = False

        self.mc_res_change_interval_m: float = 0.2 * self.mc_res_m

        self.vis_global: bool = False

        self.ego_view: bool = False
        self.ego_change_flag: bool = False

        self.debug_mode: int = 0

        self.neural_points_vis_mode: int = 0

        self.global_viewpoint: bool = False
        self.view_control = self.vis.get_view_control()
        self.camera_params = self.view_control.convert_to_pinhole_camera_parameters()

    def update_view(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def pause_view(self):
        while self.block_vis:
            self.update_view()
            if self.play_crun:
                break

    def update(
        self,
        scan=None,
        pose=None,
        sdf=None,
        mesh=None,
        neural_points=None,
        data_pool=None,
        pause_now=False,
    ):
        self._update_geometries(scan, pose, sdf, mesh, neural_points, data_pool)
        self.update_view()
        self.pause_view()
        if pause_now:
            self.stop()

    def update_traj(
        self,
        cur_pose=None,
        odom_poses=None,
        gt_poses=None,
        pgo_poses=None,
        pgo_edges=None,
    ):
        self._update_traj(cur_pose, odom_poses, gt_poses, pgo_poses, pgo_edges)
        self.update_view()
        self.pause_view()

    def update_pointcloud(self, scan):
        self._update_pointcloud(scan)
        self.update_view()
        self.pause_view()

    def update_mesh(self, mesh):
        self._update_mesh(mesh)
        self.update_view()
        self.pause_view()

    def destroy_window(self):
        self.vis.destroy_window()

    def stop(self):
        self.play_crun = not self.play_crun
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break

    def _initialize_visualizer(self):
        w_name = "📍 PIN-SLAM Visualizer"
        self.vis.create_window(
            window_name=w_name, width=2560, height=1600
        )  # 1920, 1080
        self.vis.add_geometry(self.scan)
        self.vis.add_geometry(self.sdf)
        self.vis.add_geometry(self.frame_axis)
        self.vis.add_geometry(self.mesh)
        self.vis.add_geometry(self.neural_points)
        self.vis.add_geometry(self.data_pool)
        self.vis.add_geometry(self.odom_traj_pcd)
        self.vis.add_geometry(self.gt_traj_pcd)
        self.vis.add_geometry(self.pgo_traj_pcd)
        self.vis.add_geometry(self.pgo_edges)

        self.vis.get_render_option().line_width = 500
        self.vis.get_render_option().light_on = True
        self.vis.get_render_option().mesh_shade_option = (
            o3d.visualization.MeshShadeOption.Color
        )

        if self.config is not None:
            self.vis.get_render_option().point_size = self.config.vis_point_size
            if self.config.mesh_vis_normal:
                self.vis.get_render_option().mesh_color_option = (
                    o3d.visualization.MeshColorOption.Normal
                )

        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/resume\n"
            "\t[ESC/Q] to exit\n"
            "\t    [G] to toggle on/off the global/local map visualization\n"
            "\t    [E] to toggle on/off the ego/map viewpoint\n"
            "\t    [F] to toggle on/off the current point cloud\n"
            "\t    [M] to toggle on/off the mesh\n"
            "\t    [T] to toggle on/off PIN SLAM trajectory\n"
            "\t    [Y] to toggle on/off the reference trajectory\n"
            "\t    [U] to toggle on/off PIN odometry trajectory\n"
            "\t    [A] to toggle on/off the current frame axis\n"
            "\t    [P] to toggle on/off the neural points map\n"
            "\t    [D] to toggle on/off the data pool\n"
            "\t    [I] to toggle on/off the sdf map slice\n"
            "\t    [R] to center the view point\n"
            "\t    [Z] to save the currently visualized entities in the log folder\n"
        )

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["R"], self._center_viewpoint)
        self._register_key_callback(["E"], self._toggle_ego)
        self._register_key_callback(["F"], self._toggle_pointcloud)
        self._register_key_callback(["A"], self._toggle_frame_axis)
        self._register_key_callback(["I"], self._toggle_sdf)
        self._register_key_callback(["M"], self._toggle_mesh)
        self._register_key_callback(["P"], self._toggle_neural_points)
        self._register_key_callback(["D"], self._toggle_data_pool)
        self._register_key_callback(["T"], self._toggle_trajectory)
        self._register_key_callback(["Y"], self._toggle_gt_trajectory)
        self._register_key_callback(["U"], self._toggle_odom_trajectory)
        self._register_key_callback(["G"], self._toggle_global)
        self._register_key_callback(["Z"], self._save_cur_vis)
        self._register_key_callback([";"], self._toggle_loop_debug)
        self._register_key_callback(
            ["/"], self._toggle_neural_point_vis_mode
        )  # vis neural point color using feature, ts or certainty
        self._register_key_callback(["'"], self._toggle_vis_cur_sample)
        self._register_key_callback(["]"], self._toggle_increase_mesh_res)
        self._register_key_callback(["["], self._toggle_decrease_mesh_res)
        self._register_key_callback(["."], self._toggle_increase_mesh_nn)  # '>'
        self._register_key_callback([","], self._toggle_decrease_mesh_nn)  # '<'
        self._register_key_callback(["5"], self._toggle_point_color)
        self._register_key_callback(["6"], self._toggle_uniform_color)
        self._register_key_callback(["7"], self._switch_background)
        # self.vis.register_key_callback(262, partial(self._toggle_)) # right arrow # for future
        # self.vis.register_key_callback(263, partial(self._toggle_)) # left arrow
        self.vis.register_key_callback(
            265, partial(self._toggle_increase_slice_height)
        )  # up arrow
        self.vis.register_key_callback(
            264, partial(self._toggle_decrease_slice_height)
        )  # down arrow
        # leave C and V as the view copying, pasting function
        # use alt + prt sc for the window screenshot

    def _switch_background(self, vis):
        cur_background_color = vis.get_render_option().background_color
        vis.get_render_option().background_color = np.ones(3) - cur_background_color

    def _center_viewpoint(self, vis):
        self.reset_bounding_box = not self.reset_bounding_box
        vis.reset_view_point(True)

    def _toggle_point_color(
        self, vis
    ):  # actually used to show the source point cloud weight for registration
        self.vis_pc_color = not self.vis_pc_color

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _save_cur_vis(self, vis):
        if self.data_pool.has_points():
            data_pool_pc_name = str(self.cur_frame_id) + "_training_sdf_pool.ply"
            data_pool_pc_path = os.path.join(self.log_path, data_pool_pc_name)
            o3d.io.write_point_cloud(data_pool_pc_path, self.data_pool)
            print("Output current training data pool to: ", data_pool_pc_path)
        if self.scan.has_points():
            if self.vis_pc_color:
                scan_pc_name = str(self.cur_frame_id) + "_scan_map.ply"
            else:
                scan_pc_name = str(self.cur_frame_id) + "_scan_reg.ply"
            scan_pc_path = os.path.join(self.log_path, scan_pc_name)
            o3d.io.write_point_cloud(scan_pc_path, self.scan)
            print("Output current scan to: ", scan_pc_path)
        if self.neural_points.has_points():
            neural_point_name = str(self.cur_frame_id) + "_neural_points.ply"
            neural_point_path = os.path.join(self.log_path, neural_point_name)
            o3d.io.write_point_cloud(neural_point_path, self.neural_points)
            print("Output current neural points to: ", neural_point_path)
        if self.sdf.has_points():
            sdf_slice_name = str(self.cur_frame_id) + "_sdf_slice.ply"
            sdf_slice_path = os.path.join(self.log_path, sdf_slice_name)
            o3d.io.write_point_cloud(sdf_slice_path, self.sdf)
            print("Output current SDF slice to: ", sdf_slice_path)
        if self.mesh.has_triangles():
            mesh_name = str(self.cur_frame_id) + "_mesh_vis.ply"
            mesh_path = os.path.join(self.log_path, mesh_name)
            o3d.io.write_triangle_mesh(mesh_path, self.mesh)
            print("Output current mesh to: ", mesh_path)
        if self.frame_axis.has_triangles():
            ego_name = str(self.cur_frame_id) + "_sensor_vis.ply"
            ego_path = os.path.join(self.log_path, ego_name)
            o3d.io.write_triangle_mesh(ego_path, self.frame_axis)
            print("Output current sensor model to: ", ego_path)

    def _next_frame(self, vis):  # FIXME
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _toggle_pointcloud(self, vis):
        self.render_pointcloud = not self.render_pointcloud

    def _toggle_frame_axis(self, vis):
        self.render_frame_axis = not self.render_frame_axis

    def _toggle_trajectory(self, vis):
        self.render_trajectory = not self.render_trajectory

    def _toggle_gt_trajectory(self, vis):
        self.render_gt_trajectory = not self.render_gt_trajectory

    def _toggle_odom_trajectory(self, vis):
        self.render_odom_trajectory = not self.render_odom_trajectory

    def _toggle_pgo(self, vis):
        self.render_pgo = not self.render_pgo

    def _toggle_sdf(self, vis):
        self.render_sdf = not self.render_sdf

    def _toggle_mesh(self, vis):
        self.render_mesh = not self.render_mesh
        print("Show mesh: ", self.render_mesh)

    def _toggle_neural_points(self, vis):
        self.render_neural_points = not self.render_neural_points

    def _toggle_data_pool(self, vis):
        self.render_data_pool = not self.render_data_pool

    def _toggle_global(self, vis):
        self.vis_global = not self.vis_global

    def _toggle_ego(self, vis):
        self.ego_view = not self.ego_view
        self.ego_change_flag = True  # ego->global or global->ego
        self.reset_bounding_box = not self.reset_bounding_box
        vis.reset_view_point(True)

    def _toggle_uniform_color(self, vis):
        self.pc_uniform_color = not self.pc_uniform_color

    def _toggle_loop_debug(self, vis):
        self.debug_mode = (
            self.debug_mode + 1
        ) % 3  # 0,1,2 # switch between different debug mode
        print("Switch to debug mode:", self.debug_mode)

    def _toggle_vis_cur_sample(self, vis):
        self.vis_only_cur_samples = not self.vis_only_cur_samples

    def _toggle_neural_point_vis_mode(self, vis):
        self.neural_points_vis_mode = (
            self.neural_points_vis_mode + 1
        ) % 5  # 0,1,2,3,4 # switch between different vis mode
        print("Switch to neural point visualization mode:", self.neural_points_vis_mode)

    def _toggle_increase_mesh_res(self, vis):
        self.mc_res_m += self.mc_res_change_interval_m
        print("Current marching cubes voxel size [m]:", f"{self.mc_res_m:.2f}")

    def _toggle_decrease_mesh_res(self, vis):
        self.mc_res_m = max(
            self.mc_res_change_interval_m, self.mc_res_m - self.mc_res_change_interval_m
        )
        print("Current marching cubes voxel size [m]:", f"{self.mc_res_m:.2f}")

    def _toggle_increase_slice_height(self, vis):
        self.sdf_slice_height += self.sdf_slice_height_step
        print("Current SDF slice height [m]:", f"{self.sdf_slice_height:.2f}")

    def _toggle_decrease_slice_height(self, vis):
        self.sdf_slice_height -= self.sdf_slice_height_step
        print("Current SDF slice height [m]:", f"{self.sdf_slice_height:.2f}")

    def _toggle_increase_mesh_nn(self, vis):
        self.mesh_min_nn += 1
        print("Current marching cubes mask nn count:", self.mesh_min_nn)

    def _toggle_decrease_mesh_nn(self, vis):
        self.mesh_min_nn = max(5, self.mesh_min_nn - 1)
        print("Current marching cubes mask nn count:", self.mesh_min_nn)

    def _toggle_help(self, vis):
        print(
            f"Instructions. Press:\n"
            "\t[SPACE] to pause/resume\n"
            "\t[ESC/Q] to exit\n"
            "\t    [G] to toggle on/off the global/local map visualization\n"
            "\t    [E] to toggle on/off the ego/map viewpoint\n"
            "\t    [F] to toggle on/off the current point cloud\n"
            "\t    [M] to toggle on/off the mesh\n"
            "\t    [T] to toggle on/off PIN SLAM trajectory\n"
            "\t    [Y] to toggle on/off the reference trajectory\n"
            "\t    [U] to toggle on/off PIN odometry trajectory\n"
            "\t    [A] to toggle on/off the current frame axis\n"
            "\t    [P] to toggle on/off the neural points map\n"
            "\t    [D] to toggle on/off the data pool\n"
            "\t    [I] to toggle on/off the sdf map slice\n"
            "\t    [R] to center the view point\n"
            "\t    [Z] to save the currently visualized entities in the log folder\n"
        )
        self.play_crun = not self.play_crun
        return False

    def _update_mesh(self, mesh):
        if self.render_mesh:
            if mesh is not None:
                self.vis.remove_geometry(self.mesh, self.reset_bounding_box)
                self.mesh = mesh
                self.vis.add_geometry(self.mesh, self.reset_bounding_box)
        else:
            self.vis.remove_geometry(self.mesh, self.reset_bounding_box)

    def _update_pointcloud(self, scan):
        if scan is not None:
            self.vis.remove_geometry(self.scan, self.reset_bounding_box)
            self.scan = scan
            self.vis.add_geometry(self.scan, self.reset_bounding_box)

            if self.reset_bounding_box:
                self.vis.reset_view_point(True)
                self.reset_bounding_box = False

    def _update_geometries(
        self,
        scan=None,
        pose=None,
        sdf=None,
        mesh=None,
        neural_points=None,
        data_pool=None,
    ):

        # Scan (toggled by "F")
        if self.render_pointcloud:
            if scan is not None:
                self.scan.points = o3d.utility.Vector3dVector(scan.points)
                self.scan.colors = o3d.utility.Vector3dVector(scan.colors)
                self.scan.normals = o3d.utility.Vector3dVector(scan.normals)
                if self.pc_uniform_color or (
                    self.vis_pc_color
                    and (self.config.color_channel == 0)
                    and (not self.config.semantic_on)
                    and (not self.config.dynamic_filter_on)
                ):
                    self.scan.paint_uniform_color(GOLDEN)
            else:
                self.scan.points = o3d.utility.Vector3dVector()
        else:
            self.scan.points = o3d.utility.Vector3dVector()
            # self.scan.colors = o3d.utility.Vector3dVector()
        if self.ego_view and pose is not None:
            self.scan.transform(np.linalg.inv(pose))
        self.vis.update_geometry(self.scan)

        # Mesh Map (toggled by "M")
        if self.render_mesh:
            if mesh is not None:
                if not self.keep_local_mesh:
                    self.vis.remove_geometry(
                        self.mesh, self.reset_bounding_box
                    )  # if comment, then we keep the previous reconstructed mesh (for the case we use local map reconstruction)
                self.mesh = mesh
                if self.ego_view and pose is not None:
                    self.mesh.transform(np.linalg.inv(pose))
                self.vis.add_geometry(self.mesh, self.reset_bounding_box)
            else:  # None, meshing for every frame can be time consuming, we just keep the mesh reconstructed from last frame for vis
                if self.ego_view and pose is not None:
                    self.vis.remove_geometry(self.mesh, self.reset_bounding_box)
                    if self.ego_change_flag:  # global -> ego view
                        self.mesh.transform(np.linalg.inv(pose))
                        self.ego_change_flag = False
                    else:
                        self.mesh.transform(np.linalg.inv(pose) @ self.last_pose)
                    self.vis.add_geometry(self.mesh, self.reset_bounding_box)
                elif self.ego_change_flag:  # ego -> global view
                    self.vis.remove_geometry(self.mesh, self.reset_bounding_box)
                    self.mesh.transform(self.last_pose)
                    self.vis.add_geometry(self.mesh, self.reset_bounding_box)
                    self.ego_change_flag = False
        else:
            self.vis.remove_geometry(self.mesh, self.reset_bounding_box)

        # Neural Points Map (toggled by "P")
        if neural_points is not None:
            if self.render_neural_points:
                self.neural_points.points = o3d.utility.Vector3dVector(
                    neural_points.points
                )
                self.neural_points.colors = o3d.utility.Vector3dVector(
                    neural_points.colors
                )
            else:
                self.neural_points.points = o3d.utility.Vector3dVector()
        else:
            self.neural_points.points = o3d.utility.Vector3dVector()
        if self.ego_view and pose is not None:
            self.neural_points.transform(np.linalg.inv(pose))
        self.vis.update_geometry(self.neural_points)

        # Data Pool (toggled by "D")
        if data_pool is not None:
            if self.render_data_pool:
                self.data_pool.points = o3d.utility.Vector3dVector(data_pool.points)
                self.data_pool.colors = o3d.utility.Vector3dVector(data_pool.colors)
            else:
                self.data_pool.points = o3d.utility.Vector3dVector()
        else:
            self.data_pool.points = o3d.utility.Vector3dVector()
        if self.ego_view and pose is not None:
            self.data_pool.transform(np.linalg.inv(pose))
        self.vis.update_geometry(self.data_pool)

        # SDF map (toggled by "I")
        if sdf is not None:
            if self.render_sdf:
                self.sdf.points = o3d.utility.Vector3dVector(sdf.points)
                self.sdf.colors = o3d.utility.Vector3dVector(sdf.colors)
            else:
                self.sdf.points = o3d.utility.Vector3dVector()
        else:
            self.sdf.points = o3d.utility.Vector3dVector()
        if self.ego_view and pose is not None:
            self.sdf.transform(np.linalg.inv(pose))
        self.vis.update_geometry(self.sdf)

        # Coordinate frame axis (toggled by "A")
        if self.render_frame_axis:
            if pose is not None:
                self.vis.remove_geometry(self.frame_axis, self.reset_bounding_box)
                self.frame_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=self.frame_axis_len, origin=np.zeros(3)
                )
                self.frame_axis += self.sensor_cad
                if not self.ego_view:
                    self.frame_axis = self.frame_axis.transform(pose)
                self.vis.add_geometry(self.frame_axis, self.reset_bounding_box)
        else:
            self.vis.remove_geometry(self.frame_axis, self.reset_bounding_box)

        if pose is not None:
            self.last_pose = pose

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False

    # show traj as lineset
    # long list to np conversion time
    def _update_traj(
        self,
        cur_pose=None,
        odom_poses_np=None,
        gt_poses_np=None,
        pgo_poses_np=None,
        loop_edges=None,
    ):

        self.vis.remove_geometry(self.odom_traj, self.reset_bounding_box)
        self.vis.remove_geometry(self.gt_traj, self.reset_bounding_box)
        self.vis.remove_geometry(self.pgo_traj, self.reset_bounding_box)
        self.vis.remove_geometry(self.pgo_edges, self.reset_bounding_box)

        if (self.render_trajectory and odom_poses_np is not None and odom_poses_np.shape[0] > 1):
            if pgo_poses_np is not None and (not self.render_odom_trajectory):
                self.odom_traj = o3d.geometry.LineSet()
            else:
                odom_position_np = odom_poses_np[:, :3, 3]
                self.odom_traj.points = o3d.utility.Vector3dVector(odom_position_np)
                odom_edges = np.array([[i, i + 1] for i in range(odom_poses_np.shape[0] - 1)])
                self.odom_traj.lines = o3d.utility.Vector2iVector(odom_edges)

                if pgo_poses_np is None or self.before_pgo:
                    self.odom_traj.paint_uniform_color(RED)
                else:
                    self.odom_traj.paint_uniform_color(BLUE)

                if self.ego_view and cur_pose is not None:
                    self.odom_traj.transform(np.linalg.inv(cur_pose))        
        else:
            self.odom_traj = o3d.geometry.LineSet()

        if (
            self.render_trajectory
            and pgo_poses_np is not None
            and pgo_poses_np.shape[0] > 1
            and (not self.before_pgo)
        ):
            pgo_position_np = pgo_poses_np[:, :3, 3]

            self.pgo_traj.points = o3d.utility.Vector3dVector(pgo_position_np)
            pgo_traj_edges = np.array([[i, i + 1] for i in range(pgo_poses_np.shape[0] - 1)])
            self.pgo_traj.lines = o3d.utility.Vector2iVector(pgo_traj_edges)
            self.pgo_traj.paint_uniform_color(RED)

            if self.ego_view and cur_pose is not None:
                self.pgo_traj.transform(np.linalg.inv(cur_pose))

            if self.render_pgo and loop_edges is not None and len(loop_edges) > 0:
                edges = np.array(loop_edges)
                self.pgo_edges.points = o3d.utility.Vector3dVector(pgo_position_np)
                self.pgo_edges.lines = o3d.utility.Vector2iVector(edges)
                self.pgo_edges.paint_uniform_color(GREEN)

                if self.ego_view and cur_pose is not None:
                    self.pgo_edges.transform(np.linalg.inv(cur_pose))
            else:
                self.pgo_edges = o3d.geometry.LineSet()
        else:
            self.pgo_traj = o3d.geometry.LineSet()
            self.pgo_edges = o3d.geometry.LineSet()

        if (
            self.render_trajectory
            and self.render_gt_trajectory
            and gt_poses_np is not None
            and gt_poses_np.shape[0] > 1
        ):
            gt_position_np = gt_poses_np[:, :3, 3]
            self.gt_traj.points = o3d.utility.Vector3dVector(gt_position_np)
            gt_edges = np.array([[i, i + 1] for i in range(gt_poses_np.shape[0] - 1)])
            self.gt_traj.lines = o3d.utility.Vector2iVector(gt_edges)
            self.gt_traj.paint_uniform_color(BLACK)
            if odom_poses_np is None:
                self.gt_traj.paint_uniform_color(RED)
            if self.ego_view and cur_pose is not None:
                self.gt_traj.transform(np.linalg.inv(cur_pose))
        else:
            self.gt_traj = o3d.geometry.LineSet()

        self.vis.add_geometry(self.odom_traj, self.reset_bounding_box)
        self.vis.add_geometry(self.gt_traj, self.reset_bounding_box)
        self.vis.add_geometry(self.pgo_traj, self.reset_bounding_box)
        self.vis.add_geometry(self.pgo_edges, self.reset_bounding_box)

    def _toggle_view(self, vis):
        self.global_viewpoint = not self.global_viewpoint
        vis.update_renderer()
        vis.reset_view_point(True)
        current_camera = self.view_control.convert_to_pinhole_camera_parameters()
        if self.camera_params and not self.global_viewpoint:
            self.view_control.convert_from_pinhole_camera_parameters(self.camera_params)
        self.camera_params = current_camera
