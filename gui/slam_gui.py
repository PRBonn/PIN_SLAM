#!/usr/bin/env python3
# @file      slam_gui.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]

# This GUI is built on top of the great work of MonoGS (https://github.com/muskie82/MonoGS/blob/main/gui/slam_gui.py) 

from typing import Dict, List, Tuple
import threading
import time
from datetime import datetime

import cv2
import os
import matplotlib.cm as cm
import numpy as np
import copy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from pickle import load, dump

from gui.gui_utils import ParamsGUI, VisPacket, ControlPacket, get_latest_queue

from utils.tools import find_closest_prime
from utils.config import Config

RED = np.array([255, 0, 0]) / 255.0
PURPLE = np.array([238, 130, 238]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
GOLDEN = np.array([255, 215, 0]) / 255.0
SILVER = np.array([192, 192, 192]) / 255.0
GREEN = np.array([0, 128, 0]) / 255.0
BLUE = np.array([0, 0, 128]) / 255.0
LIGHTBLUE = np.array([0, 166, 237]) / 255.0

ToGLCamera = np.array([
    [1,  0,  0,  0],
    [0,  -1,  0,  0],
    [0,  0,  -1,  0],
    [0,  0,  0,  1]
])
FromGLGamera = np.linalg.inv(ToGLCamera)


os.environ["PYOPENGL_PLATFORM"] = "osmesa"


class SLAM_GUI:
    def __init__(self, params_gui: ParamsGUI = None):
        self.step = 0
        self.process_finished = False
        self.slam_finished = False
        self.device = "cuda"

        self.frustum_dict = {}
        self.keyframe_dict = {}
        self.model_dict = {}

        self.q_main2vis = None
        # self.q_vis2main = None
        self.cur_data_packet = None

        self.config: Config = None

        self.init = False

        self.local_map_default_on = True
        self.mesh_default_on = False
        self.sdf_default_on = False
        self.neural_point_map_default_on = False
        self.robot_default_on = True
        self.neural_point_color_default_mode = 1
        self.neural_point_vis_down_rate = 1

        self.ego_state_changed = False

        self.cur_pose = np.eye(4)

        if params_gui is not None:
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main
            self.config = params_gui.config
            self.robot_default_on = params_gui.robot_default_on
            self.neural_point_map_default_on = params_gui.neural_point_map_default_on
            self.mesh_default_on = params_gui.mesh_default_on
            self.sdf_default_on = params_gui.sdf_default_on
            self.local_map_default_on = params_gui.local_map_default_on
            self.neural_point_color_default_mode = params_gui.neural_point_color_default_mode
            self.neural_point_vis_down_rate = params_gui.neural_point_vis_down_rate
        
        self.frame_axis_len = 0.5
        if self.config is not None:
            self.frame_axis_len = self.config.vis_frame_axis_len

        self.init_widget()

        self.cur_frame_id = -1

        self.recorded_poses = []

        access = 0o755
        self.view_save_base_path = os.path.expanduser("~/.viewpoints")
        os.makedirs(self.view_save_base_path, access, exist_ok=True)

        # screenshot / logging saving path

        save_path = os.path.join(self.config.run_path, "log")
        os.makedirs(save_path, access, exist_ok=True)

        self.save_dir_2d_screenshots = os.path.join(save_path, "2d_screenshots")
        os.makedirs(self.save_dir_2d_screenshots, access, exist_ok=True)

        self.save_dir_3d_screenshots = os.path.join(save_path, "3d_screenshots")
        os.makedirs(self.save_dir_3d_screenshots, access, exist_ok=True)

        threading.Thread(target=self._update_thread).start()

    # has some issue here
    def init_widget(self):
        self.window_w, self.window_h = 1600, 900
        # self.window_w, self.window_h = 2560, 1600

        self.window = gui.Application.instance.create_window(
           "📍 PIN-SLAM Viewer", self.window_w, self.window_h
        ) 
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)
        # self.widget3d.scene.show_skybox(False)

        self.window.add_child(self.widget3d)


        # scan
        self.scan_render = rendering.MaterialRecord()
        self.scan_render.shader = "defaultLit" # "defaultUnlit", "normals", "depth"
        self.scan_render_init_size_unit = 2
        self.scan_render.point_size = self.scan_render_init_size_unit * self.window.scaling
        self.scan_render.base_color = [0.9, 0.9, 0.9, 0.8]

        # neural points
        self.neural_points_render = rendering.MaterialRecord()
        self.neural_points_render.shader = "defaultLit"
        self.neural_points_render_init_size_unit = 2
        self.neural_points_render.point_size = self.neural_points_render_init_size_unit * self.window.scaling
        self.neural_points_render.base_color = [0.9, 0.9, 0.9, 0.8]

        # sdf slice
        self.sdf_render = rendering.MaterialRecord()
        self.sdf_render.shader = "defaultLit"
        self.sdf_render.point_size = 8 * self.window.scaling
        self.sdf_render.base_color = [1.0, 1.0, 1.0, 1.0]

        # sdf sample pool
        self.sdf_pool_render = rendering.MaterialRecord()
        self.sdf_pool_render.shader = "defaultLit"
        self.sdf_pool_render.point_size = 1 * self.window.scaling
        self.sdf_pool_render.base_color = [1.0, 1.0, 1.0, 1.0]

        # mesh 
        self.mesh_render = rendering.MaterialRecord()
        self.mesh_render.shader = "normals" 

        # trajectory
        self.traj_render = rendering.MaterialRecord()
        self.traj_render.shader = "unlitLine"
        self.traj_render.line_width = 4 * self.window.scaling  # note that this is scaled with respect to pixels,

        # range ring
        self.ring_render = rendering.MaterialRecord()
        self.ring_render.shader = "unlitLine"
        self.ring_render.line_width = 2 * self.window.scaling  # note that this is scaled with respect to pixels,

        self.cad_render = rendering.MaterialRecord()
        self.cad_render.shader = "defaultLit"
        self.cad_render.base_color = [0.9, 0.9, 0.9, 1.0]

        # deprecated
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        # other geometry entities
        self.mesh = o3d.geometry.TriangleMesh()
        self.scan = o3d.geometry.PointCloud()
        self.rendered_scan = o3d.geometry.PointCloud()
        self.sdf_pool = o3d.geometry.PointCloud() # sample pool
        self.sdf_slice = o3d.geometry.PointCloud()
        self.neural_points = o3d.geometry.PointCloud()
        self.invalid_neural_points = o3d.geometry.PointCloud()
        self.sensor_cad = o3d.geometry.TriangleMesh()
        self.sensor_cad_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.frame_axis_len, origin=np.zeros(3)
        )

        if self.config.sensor_cad_path is not None:
            sensor_cad_origin = o3d.io.read_triangle_mesh(self.config.sensor_cad_path)
            sensor_cad_origin.compute_vertex_normals()
            self.sensor_cad_origin += sensor_cad_origin

        self.odom_traj = o3d.geometry.LineSet()
        self.slam_traj = o3d.geometry.LineSet()
        self.gt_traj = o3d.geometry.LineSet()
        self.loop_edges = o3d.geometry.LineSet()

        # range circles
        self.range_circle = o3d.geometry.LineSet()
        circle_points_1 = generate_circle(radius=self.config.max_range/2, num_points=100)
        lines1 = [[i, (i + 1) % len(circle_points_1)] for i in range(len(circle_points_1))]
        range_circle1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points_1),
            lines=o3d.utility.Vector2iVector(lines1),
        )
        circle_points_2 = generate_circle(radius=self.config.max_range, num_points=100)
        lines2 = [[i, (i + 1) % len(circle_points_2)] for i in range(len(circle_points_2))]
        range_circle2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points_2),
            lines=o3d.utility.Vector2iVector(lines2),
        )
        self.range_circle_origin = range_circle1 + range_circle2
        self.range_circle_origin.paint_uniform_color(LIGHTBLUE)

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        
        em = self.window.theme.font_size
        margin = 0.5 * em
        
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        # tabs.add_tab("Setting", tab_info) # FIXME

        slider_line = gui.Horiz(1.0 * em, gui.Margins(margin))
        
        # these are not button, but rather switch
        self.slider_slam = gui.ToggleSwitch("Pause / Resume SLAM")
        self.slider_slam.is_on = True # default on
        self.slider_slam.set_on_clicked(self._on_slam_slider)
        slider_line.add_child(self.slider_slam)

        self.slider_vis = gui.ToggleSwitch("Pause / Resume Visualization")
        self.slider_vis.is_on = True # default on
        self.slider_vis.set_on_clicked(self._on_vis_slider)
        slider_line.add_child(self.slider_vis)

        self.panel.add_child(slider_line)

        self.panel.add_child(gui.Label("View Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.local_map_chbox = gui.Checkbox("Local Map")
        self.local_map_chbox.checked = self.local_map_default_on
        viewpoint_tile.add_child(self.local_map_chbox)
        
        self.followcam_chbox = gui.Checkbox("Follow")
        self.followcam_chbox.checked = False
        viewpoint_tile.add_child(self.followcam_chbox)

        self.ego_chbox = gui.Checkbox("Ego")
        self.ego_chbox.checked = False
        self.ego_chbox.set_on_checked(self._on_ego_mode)
        viewpoint_tile.add_child(self.ego_chbox)

        self.fly_chbox = gui.Checkbox("Fly")
        # NOTE: in fly mode, you can control like a game using WASD,Q,Z,E,R, up, right, left, down
        self.fly_chbox.checked = False
        self.fly_chbox.set_on_checked(self._set_mouse_mode)
        viewpoint_tile.add_child(self.fly_chbox)

        self.panel.add_child(viewpoint_tile)

        viewpoint_tile_2 = gui.Horiz(0.5 * em, gui.Margins(margin))

        ##Combo panels for preset views 
        combo_tile3 = gui.Vert(0.5 * em, gui.Margins(margin))
        self.combo_preset_cams = gui.Combobox()
        for i in range(30):
            self.combo_preset_cams.add_item(str(i))

        # self.combo_preset_cams.set_on_selection_changed(self._on_combo_preset_cams) 
        # combo_tile3.add_child(gui.Label("Preset"))
        combo_tile3.add_child(self.combo_preset_cams)
        viewpoint_tile_2.add_child(combo_tile3)

        self.save_view_btn = gui.Button("Save")
        self.save_view_btn.set_on_clicked(
            self._on_save_view_btn
        )  # set the callback function

        self.load_view_btn = gui.Button("Load")
        self.load_view_btn.set_on_clicked(
            self._on_load_view_btn
        )  # set the callback function

        self.reset_view_btn = gui.Button("Reset Viewpoint")
        self.reset_view_btn.set_on_clicked(
            self._on_reset_view_btn
        )  # set the callback function

        viewpoint_tile_2.add_child(self.save_view_btn)
        viewpoint_tile_2.add_child(self.load_view_btn)
        viewpoint_tile_2.add_child(self.reset_view_btn)
        
        self.panel.add_child(viewpoint_tile_2)

        self.panel.add_child(gui.Label("3D Objects"))

        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.scan_chbox = gui.Checkbox("Scan")
        self.scan_chbox.checked = True
        self.scan_chbox.set_on_checked(self._on_scan_chbox)
        chbox_tile_3dobj.add_child(self.scan_chbox)
        self.scan_name = "cur_scan"

        self.neural_point_chbox = gui.Checkbox("Neural Point Map")
        self.neural_point_chbox.checked = self.neural_point_map_default_on
        self.neural_point_chbox.set_on_checked(self._on_neural_point_chbox)
        chbox_tile_3dobj.add_child(self.neural_point_chbox)
        self.neural_point_name = "neural_points"

        self.mesh_chbox = gui.Checkbox("Mesh")
        self.mesh_chbox.checked = self.mesh_default_on
        self.mesh_chbox.set_on_checked(self._on_mesh_chbox)
        chbox_tile_3dobj.add_child(self.mesh_chbox)
        self.mesh_name = "pin_mesh"

        self.sdf_chbox = gui.Checkbox("SDF")
        self.sdf_chbox.checked = self.sdf_default_on
        self.sdf_chbox.set_on_checked(self._on_sdf_chbox)
        chbox_tile_3dobj.add_child(self.sdf_chbox)
        self.sdf_name = "cur_sdf_slice"  

        self.cad_chbox = gui.Checkbox("Robot")
        self.cad_chbox.checked = self.robot_default_on
        self.cad_chbox.set_on_checked(self._on_cad_chbox)
        chbox_tile_3dobj.add_child(self.cad_chbox)
        self.cad_name = "sensor_cad"
        
        chbox_tile_3dobj_2 = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.gt_traj_chbox = gui.Checkbox("GT Traj.")
        self.gt_traj_chbox.checked = False
        self.gt_traj_chbox.set_on_checked(self._on_gt_traj_chbox)
        chbox_tile_3dobj_2.add_child(self.gt_traj_chbox)
        self.gt_traj_name = "gt_trajectory"

        self.slam_traj_chbox = gui.Checkbox("SLAM Traj.")
        self.slam_traj_chbox.checked = False
        self.slam_traj_chbox.set_on_checked(self._on_slam_traj_chbox)
        chbox_tile_3dobj_2.add_child(self.slam_traj_chbox)
        self.slam_traj_name = "slam_trajectory"

        self.odom_traj_chbox = gui.Checkbox("Odom. Traj.")
        self.odom_traj_chbox.checked = False
        self.odom_traj_chbox.set_on_checked(self._on_odom_traj_chbox)
        chbox_tile_3dobj_2.add_child(self.odom_traj_chbox)
        self.odom_traj_name = "odom_trajectory"

        self.loop_edges_chbox = gui.Checkbox("Loops")
        self.loop_edges_chbox.checked = False
        chbox_tile_3dobj_2.add_child(self.loop_edges_chbox)
        self.loop_edges_name = "loop_edges"

        chbox_tile_3dobj_3 = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.sdf_pool_chbox = gui.Checkbox("SDF Training Samples")
        self.sdf_pool_chbox.checked = False
        self.sdf_pool_chbox.set_on_checked(self._on_sdf_pool_chbox)
        chbox_tile_3dobj_3.add_child(self.sdf_pool_chbox)
        self.sdf_pool_name = "sdf_sample_pool" 

        self.range_circle_chbox = gui.Checkbox("Range Rings")
        self.range_circle_chbox.checked = False
        self.range_circle_chbox.set_on_checked(self._on_range_circle_chbox)
        chbox_tile_3dobj_3.add_child(self.range_circle_chbox)
        self.range_circle_name = "range_circle"

        self.panel.add_child(chbox_tile_3dobj)
        self.panel.add_child(chbox_tile_3dobj_2)
        self.panel.add_child(chbox_tile_3dobj_3)
        
        self.panel.add_child(gui.Label("Scan Color Options"))
        chbox_tile_scan_color = gui.Horiz(0.5 * em, gui.Margins(margin))

        # mode 1
        self.scan_color_chbox = gui.Checkbox("Color")
        self.scan_color_chbox.checked = True
        self.scan_color_chbox.set_on_checked(self._on_scan_color_chbox)
        chbox_tile_scan_color.add_child(self.scan_color_chbox)
        
        # mode 2
        self.scan_regis_color_chbox = gui.Checkbox("Registration Weight")
        self.scan_regis_color_chbox.checked = False
        self.scan_regis_color_chbox.set_on_checked(self._on_scan_regis_color_chbox)
        chbox_tile_scan_color.add_child(self.scan_regis_color_chbox)

        # mode 3
        self.scan_height_color_chbox = gui.Checkbox("Height")
        self.scan_height_color_chbox.checked = False
        self.scan_height_color_chbox.set_on_checked(self._on_scan_height_color_chbox)
        chbox_tile_scan_color.add_child(self.scan_height_color_chbox)

        self.panel.add_child(chbox_tile_scan_color)
        
        scan_point_size_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        scan_point_size_slider_label = gui.Label("Scan point size (1-6)   ")
        self.scan_point_size_slider = gui.Slider(gui.Slider.INT)
        self.scan_point_size_slider.set_limits(1, 6)
        self.scan_point_size_slider.int_value = self.scan_render_init_size_unit
        self.scan_point_size_slider.set_on_value_changed(self._on_scan_point_size_changed)
        scan_point_size_slider_tile.add_child(scan_point_size_slider_label)
        scan_point_size_slider_tile.add_child(self.scan_point_size_slider)
        self.panel.add_child(scan_point_size_slider_tile)


        self.panel.add_child(gui.Label("Neural Point Color Options"))
        chbox_tile_neuralpoint = gui.Horiz(0.5 * em, gui.Margins(margin))

        # mode 1
        self.neuralpoint_geofeature_chbox = gui.Checkbox("Geo. Feature")
        self.neuralpoint_geofeature_chbox.checked = (self.neural_point_color_default_mode==1)
        self.neuralpoint_geofeature_chbox.set_on_checked(self._on_neuralpoint_geofeature_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_geofeature_chbox)

        # mode 2
        self.neuralpoint_colorfeature_chbox = gui.Checkbox("Photo. Feature")
        self.neuralpoint_colorfeature_chbox.checked = (self.neural_point_color_default_mode==2)
        self.neuralpoint_colorfeature_chbox.set_on_checked(self._on_neuralpoint_colorfeature_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_colorfeature_chbox)

        # mode 3
        self.neuralpoint_ts_chbox = gui.Checkbox("Time")
        self.neuralpoint_ts_chbox.checked = (self.neural_point_color_default_mode==3)
        self.neuralpoint_ts_chbox.set_on_checked(self._on_neuralpoint_ts_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_ts_chbox)

        # mode 4
        self.neuralpoint_height_chbox = gui.Checkbox("Height")
        self.neuralpoint_height_chbox.checked = (self.neural_point_color_default_mode==4)
        self.neuralpoint_height_chbox.set_on_checked(self._on_neuralpoint_height_chbox)
        chbox_tile_neuralpoint.add_child(self.neuralpoint_height_chbox)

        self.panel.add_child(chbox_tile_neuralpoint)
       
        map_point_size_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        map_point_size_slider_label = gui.Label("Neural point size (1-6)")
        self.map_point_size_slider = gui.Slider(gui.Slider.INT)
        self.map_point_size_slider.set_limits(1, 6)
        self.map_point_size_slider.int_value = self.neural_points_render_init_size_unit
        self.map_point_size_slider.set_on_value_changed(self._on_neural_point_point_size_changed)
        map_point_size_slider_tile.add_child(map_point_size_slider_label)
        map_point_size_slider_tile.add_child(self.map_point_size_slider)
        self.panel.add_child(map_point_size_slider_tile)

        self.panel.add_child(gui.Label("Mesh Color Options"))

        chbox_tile_mesh_color = gui.Horiz(0.5 * em, gui.Margins(margin))
        
        # mode 1
        self.mesh_normal_chbox = gui.Checkbox("Normal")
        self.mesh_normal_chbox.checked = True
        self.mesh_normal_chbox.set_on_checked(self._on_mesh_normal_chbox)
        chbox_tile_mesh_color.add_child(self.mesh_normal_chbox)

        # mode 2
        self.mesh_color_chbox = gui.Checkbox("Color")
        self.mesh_color_chbox.checked = False
        self.mesh_color_chbox.set_on_checked(self._on_mesh_color_chbox)
        chbox_tile_mesh_color.add_child(self.mesh_color_chbox)
        
        # mode 3
        self.mesh_height_chbox = gui.Checkbox("Height")
        self.mesh_height_chbox.checked = False
        self.mesh_height_chbox.set_on_checked(self._on_mesh_height_chbox)
        chbox_tile_mesh_color.add_child(self.mesh_height_chbox)

        self.panel.add_child(chbox_tile_mesh_color)

        mesh_freq_frame_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        mesh_freq_frame_slider_label = gui.Label("Mesh update per X frames (1-100)")
        self.mesh_freq_frame_slider = gui.Slider(gui.Slider.INT)
        self.mesh_freq_frame_slider.set_limits(1, 100)
        self.mesh_freq_frame_slider.int_value = self.config.mesh_freq_frame
        mesh_freq_frame_slider_tile.add_child(mesh_freq_frame_slider_label)
        mesh_freq_frame_slider_tile.add_child(self.mesh_freq_frame_slider)
        self.panel.add_child(mesh_freq_frame_slider_tile)
        
        mesh_mc_res_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        mesh_mc_res_slider_label = gui.Label("Mesh MC resolution (5cm-100cm)")
        self.mesh_mc_res_slider = gui.Slider(gui.Slider.INT)
        self.mesh_mc_res_slider.set_limits(5, 100)
        self.mesh_mc_res_slider.int_value = int(self.config.mc_res_m * 100)
        mesh_mc_res_slider_tile.add_child(mesh_mc_res_slider_label)
        mesh_mc_res_slider_tile.add_child(self.mesh_mc_res_slider)
        self.panel.add_child(mesh_mc_res_slider_tile)

        mesh_min_nn_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        mesh_min_nn_slider_label = gui.Label("Mesh query min neighbors (5-25)  ")
        self.mesh_min_nn_slider = gui.Slider(gui.Slider.INT)
        self.mesh_min_nn_slider.set_limits(5, 25)
        self.mesh_min_nn_slider.int_value = self.config.mesh_min_nn
        mesh_min_nn_slider_tile.add_child(mesh_min_nn_slider_label)
        mesh_min_nn_slider_tile.add_child(self.mesh_min_nn_slider)
        self.panel.add_child(mesh_min_nn_slider_tile)

        self.panel.add_child(gui.Label("SDF Slice Options"))

        sdf_freq_frame_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        sdf_freq_frame_slider_label = gui.Label("SDF slice update per X frames (1-100) ")
        self.sdf_freq_frame_slider = gui.Slider(gui.Slider.INT)
        self.sdf_freq_frame_slider.set_limits(1, 100)
        self.sdf_freq_frame_slider.int_value = self.config.sdfslice_freq_frame
        sdf_freq_frame_slider_tile.add_child(sdf_freq_frame_slider_label)
        sdf_freq_frame_slider_tile.add_child(self.sdf_freq_frame_slider)
        self.panel.add_child(sdf_freq_frame_slider_tile)

        sdf_slice_height_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        sdf_slice_height_slider_label = gui.Label("SDF slice height (m)                                 ")
        self.sdf_slice_height_slider = gui.Slider(gui.Slider.DOUBLE)
        self.sdf_slice_height_slider.set_limits(-2.0, 3.0)
        self.sdf_slice_height_slider.double_value = self.config.sdf_slice_height
        sdf_slice_height_slider_tile.add_child(sdf_slice_height_slider_label)
        sdf_slice_height_slider_tile.add_child(self.sdf_slice_height_slider)
        self.panel.add_child(sdf_slice_height_slider_tile)

        sdf_res_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        sdf_res_slider_label = gui.Label("SDF slice resolution (5cm-30cm)          ")
        self.sdf_res_slider = gui.Slider(gui.Slider.INT)
        self.sdf_res_slider.set_limits(5, 30)
        self.sdf_res_slider.int_value = int(self.config.vis_sdf_res_m * 100)
        sdf_res_slider_tile.add_child(sdf_res_slider_label)
        sdf_res_slider_tile.add_child(self.sdf_res_slider)
        self.panel.add_child(sdf_res_slider_tile)

        chbox_save_tile = gui.Horiz(0.5 * em, gui.Margins(margin))

        # screenshot buttom
        self.screenshot_btn = gui.Button("2D Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        chbox_save_tile.add_child(self.screenshot_btn)

        self.screenshot_3d_btn = gui.Button("3D Screenshot")
        self.screenshot_3d_btn.set_on_clicked(
            self._on_screenshot_3d_btn
        )  # set the callback function
        chbox_save_tile.add_child(self.screenshot_3d_btn)
        
        self.panel.add_child(chbox_save_tile)

        ## Info Tab
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()
        tab_info = gui.Vert(0, tab_margins)

        run_name_info = gui.Label("Mission: {}".format(self.config.run_name))
        tab_info.add_child(run_name_info)

        self.frame_info = gui.Label("Frame: ")
        tab_info.add_child(self.frame_info)

        self.neural_points_info = gui.Label("# Neural points: ")
        tab_info.add_child(self.neural_points_info)

        self.loop_info = gui.Label("# Loop Closures: 0")
        tab_info.add_child(self.loop_info)

        self.dist_info = gui.Label("Travel Distance: 0.00 m")
        tab_info.add_child(self.dist_info)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)

        self.window.add_child(self.panel)


    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        #self.widget3d_width_ratio = 0.6 # 0.7 # FIXME
        self.widget3d_width_ratio = self.config.visualizer_split_width_ratio
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True

        print("[GUI] Received terminate signal")
        # clean up the pipe
        while not self.q_main2vis.empty():
            self.q_main2vis.get()
        while not self.q_vis2main.empty():
            self.q_vis2main.get()
        self.q_vis2main = None
        self.q_main2vis = None
        self.process_finished = True

        return True 
    
    def _on_ego_mode(self, is_checked):
        self.ego_state_changed = True # TODO

    def _on_cad_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.cad_name)
            self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
        else:
            self.widget3d.scene.remove_geometry(self.cad_name)
    
    def _on_neural_point_chbox(self, is_checked):
        self.visualize_neural_points()

    def _on_mesh_chbox(self, is_checked):
        self.visualize_mesh()

    def _on_scan_chbox(self, is_checked):
        self.visualize_scan()
    
    def _on_sdf_pool_chbox(self, is_checked):
        self.visualize_sdf_pool()

    def _on_sdf_chbox(self, is_checked):
        self.visualize_sdf_slice()

    def _on_gt_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.gt_traj_name)
            self.widget3d.scene.add_geometry(self.gt_traj_name, self.gt_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.gt_traj_name)

    def _on_slam_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.slam_traj_name)
            self.widget3d.scene.add_geometry(self.slam_traj_name, self.slam_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.slam_traj_name)

    def _on_odom_traj_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.odom_traj_name)
            self.widget3d.scene.add_geometry(self.odom_traj_name, self.odom_traj, self.traj_render)
        else:
            self.widget3d.scene.remove_geometry(self.odom_traj_name)

    def _on_range_circle_chbox(self, is_checked):
        if is_checked:
            self.widget3d.scene.remove_geometry(self.range_circle_name)
            self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)
        else:
            self.widget3d.scene.remove_geometry(self.range_circle_name)

    def _on_mesh_normal_chbox(self, is_checked):
        if is_checked:
            self.mesh_render.shader = "normals"
            self.mesh_color_chbox.checked = False
            self.mesh_height_chbox.checked = False
        self.visualize_mesh()

    def _on_mesh_color_chbox(self, is_checked):
        if is_checked:
            self.mesh_render.shader = "defaultLit"
            self.mesh_normal_chbox.checked = False
            self.mesh_height_chbox.checked = False
        self.visualize_mesh()

    def _on_mesh_height_chbox(self, is_checked):
        if is_checked:
            self.mesh_render.shader = "defaultLit"
            self.mesh_normal_chbox.checked = False
            self.mesh_color_chbox.checked = False
        self.visualize_mesh()

    def _on_scan_color_chbox(self, is_checked):
        if is_checked:
            self.scan_height_color_chbox.checked = False
            self.scan_regis_color_chbox.checked = False
        self.visualize_scan()
    
    def _on_scan_regis_color_chbox(self, is_checked):
        if is_checked:
            self.scan_height_color_chbox.checked = False
            self.scan_color_chbox.checked = False
        self.visualize_scan()

    def _on_scan_height_color_chbox(self, is_checked):
        if is_checked:
            self.scan_color_chbox.checked = False
            self.scan_regis_color_chbox.checked = False
        self.visualize_scan()

    def _on_neuralpoint_geofeature_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_colorfeature_chbox.checked = False
            self.neuralpoint_height_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False
        self.visualize_neural_points()

    def _on_neuralpoint_colorfeature_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_height_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False
        self.visualize_neural_points()

    def _on_neuralpoint_ts_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_height_chbox.checked = False
            self.neuralpoint_colorfeature_chbox.checked = False
        self.visualize_neural_points()

    def _on_neuralpoint_height_chbox(self, is_checked):
        if is_checked:
            self.neuralpoint_geofeature_chbox.checked = False
            self.neuralpoint_ts_chbox.checked = False
            self.neuralpoint_colorfeature_chbox.checked = False
        self.visualize_neural_points()

    def _on_neural_point_point_size_changed(self, value):
        self.neural_points_render.point_size = value * self.window.scaling
        self.visualize_neural_points()

    def _on_scan_point_size_changed(self, value):
        self.scan_render.point_size = value * self.window.scaling
        self.visualize_scan()

    def _on_slam_slider(self, is_on):
        if self.slider_slam.is_on:
            print("[GUI] SLAM resumed")
        else:
            print("[GUI] SLAM paused")

    def _on_vis_slider(self, is_on):
        if self.slider_vis.is_on:
            print("[GUI] Visualization resumed")
        else:
            print("[GUI] Visualization paused")

    def _on_screenshot_btn(self):
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = os.path.join(self.save_dir_2d_screenshots, f"{dt}-gui.png")
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img)

        print("[GUI] 2D Screenshot save at {}".format(filename))

    def _on_screenshot_3d_btn(self):

        if self.sdf_pool.has_points() and self.sdf_pool_chbox.checked:
            data_pool_pc_name = str(self.cur_frame_id) + "_training_sdf_pool"
            if self.ego_chbox.checked:
                data_pool_pc_name += "_ego"
            data_pool_pc_path = os.path.join(self.save_dir_3d_screenshots, data_pool_pc_name)
            o3d.io.write_point_cloud(data_pool_pc_path, self.sdf_pool)
            print("[GUI] Output current SDF training pool to: ", data_pool_pc_path)
        if self.scan.has_points() and self.scan_chbox.checked:
            scan_pc_name = str(self.cur_frame_id) + "_scan"
            if self.ego_chbox.checked:
                scan_pc_name += "_ego"
            scan_pc_name += ".ply"
            scan_pc_path = os.path.join(self.save_dir_3d_screenshots, scan_pc_name)
            o3d.io.write_point_cloud(scan_pc_path, self.scan)
            print("[GUI] Output current scan to: ", scan_pc_path)
        if self.neural_points.has_points() and self.neural_point_chbox.checked:
            neural_point_name = str(self.cur_frame_id) + "_neural_point_map"
            if self.local_map_chbox.checked:
                neural_point_name += "_local"
            if self.ego_chbox.checked:
                neural_point_name += "_ego"
            neural_point_name += ".ply"
            neural_point_path = os.path.join(self.save_dir_3d_screenshots, neural_point_name)
            o3d.io.write_point_cloud(neural_point_path, self.neural_points)
            print("[GUI] Output current neural point map to: ", neural_point_path)
        if self.sdf_slice.has_points() and self.sdf_chbox.checked:
            sdf_slice_name = str(self.cur_frame_id) + "_sdf_slice"
            if self.ego_chbox.checked:
                sdf_slice_name += "_ego"
            sdf_slice_name += ".ply"
            sdf_slice_path = os.path.join(self.save_dir_3d_screenshots, sdf_slice_name)
            o3d.io.write_point_cloud(sdf_slice_path, self.sdf_slice)
            print("[GUI] Output current SDF slice to: ", sdf_slice_path)
        if self.mesh.has_triangles() and self.mesh_chbox.checked:
            mesh_name = str(self.cur_frame_id) + "_mesh_vis"
            if self.local_map_chbox.checked:
                mesh_name += "_local"
            if self.ego_chbox.checked:
                mesh_name += "_ego"
            mesh_name += ".ply"
            mesh_path = os.path.join(self.save_dir_3d_screenshots, mesh_name)
            o3d.io.write_triangle_mesh(mesh_path, self.mesh)
            print("[GUI] Output current mesh to: ", mesh_path)
        if self.sensor_cad.has_triangles() and self.cad_chbox.checked:
            cad_name = str(self.cur_frame_id) + "_sensor_vis"
            if self.ego_chbox.checked:
                cad_name += "_ego"
            cad_name += ".ply"
            cad_path = os.path.join(self.save_dir_3d_screenshots, cad_name)
            o3d.io.write_triangle_mesh(cad_path, self.sensor_cad)
            print("[GUI] Output current sensor model to: ", cad_path)


    def _on_reset_view_btn(self):
        self.center_bev()
        self.fly_chbox.checked = False
        self.widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
    
    def _on_save_view_btn(self):
        save_view_file_name = 'saved_view_{}.pkl'.format(self.combo_preset_cams.selected_text)
        save_view_file_path = os.path.join(self.view_save_base_path, save_view_file_name)
        if self.save_view(save_view_file_path):
            print("[GUI] Viewpoint {} saved".format(self.combo_preset_cams.selected_text))
    
    def _on_load_view_btn(self):
        load_view_file_name = 'saved_view_{}.pkl'.format(self.combo_preset_cams.selected_text)
        load_view_file_path = os.path.join(self.view_save_base_path, load_view_file_name)
        if self.load_view(load_view_file_path):
            print("[GUI] Viewpoint {} loaded".format(self.combo_preset_cams.selected_text))

    def _set_mouse_mode(self, is_on):
        if is_on:
            self.widget3d.set_view_controls(gui.SceneWidget.Controls.FLY)
        else:
            self.widget3d.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)


    def save_view(self, fname='.saved_view.pkl'):
        try:
            model_matrix = np.asarray(self.widget3d.scene.camera.get_model_matrix())
            extrinsic = model_matrix_to_extrinsic_matrix(model_matrix)
            height, width = int(self.window.size.height), int(self.widget3d_width)
            intrinsic = create_camera_intrinsic_from_size(width, height)
            saved_view = dict(extrinsic=extrinsic, intrinsic=intrinsic, width=width, height=height)
            with open(fname, 'wb') as pickle_file:
                dump(saved_view, pickle_file)
            return True
        except Exception as e:
            print("[GUI]", e)
            return False

    def load_view(self, fname=".saved_view.pkl"):
        try:
            with open(fname, 'rb') as pickle_file:
                saved_view = load(pickle_file)
            self.widget3d.setup_camera(saved_view['intrinsic'], saved_view['extrinsic'], saved_view['width'], saved_view['height'], self.widget3d.scene.bounding_box)
            # Looks like the ground plane gets messed up, no idea how to fix
            return True
        except Exception as e:
            print("[GUI] Can't find file", e)
            return False

    def visualize_mesh(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if data_packet.mesh_verts is not None and data_packet.mesh_faces is not None:
            self.mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(data_packet.mesh_verts),
                o3d.utility.Vector3iVector(data_packet.mesh_faces),
                )
            self.mesh.compute_vertex_normals()

            if data_packet.mesh_verts_rgb is not None:
                self.mesh.vertex_colors = o3d.utility.Vector3dVector(data_packet.mesh_verts_rgb)
            
            if self.mesh_height_chbox.checked:
                z_values = np.array(self.mesh.vertices, dtype=np.float64)[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
                color_map = cm.get_cmap("jet")
                mesh_verts_colors_np = color_map(z_normalized)[:, :3].astype(np.float64)
                self.mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_verts_colors_np)

        if self.ego_chbox.checked:
            self.mesh.transform(np.linalg.inv(self.cur_pose))

        self.widget3d.scene.remove_geometry(self.mesh_name)
        self.widget3d.scene.add_geometry(self.mesh_name, self.mesh, self.mesh_render) 
        self.widget3d.scene.show_geometry(self.mesh_name, self.mesh_chbox.checked)

    def visualize_sdf_pool(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if self.sdf_pool_chbox.checked and data_packet.sdf_pool_xyz is not None and data_packet.sdf_pool_rgb is not None:
            self.sdf_pool.points = o3d.utility.Vector3dVector(data_packet.sdf_pool_xyz)
            self.sdf_pool.colors = o3d.utility.Vector3dVector(data_packet.sdf_pool_rgb)

            if self.ego_chbox.checked:
                self.sdf_pool.transform(np.linalg.inv(self.cur_pose))

            self.widget3d.scene.remove_geometry(self.sdf_pool_name)
            self.widget3d.scene.add_geometry(self.sdf_pool_name, self.sdf_pool, self.sdf_pool_render)

        self.widget3d.scene.show_geometry(self.sdf_pool_name, self.sdf_pool_chbox.checked)


    def visualize_neural_points(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet
        
        if self.neural_point_chbox.checked:

            dict_keys = list(data_packet.neural_points_data.keys())

            neural_point_vis_down_rate = self.neural_point_vis_down_rate

            local_mask = None
            # global map is being loaded here
            if "local_mask" in dict_keys:
                local_mask = data_packet.neural_points_data["local_mask"]
                # check if we need to downsample the global map a bit for fast visualization
            
            point_count = data_packet.neural_points_data["count"]
            if point_count > 300000 and not self.local_map_chbox.checked:
                neural_point_vis_down_rate = find_closest_prime(point_count // 200000)

            if local_mask is not None and self.local_map_chbox.checked:
                neural_point_position = data_packet.neural_points_data["position"][local_mask]
            else:
                neural_point_position = data_packet.neural_points_data["position"]

            neural_point_position_np = neural_point_position[::neural_point_vis_down_rate, :].detach().cpu().numpy()
                    
            neural_point_colors_np = None
            
            if "color_pca_geo" in dict_keys and self.neuralpoint_geofeature_chbox.checked:
                if local_mask is not None and self.local_map_chbox.checked:
                    neural_point_colors = data_packet.neural_points_data["color_pca_geo"][local_mask]
                else:
                    neural_point_colors = data_packet.neural_points_data["color_pca_geo"]
                neural_point_colors_np = neural_point_colors[::neural_point_vis_down_rate, :].detach().cpu().numpy()
            elif "color_pca_color" in dict_keys and self.neuralpoint_colorfeature_chbox.checked:
                if local_mask is not None and self.local_map_chbox.checked:
                    neural_point_colors = data_packet.neural_points_data["color_pca_color"][local_mask]
                else:
                    neural_point_colors = data_packet.neural_points_data["color_pca_color"]
                neural_point_colors_np = neural_point_colors[::neural_point_vis_down_rate, :].detach().cpu().numpy()
            elif "ts" in dict_keys and self.neuralpoint_ts_chbox.checked:
                if local_mask is not None and self.local_map_chbox.checked:
                    ts_np = (data_packet.neural_points_data["ts"][local_mask])
                else:
                    ts_np = (data_packet.neural_points_data["ts"])
                ts_np = ts_np[::neural_point_vis_down_rate].detach().cpu().numpy()
                ts_np = ts_np / ts_np.max()
                color_map = cm.get_cmap("jet")
                neural_point_colors_np = color_map(ts_np)[:, :3].astype(np.float64)
            elif self.neuralpoint_height_chbox.checked:
                z_values = neural_point_position_np[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
                color_map = cm.get_cmap("jet")
                neural_point_colors_np = color_map(z_normalized)[:, :3].astype(np.float64)
                
            self.neural_points.points = o3d.utility.Vector3dVector(neural_point_position_np)

            if neural_point_colors_np is not None:
                self.neural_points.colors = o3d.utility.Vector3dVector(neural_point_colors_np)
            else:
                self.neural_points.paint_uniform_color(LIGHTBLUE)

            if self.ego_chbox.checked:
                self.neural_points.transform(np.linalg.inv(self.cur_pose))

            self.widget3d.scene.remove_geometry(self.neural_point_name)
            self.widget3d.scene.add_geometry(self.neural_point_name, self.neural_points, self.neural_points_render)

        self.widget3d.scene.show_geometry(self.neural_point_name, self.neural_point_chbox.checked)

    def visualize_scan(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if self.scan_chbox.checked and data_packet.current_pointcloud_xyz is not None:
            self.scan.points = o3d.utility.Vector3dVector(data_packet.current_pointcloud_xyz)
            if data_packet.current_pointcloud_rgb is not None:
                self.scan.colors = o3d.utility.Vector3dVector(data_packet.current_pointcloud_rgb)
        
            if not (self.config.color_on or self.config.semantic_on or self.scan_regis_color_chbox.checked):
                self.scan.paint_uniform_color(SILVER)

            if self.scan_height_color_chbox.checked:
                z_values = data_packet.current_pointcloud_xyz[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
                color_map = cm.get_cmap("jet")
                scan_colors_np = color_map(z_normalized)[:, :3].astype(np.float64)
                self.scan.colors = o3d.utility.Vector3dVector(scan_colors_np)

            if self.ego_chbox.checked:
                self.scan.transform(np.linalg.inv(self.cur_pose))

            self.widget3d.scene.remove_geometry(self.scan_name)
            self.widget3d.scene.add_geometry(self.scan_name, self.scan, self.scan_render)

        self.widget3d.scene.show_geometry(self.scan_name, self.scan_chbox.checked)

    def visualize_sdf_slice(self, data_packet = None):
        if data_packet is None:
            data_packet = self.cur_data_packet

        if self.sdf_chbox.checked and data_packet.sdf_slice_xyz is not None and data_packet.sdf_slice_rgb is not None:
            self.sdf_slice.points = o3d.utility.Vector3dVector(data_packet.sdf_slice_xyz)
            self.sdf_slice.colors = o3d.utility.Vector3dVector(data_packet.sdf_slice_rgb)

            if self.ego_chbox.checked:
                self.sdf_slice.transform(np.linalg.inv(self.cur_pose))

            self.widget3d.scene.remove_geometry(self.sdf_name)
            self.widget3d.scene.add_geometry(self.sdf_name, self.sdf_slice, self.sdf_render)

        self.widget3d.scene.show_geometry(self.sdf_name, self.sdf_chbox.checked)
    
    def send_data(self):
        packet = ControlPacket()
        packet.flag_pause = not self.slider_slam.is_on
        packet.flag_vis = self.slider_vis.is_on
        packet.flag_source = self.scan_regis_color_chbox.checked
        packet.flag_mesh = self.mesh_chbox.checked
        packet.flag_sdf = self.sdf_chbox.checked
        packet.flag_global = not self.local_map_chbox.checked
        packet.mc_res_m = self.mesh_mc_res_slider.int_value / 100.0
        packet.mesh_min_nn = self.mesh_min_nn_slider.int_value
        packet.mesh_freq_frame = self.mesh_freq_frame_slider.int_value
        packet.sdf_freq_frame = self.sdf_freq_frame_slider.int_value
        packet.sdf_slice_height = self.sdf_slice_height_slider.double_value
        packet.sdf_res_m = self.sdf_res_slider.int_value / 100.0
        packet.cur_frame_id = self.cur_frame_id

        self.q_vis2main.put(packet)

    def receive_data(self, q):
        if q is None:
            return

        data_packet: VisPacket = get_latest_queue(q)

        if data_packet is None:
            return
        
        self.cur_data_packet = data_packet

        self.cur_frame_id = data_packet.frame_id

        if data_packet.slam_finished:
            self.slam_finished = True
            self.local_map_chbox.checked = False
            self.ego_chbox.checked = False
            self.scan_chbox.checked = False
            self.followcam_chbox.checked = False

        if data_packet.travel_dist is not None:
            self.dist_info.text = "Travel Distance: {:.2f} m".format(data_packet.travel_dist)

        if data_packet.frame_id is not None:
            self.frame_info.text = "Frame: {}".format(data_packet.frame_id)
                
        if data_packet.has_neural_points:
            self.neural_points_info.text = "# Neural points: {} (local {}) [PIN Map size: {:.1f} MB]".format(
                data_packet.neural_points_data["count"],
                data_packet.neural_points_data["local_count"],
                data_packet.neural_points_data["map_memory_mb"]
            )
            
            self.visualize_neural_points()
        
        self.visualize_scan()

        self.visualize_sdf_slice()

        self.visualize_sdf_pool()

        self.visualize_mesh()

        if data_packet.gt_poses is not None:
            gt_position_np = data_packet.gt_poses[:, :3, 3]
            if gt_position_np.shape[0] > 1:
                self.gt_traj.points = o3d.utility.Vector3dVector(gt_position_np)
                gt_edges = np.array([[i, i + 1] for i in range(gt_position_np.shape[0] - 1)])
                self.gt_traj.lines = o3d.utility.Vector2iVector(gt_edges)
                self.gt_traj.paint_uniform_color(BLACK)
            
            if data_packet.slam_poses is None:

                self.cur_pose = data_packet.gt_poses[-1]
                
                self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
                self.sensor_cad.transform(data_packet.gt_poses[-1])

                self.range_circle = copy.deepcopy(self.range_circle_origin)
                self.range_circle.transform(data_packet.gt_poses[-1])  

                if self.ego_chbox.checked:
                    self.sensor_cad.transform(np.linalg.inv(self.cur_pose))
                    self.range_circle.transform(np.linalg.inv(self.cur_pose))

                if self.cad_chbox.checked:
                    self.widget3d.scene.remove_geometry(self.cad_name)
                    self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
                
                if self.range_circle_chbox.checked: 
                    self.widget3d.scene.remove_geometry(self.range_circle_name)
                    self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)

            if self.ego_chbox.checked:
                self.gt_traj.transform(np.linalg.inv(self.cur_pose))
                
            if self.gt_traj_chbox.checked:
                self.widget3d.scene.remove_geometry(self.gt_traj_name)
                self.widget3d.scene.add_geometry(self.gt_traj_name, self.gt_traj, self.traj_render)

        if data_packet.slam_poses is not None:

            self.cur_pose = data_packet.slam_poses[-1]
            
            slam_position_np = data_packet.slam_poses[:, :3, 3]
            if slam_position_np.shape[0] > 1:
                self.slam_traj.points = o3d.utility.Vector3dVector(slam_position_np)
                slam_edges = np.array([[i, i + 1] for i in range(slam_position_np.shape[0] - 1)])
                self.slam_traj.lines = o3d.utility.Vector2iVector(slam_edges)
                self.slam_traj.paint_uniform_color(RED)
            
            self.sensor_cad = copy.deepcopy(self.sensor_cad_origin)
            self.sensor_cad.transform(data_packet.slam_poses[-1])

            self.range_circle = copy.deepcopy(self.range_circle_origin)
            self.range_circle.transform(data_packet.slam_poses[-1])

            if self.ego_chbox.checked:
                self.slam_traj.transform(np.linalg.inv(self.cur_pose))
                self.sensor_cad.transform(np.linalg.inv(self.cur_pose))
                self.range_circle.transform(np.linalg.inv(self.cur_pose))

            if self.slam_traj_chbox.checked:
                self.widget3d.scene.remove_geometry(self.slam_traj_name)
                self.widget3d.scene.add_geometry(self.slam_traj_name, self.slam_traj, self.traj_render)

            if self.cad_chbox.checked:
                self.widget3d.scene.remove_geometry(self.cad_name)
                self.widget3d.scene.add_geometry(self.cad_name, self.sensor_cad, self.cad_render)
            
            if self.range_circle_chbox.checked: 
                self.widget3d.scene.remove_geometry(self.range_circle_name)
                self.widget3d.scene.add_geometry(self.range_circle_name, self.range_circle, self.ring_render)

            if data_packet.loop_edges is not None:
                loop_count = len(data_packet.loop_edges)
                self.loop_info.text = f"# Loop Closures: {loop_count}"

                if loop_count > 0:
                    self.loop_edges.points = o3d.utility.Vector3dVector(slam_position_np)
                    self.loop_edges.lines = o3d.utility.Vector2iVector(np.array(data_packet.loop_edges))
                    self.loop_edges.paint_uniform_color(GREEN)

                    if self.ego_chbox.checked:
                        self.loop_edges.transform(np.linalg.inv(self.cur_pose))

                    self.widget3d.scene.remove_geometry(self.loop_edges_name)
                    self.widget3d.scene.add_geometry(self.loop_edges_name, self.loop_edges, self.traj_render)
                    self.widget3d.scene.show_geometry(self.loop_edges_name, self.loop_edges_chbox.checked)
            
        if data_packet.odom_poses is not None:
            
            odom_position_np = data_packet.odom_poses[:, :3, 3]
            if odom_position_np.shape[0] > 1:
                self.odom_traj.points = o3d.utility.Vector3dVector(odom_position_np)
                odom_edges = np.array([[i, i + 1] for i in range(odom_position_np.shape[0] - 1)])
                self.odom_traj.lines = o3d.utility.Vector2iVector(odom_edges)
                self.odom_traj.paint_uniform_color(BLUE)

                if self.ego_chbox.checked:
                    self.odom_traj.transform(np.linalg.inv(self.cur_pose))

            if self.odom_traj_chbox.checked:
                self.widget3d.scene.remove_geometry(self.odom_traj_name)
                self.widget3d.scene.add_geometry(self.odom_traj_name, self.odom_traj, self.traj_render)


        # set up inital camera
        if not self.init:
            self.center_bev()

        self.init = True

        if self.ego_state_changed:
            self.center_bev_local()
            self.ego_state_changed = False

        if data_packet.finish:
            print("[GUI] Received terminate signal")
            # clean up the pipe
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True

    def center_bev(self):
        # set the view point to BEV of the current 3d objects
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(45, bounds, bounds.get_center())  # field of view, bound, center

    def center_bev_local(self):
        bounds = self.range_circle.get_axis_aligned_bounding_box()
        self.widget3d.setup_camera(45, bounds, bounds.get_center())  # field of view, bound, center

    def _update_thread(self):
        while True:
            if self.local_map_chbox.checked:
                time.sleep(0.1)
            elif self.slam_finished:
                time.sleep(1.5)
            else:
                time.sleep(0.2)
            
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                print("[GUI] Closing Visualization")
                break

            def update():

                if self.step % 1 == 0 and self.slider_vis.is_on: # receive latest data
                    self.receive_data(self.q_main2vis)

                if self.step % 5 == 0:
                    self.send_data()

                if self.followcam_chbox.checked and self.step % 1 == 0: 
                    if self.local_map_chbox.checked:
                        self.center_bev_local()
                    else:
                        self.center_bev()
                
                else:
                    while not self.q_main2vis.empty(): # free the queue
                        self.q_main2vis.get()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()

def generate_circle(radius=1.0, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    # Circle in the XY plane
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(num_points)  # Z-coordinates are 0 for a flat circle in XY-plane
    circle_points = np.vstack((x, y, z)).T  # Shape (num_points, 3)
    return circle_points

def model_matrix_to_extrinsic_matrix(model_matrix):
    return np.linalg.inv(model_matrix @ FromGLGamera)

def create_camera_intrinsic_from_size(width=1024, height=768, hfov=60.0, vfov=60.0):
    fx = (width / 2.0)  / np.tan(np.radians(hfov)/2)
    fy = (height / 2.0)  / np.tan(np.radians(vfov)/2)
    fx = fy # not sure why, but it looks like fx should be governed/limited by fy
    return np.array(
        [[fx, 0, width / 2.0],
         [0, fy, height / 2.0],
         [0, 0,  1]])


if __name__ == "__main__":
    main()
