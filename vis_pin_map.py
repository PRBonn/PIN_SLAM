#!/usr/bin/env python3
# @file      vis_pin_map.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import glob
import os
import sys

import open3d as o3d
import torch
from rich import print

from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.mesher import Mesher
from utils.tools import setup_experiment, split_chunks
from utils.visualizer import MapVisualizer

'''
    load the pin-map and do the reconstruction 
'''
def vis_pin_map():

    config = Config()
    if len(sys.argv) > 1:
        result_folder = sys.argv[1]
        yaml_files = glob.glob(f"{result_folder}/*.yaml")
        if len(yaml_files) > 1: # Check if there is exactly one YAML file
            sys.exit("There are multiple YAML files. Please handle accordingly.")
        elif len(yaml_files) == 0:  # If no YAML files are found
            sys.exit("No YAML files found in the specified path.")
        config.load(yaml_files[0])
        config.model_path = os.path.join(result_folder, "model", "pin_map.pth")
    else:
        sys.exit("Please provide the path to the result folder.\n\
                 Try: python vis_pin_map.py xxx/result/path\
                [optional: mesh_res_m] [optional: cropped.ply]  [optional: output_mesh_file] [optional: mc_nn]")

    print("[bold green]Load PIN Map[/bold green]","📍" )

    run_path = setup_experiment(config, sys.argv, debug_mode=True)
    
    # initialize the mlp decoder
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, config.sem_mlp_hidden_dim, config.sem_mlp_level, config.sem_class_count + 1) 
    color_mlp = Decoder(config, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel)

    # initialize the feature octree
    neural_points = NeuralPoints(config)

    # Load the map
    loaded_model = torch.load(config.model_path)
    neural_points = loaded_model["neural_points"]

    # print(loaded_model.keys())
    geo_mlp.load_state_dict(loaded_model["geo_decoder"])
    if 'sem_decoder' in loaded_model.keys():
        sem_mlp.load_state_dict(loaded_model["sem_decoder"])
    if 'color_decoder' in loaded_model.keys():
        color_mlp.load_state_dict(loaded_model["color_decoder"])
    print("PIN Map loaded")

    if config.o3d_vis_on:
        vis = MapVisualizer(config)

    neural_points.recreate_hash(neural_points.neural_points[0], torch.eye(3).cuda(), False, False)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, geo_mlp, sem_mlp, color_mlp)

    mesh_vox_size_m = None
    if len(sys.argv) > 2:
        mesh_vox_size_m = float(sys.argv[2])
        print("Marching cubes resolution: ", mesh_vox_size_m, " m")

    down_rate = 1

    crop_file_name = "neural_points.ply" # default name
    if len(sys.argv) > 3: # only use cropped bbx for meshing
        crop_file_name = sys.argv[3]
        cropped_ply_path = os.path.join(result_folder, "map", crop_file_name)
        cropped_pc = o3d.io.read_point_cloud(cropped_ply_path)
        mesh_aabb = cropped_pc.get_axis_aligned_bounding_box()
        chunks_aabb = split_chunks(cropped_pc, mesh_aabb, mesh_vox_size_m*300) 
        print("Load cropped region")
    else:
        neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=2, random_down_ratio=down_rate)
        mesh_aabb = neural_points.get_map_o3d_bbx()
        if mesh_vox_size_m is not None:
            chunks_aabb = split_chunks(neural_pcd, mesh_aabb, mesh_vox_size_m*300) 
        # print("AABB for meshing: ", mesh_aabb)
    
    print("Number of chunks for reconstruction:", len(chunks_aabb))
    
    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=2, random_down_ratio=down_rate)
    neural_pcd_cropped = neural_pcd.crop(mesh_aabb)
    cropped_np_out_path = os.path.join(result_folder, "map", "out_ts_" + crop_file_name)
    o3d.io.write_point_cloud(cropped_np_out_path, neural_pcd_cropped)

    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=0, random_down_ratio=down_rate)
    neural_pcd_cropped = neural_pcd.crop(mesh_aabb)
    cropped_np_out_path = os.path.join(result_folder, "map", "out_feature_" + crop_file_name)
    o3d.io.write_point_cloud(cropped_np_out_path, neural_pcd_cropped)

    print("Neural point count:", neural_points.count())
    # neural_points_vis_mode = 2

    if len(sys.argv) > 4:
        out_mesh_path = os.path.join(result_folder, "mesh", sys.argv[4])
        print("Output the mesh to: ", out_mesh_path)
    else:
        out_mesh_path = None   
        print("Do not output mesh")
    
    mesh_min_nn_used = 9
    if len(sys.argv) > 5:
        mesh_min_nn_used = int(sys.argv[5])

    cur_mesh = None
    if mesh_vox_size_m is not None:
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, mesh_vox_size_m, out_mesh_path, False, config.semantic_on, 
                                                     config.color_on, filter_isolated_mesh=True, mesh_min_nn=mesh_min_nn_used)
    
    if config.o3d_vis_on:
        while True:
            if vis.render_neural_points:
                neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=vis.neural_points_vis_mode, random_down_ratio=down_rate)
            vis.update(mesh=cur_mesh, neural_points=neural_pcd)
            
    
if __name__ == "__main__":
    vis_pin_map()
