#!/usr/bin/env python3
# @file      vis_pin_map.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import glob
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import dtyper as typer
from rich import print

from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.mesher import Mesher
from utils.tools import setup_experiment, split_chunks, load_decoders, remove_gpu_cache

from gui import slam_gui
from gui.gui_utils import ParamsGUI, VisPacket


'''
    load the pin-map and do the reconstruction 
'''

app = typer.Typer(add_completion=False, rich_markup_mode="rich", context_settings={"help_option_names": ["-h", "--help"]})

docstring = f"""
:round_pushpin: Inspect the PIN Map \n

[bold green]Examples: [/bold green]

# Inspect the PIN Map stored in a mapping result folder, showing both the neural points and the mesh reconstructed with a certain marching cubes resolution
$ python3 vis_pin_map.ply <result-dir>:open_file_folder: -m <mesh_res_m>

# Additionally, you can specify the cropped point cloud file and the output mesh file
$ python3 vis_pin_map.ply <result-dir>:open_file_folder: -m <mesh_res_m> -c <cropped_ply_filename>:page_facing_up: -o <output_mesh_filename>:page_facing_up:

"""

@app.command(help=docstring)
def vis_pin_map(
    result_folder: str = typer.Argument(..., help='Path to the result folder'),
    mesh_res_m: float = typer.Option(None, '--mesh_res_m', '-m', help='Resolution of the mesh in meters'),
    cropped_ply_filename: str = typer.Option("neural_points.ply", '--cropped_ply_filename', '-c', help='Path to the cropped point cloud file'),
    output_mesh_filename: str = typer.Option(None, '--output_mesh_filename', '-o', help='Path to the output mesh file'),
    mc_nn: int = typer.Option(9, '--mc_nn', '-n', help='Minimum number of neighbors for SDF querying for marching cubes'),
    o3d_vis_on: bool = typer.Option(True, '--visualize_on', '-v', help='Turn on the visualizer'),
):

    config = Config()

    yaml_files = glob.glob(f"{result_folder}/*.yaml")
    if len(yaml_files) > 1: # Check if there is exactly one YAML file
        sys.exit("There are multiple YAML files. Please handle accordingly.")
    elif len(yaml_files) == 0:  # If no YAML files are found
        sys.exit("No YAML files found in the specified path.")
    config.load(yaml_files[0])
    config.model_path = os.path.join(result_folder, "model", "pin_map.pth")

    print("[bold green]Load and inspect PIN Map[/bold green]","📍" )

    run_path = setup_experiment(config, sys.argv, debug_mode=True)

    mp.set_start_method("spawn") # don't forget this
    
    # initialize the mlp decoder
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, config.sem_mlp_hidden_dim, config.sem_mlp_level, config.sem_class_count + 1) if config.semantic_on else None
    color_mlp = Decoder(config, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None
    
    mlp_dict = {}
    mlp_dict["sdf"] = geo_mlp
    mlp_dict["semantic"] = sem_mlp
    mlp_dict["color"] = color_mlp

    # initialize the neural point features
    neural_points: NeuralPoints = NeuralPoints(config)

    # Load the map
    loaded_model = torch.load(config.model_path)
    neural_points = loaded_model["neural_points"]
    load_decoders(loaded_model, mlp_dict) 
    neural_points.temporal_local_map_on = False
    neural_points.recreate_hash(neural_points.neural_points[0], torch.eye(3).cuda(), False, False)
    neural_points.compute_feature_principle_components(down_rate = 59)
    print("PIN Map loaded")

    # mesh reconstructor
    mesher = Mesher(config, neural_points, mlp_dict)
    
    mesh_on = (mesh_res_m is not None)  
    if mesh_on:
        config.mc_res_m = mesh_res_m
        config.mesh_min_nn = mc_nn
    
    q_main2vis = q_vis2main = None
    if o3d_vis_on:
        # communicator between the processes
        q_main2vis = mp.Queue() 
        q_vis2main = mp.Queue()

        params_gui = ParamsGUI(
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            config=config,
            local_map_default_on=False, 
            mesh_default_on=mesh_on,
            neural_point_map_default_on=config.neural_point_map_default_on,
        )
        gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
        gui_process.start()
        time.sleep(3) # second
    
    cur_mesh = None
    out_mesh_path = None

    if mesh_on:
        cropped_ply_path = os.path.join(result_folder, "map", cropped_ply_filename)
        if os.path.exists(cropped_ply_path):
            cropped_pc = o3d.io.read_point_cloud(cropped_ply_path)
            print("Load region for meshing from {}".format(cropped_ply_path))
        
        else:
            cropped_pc = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23)
    
        mesh_aabb = cropped_pc.get_axis_aligned_bounding_box()
        chunks_aabb = split_chunks(cropped_pc, mesh_aabb, mesh_res_m*300) 

        if output_mesh_filename is not None:
            out_mesh_path = os.path.join(result_folder, "mesh", output_mesh_filename)
            print("Output the mesh to: ", out_mesh_path)

        mc_cm_str = str(round(mesh_res_m*1e2))
        print("Reconstructing the mesh with resolution {} cm".format(mc_cm_str))
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, mesh_res_m, out_mesh_path, False, config.semantic_on, 
                                                      config.color_on, filter_isolated_mesh=True, mesh_min_nn=mc_nn)
        print("Reconstructing the global mesh done")
    
    remove_gpu_cache()

    if o3d_vis_on:

        while True:
            if not q_vis2main.empty():
                q_vis2main.get()

            packet_to_vis: VisPacket = VisPacket(slam_finished=True)

            if not neural_points.is_empty():
                packet_to_vis.add_neural_points_data(neural_points, only_local_map=False, pca_color_on=True)

            if cur_mesh is not None:
                packet_to_vis.add_mesh(np.array(cur_mesh.vertices, dtype=np.float64), np.array(cur_mesh.triangles), np.array(cur_mesh.vertex_colors, dtype=np.float64))
                cur_mesh = None

            q_main2vis.put(packet_to_vis)
            time.sleep(1.0) 
            
    
if __name__ == "__main__":
    app()