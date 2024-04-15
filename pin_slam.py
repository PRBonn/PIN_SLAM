#!/usr/bin/env python3
# @file      pin_slam.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import sys
import numpy as np
import wandb
import torch
from rich import print
from tqdm import tqdm

from utils.config import Config
from utils.tools import *
from utils.loss import *
from utils.pgo import PoseGraphManager
from utils.loop_detector import NeuralPointMapContextManager, GTLoopManager, detect_local_loop
from utils.mesher import Mesher
from utils.tracker import Tracker
from utils.mapper import Mapper
from utils.visualizer import MapVisualizer
from model.neural_points import NeuralPoints
from model.decoder import Decoder
from dataset.slam_dataset import SLAMDataset
from dataset.dataset_indexing import set_dataset_path

'''
    📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency
     Y. Pan et al.
'''
def run_pin_slam(config_path=None, dataset_name=None, sequence_name=None, seed=None):

    config = Config()
    if config_path is not None:
        config.load(config_path)
        set_dataset_path(config, dataset_name, sequence_name)
        if seed is not None:
            config.seed = seed
        argv = ['pin_slam.py', config_path, dataset_name, sequence_name, str(seed)]
        run_path = setup_experiment(config, argv)
    else:
        if len(sys.argv) > 1:
            config.load(sys.argv[1])
        else:
            sys.exit("Please provide the path to the config file.\nTry: \
                    python3 pin_slam.py path_to_config.yaml [dataset_name] [sequence_name] [random_seed]")       
        # specific dataset [optional]
        if len(sys.argv) > 3:
            set_dataset_path(config, sys.argv[2], sys.argv[3])
        if len(sys.argv) > 4: # random seed [optional]
            config.seed = int(sys.argv[4])
        run_path = setup_experiment(config, sys.argv)
        print("[bold green]PIN-SLAM starts[/bold green]","📍" )

    # initialize the mlp decoder
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, config.sem_mlp_hidden_dim, config.sem_mlp_level, config.sem_class_count + 1) if config.semantic_on else None
    color_mlp = Decoder(config, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None

    # initialize the feature octree
    neural_points = NeuralPoints(config)

    # Load the decoder model
    if config.load_model: # not used
        load_decoder(config, geo_mlp, sem_mlp, color_mlp)

    # dataset
    dataset = SLAMDataset(config)

    # odometry tracker
    tracker = Tracker(config, neural_points, geo_mlp, sem_mlp, color_mlp)

    # mapper
    mapper = Mapper(config, dataset, neural_points, geo_mlp, sem_mlp, color_mlp)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, geo_mlp, sem_mlp, color_mlp)

    # pose graph manager (for back-end optimization) initialization
    pgm = PoseGraphManager(config)
    if config.pgo_on:      
        if dataset.gt_pose_provided: 
            pgm.add_pose_prior(0, dataset.poses_ref[config.begin_frame], fixed=True)
        else:
            pgm.add_pose_prior(0, np.eye(4), fixed=True)

    # loop closure detector
    if config.use_gt_loop:
        lcd_gt = GTLoopManager(config) 
    lcd_npmc = NeuralPointMapContextManager(config, mapper) # npmc: neural point map context

    # non-blocking visualizer
    if config.o3d_vis_on:
        o3d_vis = MapVisualizer(config)

    last_frame = min(config.end_frame, dataset.total_pc_count-1)
    loop_reg_failed_count = 0
        
    # for each frame
    for frame_id in tqdm(range(dataset.total_pc_count)): # frame id as the idx of the frame in the data folder without skipping
        if (frame_id < config.begin_frame or frame_id > config.end_frame or frame_id % config.every_frame != 0):
            continue
        # the actual frame id for the current run (may have some frame skip, may not start from the first in the data folder)
        used_frame_id = dataset.processed_frame 

        # I. Load data and preprocessing
        T0 = get_time()

        dataset.read_frame(frame_id)

        T1 = get_time()
        
        valid_frame = dataset.preprocess_frame(frame_id)
        if not valid_frame:
            dataset.processed_frame += 1
            continue 

        T2 = get_time()
        
        # II. Odometry
        if used_frame_id > 0: 
            if config.track_on:
                tracking_result = tracker.tracking(dataset.cur_source_points, dataset.cur_pose_guess_torch, 
                                                   dataset.cur_source_colors, dataset.cur_source_normals,
                                                   vis_result=config.o3d_vis_on and not config.o3d_vis_raw)
                cur_pose_torch, cur_odom_cov, weight_pc_o3d, valid_flag = tracking_result

                dataset.lose_track = not valid_flag
                mapper.lose_track = not valid_flag

                if not valid_flag and config.o3d_vis_on:
                    if o3d_vis.debug_mode > 0:
                        o3d_vis.stop()
                dataset.update_odom_pose(cur_pose_torch) # update dataset.cur_pose_torch
            else: # incremental mapping with gt pose
                if dataset.gt_pose_provided:
                    dataset.update_odom_pose(dataset.cur_pose_guess_torch) 
                else:
                    sys.exit("You are using the mapping mode, but no pose is provided.")

        neural_points.travel_dist = torch.tensor(np.array(dataset.travel_dist), device = config.device, dtype=config.dtype) # always update this
                                                                                                                                                            
        T3 = get_time()

        # III. Loop detection and pgo 
        if config.pgo_on: 
            if config.use_gt_loop:
                lcd_gt.add_node(used_frame_id, dataset.poses_w_closed[frame_id]) # set current node in the loop detector
            elif config.global_loop_on:
                if config.local_map_context and used_frame_id >= config.local_map_context_latency: # local map context
                    cur_frame = used_frame_id-config.local_map_context_latency
                    cur_pose = torch.tensor(dataset.pgo_poses[cur_frame], device=config.device, dtype=torch.float64)
                    neural_points.reset_local_map(cur_pose[:3,3], None, cur_frame, False, config.loop_local_map_time_window) 
                    context_pc_local = transform_torch(neural_points.local_neural_points.detach(), torch.linalg.inv(cur_pose)) # transformed back into the local frame
                    neural_points_feature = neural_points.local_geo_features[:-1].detach() if config.loop_with_feature else None
                    lcd_npmc.add_node(cur_frame, context_pc_local, neural_points_feature)
                else: # first frame not yet have local map, use scan context
                    lcd_npmc.add_node(used_frame_id, dataset.cur_point_cloud_torch)
            
            pgm.add_frame_node(used_frame_id, dataset.pgo_poses[-1]) # add new node and pose initial guess
            pgm.init_poses = dataset.pgo_poses

            if used_frame_id > 0:
                cur_edge_cov = cur_odom_cov if config.use_reg_cov_mat else None                    
                pgm.add_odometry_factor(used_frame_id, used_frame_id-1, dataset.last_odom_tran, cov = cur_edge_cov) # T_p<-c
                pgm.estimate_drift(dataset.travel_dist, used_frame_id) # estimate the current drift
                if config.pgo_with_pose_prior: # add pose prior
                    pgm.add_pose_prior(used_frame_id, dataset.pgo_poses[-1])
            
            local_map_context_loop = False
            if used_frame_id - pgm.last_loop_idx > config.pgo_freq and not dataset.stop_status:
                if config.use_gt_loop:
                    loop_id, loop_dist, loop_transform = lcd_gt.detect_loop() # T_l<-c
                    if loop_id is not None and not config.silence:
                        print("[bold red]GT loop event added: [/bold red]", lcd_gt.curr_node_idx, "---", loop_id, "(" , loop_dist, ")")
                else:  # detect candidate local loop, find the nearest history pose and activate certain local map
                    cur_pgo_poses = np.stack(dataset.pgo_poses)
                    dist_to_past = np.linalg.norm(cur_pgo_poses[:,:3,3] - cur_pgo_poses[-1,:3,3], axis=1) 
                    loop_candidate_mask = (dataset.travel_dist[-1] - dataset.travel_dist > config.min_loop_travel_dist_ratio*config.local_map_radius)
                    loop_id = None
                    if loop_candidate_mask.any(): # have at least one candidate
                        # firstly try to detect the local loop
                        loop_id, loop_dist, loop_transform = detect_local_loop(dist_to_past, loop_candidate_mask, dataset.pgo_poses, pgm.drift_radius, used_frame_id, loop_reg_failed_count, config.voxel_size_m*5.0, config.silence)
                        if loop_id is None and config.global_loop_on: # global loop detection (large drift)
                            loop_id, loop_cos_dist, loop_transform, local_map_context_loop = lcd_npmc.detect_global_loop(cur_pgo_poses, dataset.pgo_poses, pgm.drift_radius*config.loop_dist_drift_ratio_thre, loop_candidate_mask, neural_points)     
                if loop_id is not None:
                    if config.loop_z_check_on and abs(loop_transform[2,3]) > config.voxel_size_m*4.0: # for multi-floor buildings, z may cause ambiguilties
                        loop_id = None
                        if not config.silence:
                            print("[bold red]Delta z check failed, reject the loop[/bold red]") 
                if loop_id is not None: # if a loop is found, we refine loop closure transform initial guess with a scan-to-map registration                    
                    pose_init_np = dataset.pgo_poses[loop_id] @ loop_transform # T_w<-c = T_w<-l @ T_l<-c 
                    pose_init_torch = torch.tensor(pose_init_np, device=config.device, dtype=torch.float64)
                    neural_points.recreate_hash(pose_init_torch[:3,3], None, True, True, loop_id) # recreate hash and local map for registration, this is the reason why we'd better to keep the duplicated neural points until the end
                    loop_reg_source_point = dataset.cur_source_points.clone()
                    pose_refine_torch, loop_cov_mat, weight_pcd, reg_valid_flag = tracker.tracking(loop_reg_source_point, pose_init_torch, loop_reg=True, vis_result=config.o3d_vis_on)
                    # visualize the loop closure and loop registration
                    if config.o3d_vis_on and o3d_vis.debug_mode > 1:
                        points_torch_init = transform_torch(loop_reg_source_point, pose_init_torch) # apply transformation
                        points_o3d_init = o3d.geometry.PointCloud()
                        points_o3d_init.points = o3d.utility.Vector3dVector(points_torch_init.detach().cpu().numpy().astype(np.float64))
                        loop_neural_pcd = neural_points.get_neural_points_o3d(query_global=False, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1)
                        o3d_vis.update(points_o3d_init, neural_points=loop_neural_pcd, pause_now=True)
                        o3d_vis.update(weight_pcd, neural_points=loop_neural_pcd, pause_now=True)
                    pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                    loop_transform = np.linalg.inv(dataset.pgo_poses[loop_id]) @ pose_refine_np # T_l<-c = T_l<-w @ T_w<-c
                    if not config.silence:
                        print("[bold green]Refine loop transformation succeed [/bold green]")
                    # only conduct pgo when the loop and loop constraint is correct
                    cur_edge_cov = loop_cov_mat if config.use_reg_cov_mat else None
                    if reg_valid_flag: # refine succeed
                        reg_valid_flag = pgm.add_loop_factor(used_frame_id, loop_id, loop_transform, cov = cur_edge_cov)
                    if reg_valid_flag:
                        pgm.optimize_pose_graph() # conduct pgo
                        cur_loop_vis_id = used_frame_id-config.local_map_context_latency if local_map_context_loop else used_frame_id
                        pgm.loop_edges_vis.append(np.array([loop_id, cur_loop_vis_id],dtype=np.uint32)) # only for vis
                        pgm.loop_edges.append(np.array([loop_id, used_frame_id],dtype=np.uint32))
                        pgm.loop_trans.append(loop_transform)
                        # update the neural points and poses
                        pose_diff_torch = torch.tensor(pgm.get_pose_diff(), device=config.device, dtype=config.dtype)
                        dataset.cur_pose_torch = torch.tensor(pgm.cur_pose, device=config.device, dtype=config.dtype)
                        neural_points.adjust_map(pose_diff_torch) # transform neural points (position and orientation) along with associated frame poses
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, (not config.pgo_merge_map), config.rehash_with_time, used_frame_id) # recreate hash from current time
                        mapper.transform_data_pool(pose_diff_torch) # transform global pool
                        dataset.update_poses_after_pgo(pgm.cur_pose, pgm.pgo_poses)
                        pgm.last_loop_idx = used_frame_id
                        pgm.min_loop_idx = min(pgm.min_loop_idx, loop_id)
                        loop_reg_failed_count = 0
                        if config.o3d_vis_on:
                            o3d_vis.before_pgo = False
                    else:
                        if not config.silence:
                            print("[bold red]Registration failed, reject the loop candidate [/bold red]")
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, True, True, used_frame_id) # if failed, you need to reset the local map back to current frame
                        loop_reg_failed_count += 1
                        if config.o3d_vis_on and o3d_vis.debug_mode > 1:
                            o3d_vis.stop()

        T4 = get_time()
        
        # IV: Mapping and bundle adjustment
        # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
        # if the robot stop, also don't process this frame, since there's no new oberservations
        if not mapper.lose_track and not dataset.stop_status:
            mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch,
                                 dataset.cur_pose_torch, used_frame_id, (config.dynamic_filter_on and used_frame_id > 0))
        else: # lose track, still need to set back the local map
            mapper.determine_used_pose()
            neural_points.reset_local_map(dataset.cur_pose_torch[:3,3], None, used_frame_id)
            mapper.static_mask = None
                                
        T5 = get_time()

        # for the first frame, we need more iterations to do the initialization (warm-up)
        cur_iter_num = config.iters * config.init_iter_ratio if used_frame_id == 0 else config.iters
        if config.adaptive_iters and dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num-10)
        if used_frame_id == config.freeze_after_frame: # freeze the decoder after certain frame 
            freeze_decoders(geo_mlp, sem_mlp, color_mlp, config)

        # conduct local bundle adjustment (with lower frequency)
        if config.track_on and config.ba_freq_frame > 0 and (used_frame_id+1) % config.ba_freq_frame == 0:
            mapper.bundle_adjustment(config.ba_iters, config.ba_frame)
        
        # mapping with fixed poses (every frame)
        if used_frame_id % config.mapping_freq_frame == 0:
            mapper.mapping(cur_iter_num)
        
        T6 = get_time()

        # regular saving logs
        if config.log_freq_frame > 0 and (used_frame_id+1) % config.log_freq_frame == 0:
            dataset.write_results_log(used_frame_id, run_path)

        if not config.silence:
            print("time for frame reading          (ms):", (T1-T0)*1e3)
            print("time for frame preprocessing    (ms):", (T2-T1)*1e3)
            if config.track_on:
                print("time for odometry               (ms):", (T3-T2)*1e3)
            if config.pgo_on:
                print("time for loop detection and PGO (ms):", (T4-T3)*1e3)
            print("time for mapping preparation    (ms):", (T5-T4)*1e3)
            print("time for training               (ms):", (T6-T5)*1e3)

        # V: Mesh reconstruction and visualization
        if config.o3d_vis_on: # if visualizer is off, there's no need to reconstruct the mesh

            o3d_vis.cur_frame_id = frame_id # frame id in the data folder

            dataset.static_mask = mapper.static_mask
            dataset.update_o3d_map()
            if config.track_on and used_frame_id > 0 and (not o3d_vis.vis_pc_color) and (weight_pc_o3d is not None): 
                dataset.cur_frame_o3d = weight_pc_o3d

            T7 = get_time()

            if frame_id == last_frame:
                o3d_vis.vis_global = True
                o3d_vis.ego_view = False
                mapper.free_pool()

            neural_pcd = None
            if o3d_vis.render_neural_points or (frame_id == last_frame): # last frame also vis
                neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1) # select from geo_feature, ts and certainty

            # reconstruction by marching cubes
            mesher.ts = frame_id # deprecated
            cur_mesh = None
            if config.mesh_freq_frame > 0:
                if o3d_vis.render_mesh and (used_frame_id == 0 or frame_id == last_frame or (used_frame_id+1) % config.mesh_freq_frame == 0 or pgm.last_loop_idx == used_frame_id):             
                    
                    # update map bbx
                    global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23) # prime number
                    dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
                    
                    mesh_path = None # no need to save the mesh
                    if frame_id == last_frame and config.save_mesh: # for last frame
                        mc_cm_str = str(round(o3d_vis.mc_res_m*1e2))
                        mesh_path = os.path.join(run_path, "mesh", 'mesh_frame_' + str(frame_id) + "_" + mc_cm_str + "cm.ply")
                    
                    # figure out how to do it efficiently
                    if not o3d_vis.vis_global: # only build the local mesh
                        # cur_mesh = mesher.recon_aabb_mesh(dataset.cur_bbx, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)
                        chunks_aabb = split_chunks(global_neural_pcd_down, dataset.cur_bbx, o3d_vis.mc_res_m*100) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, o3d_vis.mc_res_m * 300) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
            cur_sdf_slice = None
            if config.sdfslice_freq_frame > 0:
                if o3d_vis.render_sdf and (used_frame_id == 0 or frame_id == last_frame or (used_frame_id + 1) % config.sdfslice_freq_frame == 0):
                    slice_res_m = config.voxel_size_m * 0.2
                    sdf_bound = config.surface_sample_range_m * 4.0
                    query_sdf_locally = True
                    if o3d_vis.vis_global:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.map_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, False, -sdf_bound, sdf_bound) # horizontal slice
                    else:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.cur_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # horizontal slice (local)
                    if config.vis_sdf_slice_v:
                        cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(dataset.cur_bbx, dataset.cur_pose_ref[0,3], slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # vertical slice (local)
                        cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                    else:
                        cur_sdf_slice = cur_sdf_slice_h
                                
            pool_pcd = mapper.get_data_pool_o3d(down_rate=17, only_cur_data=o3d_vis.vis_only_cur_samples) if o3d_vis.render_data_pool else None # down rate should be a prime number
            loop_edges = pgm.loop_edges_vis if config.pgo_on else None
            o3d_vis.update_traj(dataset.cur_pose_ref, dataset.odom_poses, dataset.gt_poses, dataset.pgo_poses, loop_edges)
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
            
            T8 = get_time()

            if not config.silence:
                print("time for o3d update             (ms):", (T7-T6)*1e3)
                print("time for visualization          (ms):", (T8-T7)*1e3)

        cur_frame_process_time = np.array([T2-T1, T3-T2, T5-T4, T6-T5, T4-T3]) # loop & pgo in the end, visualization and I/O time excluded
        dataset.time_table.append(cur_frame_process_time) # in s

        if config.wandb_vis_on:
            wandb_log_content = {'frame': used_frame_id, 'timing(s)/preprocess': T2-T1, 'timing(s)/tracking': T3-T2, 'timing(s)/pgo': T4-T3, 'timing(s)/mapping': T6-T4} 
            wandb.log(wandb_log_content)

        dataset.processed_frame += 1
    
    # VI. Save results
    if config.track_on:
        pose_eval_results = dataset.write_results(run_path)
    if config.pgo_on and pgm.pgo_count>0:
        print("# Loop corrected: ", pgm.pgo_count)
        pgm.write_g2o(os.path.join(run_path, "final_pose_graph.g2o"))
        pgm.write_loops(os.path.join(run_path, "loop_log.txt"))
        if config.o3d_vis_on:
            pgm.plot_loops(os.path.join(run_path, "loop_plot.png"), vis_now=False)  

    neural_points.recreate_hash(None, None, False, False) # merge the final neural point map
    neural_points.prune_map(config.max_prune_certainty) # prune uncertain points for the final output 
    if config.o3d_vis_on:
        neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode=o3d_vis.neural_points_vis_mode)
        if config.save_map:
            o3d.io.write_point_cloud(os.path.join(run_path, "map", "neural_points.ply"), neural_pcd) # write the neural point cloud
    neural_points.clear_temp() # clear temp data for output

    if config.save_map:
        save_implicit_map(run_path, neural_points, geo_mlp, color_mlp, sem_mlp)
    if config.save_merged_pc:
        dataset.write_merged_point_cloud(run_path) # replay: save merged point cloud map
    
    if config.o3d_vis_on:
        while True:
            o3d_vis.ego_view = False
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
            o3d_vis.update_traj(dataset.cur_pose_ref, dataset.odom_poses, dataset.gt_poses, dataset.pgo_poses, loop_edges)
    
    return pose_eval_results

if __name__ == "__main__":

    run_pin_slam()