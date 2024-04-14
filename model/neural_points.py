#!/usr/bin/env python3
# @file      neural_points.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import open3d as o3d
from rich import print
import numpy as np
import matplotlib.cm as cm

from utils.config import Config
from utils.tools import voxel_down_sample_torch, voxel_down_sample_min_value_torch, apply_quaternion_rotation, transform_batch_torch, rotmat_to_quat, quat_multiply, get_time

class NeuralPoints(nn.Module):

    def __init__(self, config: Config) -> None:
        
        super().__init__()

        self.config = config

        self.silence = config.silence

        # self.point_level_num = config.point_level_num

        self.geo_feature_dim = config.feature_dim
        self.geo_feature_std = config.feature_std

        self.color_feature_dim = config.feature_dim
        self.color_feature_std = config.feature_std

        if config.use_gaussian_pe:
            self.position_encoder_geo = GaussianFourierFeatures(config)
            self.position_encoder_color = GaussianFourierFeatures(config)
        else:
            self.position_encoder_geo = PositionalEncoder(config)
            self.position_encoder_color = PositionalEncoder(config)

        self.mean_grid_sampling = False # NOTE: sample the gravity center of the points inside the voxel or keep the point that is closest to the voxel center

        self.device = config.device
        self.dtype = config.dtype
        self.idx_dtype = torch.int64 # torch.int64/32 does not have much speed difference
        
        self.resolution = config.voxel_size_m

        self.buffer_size = config.buffer_size

        self.temporal_local_map_on = True
        self.local_map_radius = self.config.local_map_radius
        self.diff_travel_dist_local = self.config.local_map_radius * self.config.local_map_travel_dist_ratio 
        
        self.diff_ts_local = self.config.diff_ts_local # not used now, switch to travel distance

        self.local_orientation = torch.eye(3, device=self.device)

        self.cur_ts = 0 # current frame No. or the current timestamp
        self.max_ts = 0

        self.travel_dist = None # for determine the local map, update from the dataset class for each frame
        self.est_poses = None
        self.after_pgo = False

        # for hashing (large prime numbers)
        self.primes = torch.tensor([73856093, 19349669, 83492791], dtype=self.idx_dtype, device=self.device)
        
        # initialization
        # the global map
        self.buffer_pt_index = torch.full((self.buffer_size,), -1, dtype=self.idx_dtype, device=self.device)
        
        self.neural_points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.point_orientations = torch.empty((0, 4), dtype=self.dtype, device=self.device) # as quaternion
        self.geo_features = torch.empty((1, self.geo_feature_dim), dtype=self.dtype, device=self.device)
        if self.config.color_on:
            self.color_features = torch.empty((1, self.color_feature_dim), dtype=self.dtype, device=self.device)
        else:
            self.color_features = None
        self.point_ts_create = torch.empty((0), device=self.device, dtype=torch.long) # create ts
        self.point_ts_update = torch.empty((0), device=self.device, dtype=torch.long) # last update ts
        self.point_certainties = torch.empty((0), dtype=self.dtype, device=self.device)

        # the local map
        self.local_neural_points = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        self.local_point_orientations = torch.empty((0, 4), dtype=self.dtype, device=self.device) # as quaternion
        self.local_geo_features = nn.Parameter() 
        self.local_color_features = nn.Parameter() 
        self.local_point_certainties = torch.empty((0), dtype=self.dtype, device=self.device)
        self.local_point_ts_update = torch.empty((0), device=self.device, dtype=torch.long)
        self.local_mask = None
        self.global2local = None

        # set neighborhood search region
        self.set_search_neighborhood(num_nei_cells=config.num_nei_cells, search_alpha=config.search_alpha)

        self.memory_footprint = []

        self.to(self.device)

    def is_empty(self):
        return self.neural_points.shape[0] == 0
    
    def count(self): 
        return self.neural_points.shape[0]
    
    def local_count(self): 
        return self.local_neural_points.shape[0]
    
    def print_memory(self):
        if not self.silence:
            print("# Global neural point: %d" % (self.count()))
            print("# Local  neural point: %d" % (self.local_count()))
        neural_point_count = self.count()
        point_dim = self.config.feature_dim + 3 + 4 # feature plus neural point position and orientation
        if self.color_features is not None:
            point_dim += self.config.feature_dim # also include the color feature
        cur_memory = neural_point_count * point_dim * 4 / 1024 / 1024 # as float32
        if not self.silence:
            print("Memory consumption: %f (MB)" %cur_memory)
        self.memory_footprint.append(cur_memory)

    def get_neural_points_o3d(self, query_global: bool = True, color_mode: int = -1, 
                              random_down_ratio: int = 1):
        
        ratio_vis = 1.5
        # TODO: visualize orientation as normal

        if query_global:
            neural_points_np = self.neural_points[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
            # points_orientation_np = self.point_orientations[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
        else:
            neural_points_np = self.local_neural_points[::random_down_ratio].cpu().detach().numpy().astype(np.float64)

        # neural_points_np = self.neural_points[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
        neural_pc_o3d = o3d.geometry.PointCloud()
        neural_pc_o3d.points = o3d.utility.Vector3dVector(neural_points_np)
        
        if color_mode == 0: # "geo_feature"
            if query_global:
                neural_features_vis = self.geo_features[:-1:random_down_ratio].detach()
            else:
                neural_features_vis = self.local_geo_features[:-1:random_down_ratio].detach() 
            neural_features_vis = F.normalize(neural_features_vis, p=2, dim=1)
            neural_features_np = neural_features_vis.cpu().numpy().astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(neural_features_np[:,0:3]*ratio_vis)

        elif color_mode == 1: # "color_feature"
            if self.color_features is None:
                return neural_pc_o3d
            if query_global:
                neural_features_vis = self.color_features[:-1:random_down_ratio].detach()
            else:
                neural_features_vis = self.local_color_features[:-1:random_down_ratio].detach()
            neural_features_vis = F.normalize(neural_features_vis, p=2, dim=1)
            neural_features_np = neural_features_vis.cpu().numpy().astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(neural_features_np[:,0:3]*ratio_vis)
        
        elif color_mode == 2: # "ts": # frame number (ts) as the color
            if query_global:
                if self.config.use_mid_ts:
                    show_ts = ((self.point_ts_create + self.point_ts_update) / 2).long()
                else:
                    show_ts = self.point_ts_create
                ts_np = show_ts[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
            else:
                ts_np = self.local_point_ts_update[::random_down_ratio].cpu().detach().numpy().astype(np.float64)
            ts_np = np.clip(ts_np/self.max_ts, 0., 1.)
            color_map = cm.get_cmap('jet')
            ts_color = color_map(ts_np)[:, :3].astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(ts_color)
        
        elif color_mode == 3: # "certainty" # certainty as color 
            if query_global:
                certainty_np = 1. - self.point_certainties[::random_down_ratio].cpu().detach().numpy().astype(np.float64)/1000.
            else:
                certainty_np = 1. - self.local_point_certainties[::random_down_ratio].cpu().detach().numpy().astype(np.float64)/1000.
            # print(self.local_point_certainties)
            certainty_color = np.repeat(certainty_np.reshape(-1, 1),3,axis=1) 
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(certainty_color)

        elif color_mode == 4: # "random" # random color 
            random_color = np.random.rand(neural_points_np.shape[0], 3).astype(np.float64)
            neural_pc_o3d.colors = o3d.utility.Vector3dVector(random_color)
        
        return neural_pc_o3d
        
    def update(self, points: torch.Tensor, sensor_position: torch.Tensor, sensor_orientation: torch.Tensor, cur_ts):
        # update the neural point map using new observations
        
        cur_resolution = self.resolution
        # if self.mean_grid_sampling:
        #     sample_points = meanGridSampling(points, resolution=cur_resolution) 
        sample_idx = voxel_down_sample_torch(points, cur_resolution) # take the point that is the closest to the voxel center (now used)
        sample_points = points[sample_idx]

        grid_coords = (sample_points / cur_resolution).floor().to(self.primes)
        buffer_size = int(self.buffer_size)
        hash = torch.fmod((grid_coords * self.primes).sum(-1), buffer_size)
        
        hash_idx = self.buffer_pt_index[hash]

        # not occupied before or is occupied but already far away (then it would be a hash collision)
        if not self.is_empty():
            vec_points = self.neural_points[hash_idx] - sample_points
            dist2 = torch.sum(vec_points**2, dim=-1)

            # the voxel is not occupied before or the case when hash collision happens
            
            # delta_t = (cur_ts - self.point_ts_create[hash_idx]) # use time diff
            delta_travel_dist = self.travel_dist[cur_ts] - self.travel_dist[self.point_ts_update[hash_idx]] # use travel dist diff

            # the last time mask is necessary (but better change to the accumulated distance or the pose uncertainty), done
            update_mask = (hash_idx == -1) | (dist2 > 3*cur_resolution**2) | (delta_travel_dist > self.diff_travel_dist_local) 
            
            # if torch.sum((hash_idx != -1) & (dist2 > 3*cur_resolution**2)) > 0:
            #     print("Collision detected !!!!!!!")   
            # so actually almost no collision, for larger scenes, there are collisions (but would not exsit for local map)
        else:
            update_mask = torch.ones(hash_idx.shape, dtype=torch.bool, device=self.device)

        added_pt = sample_points[update_mask] 
        new_point_count = added_pt.shape[0]

        cur_pt_idx = self.buffer_pt_index[hash]
        # allocate new neural points
        cur_pt_count = self.neural_points.shape[0]
        cur_pt_idx[update_mask] = torch.arange(new_point_count, dtype=self.idx_dtype, device=self.device) + cur_pt_count
        
        self.buffer_pt_index[hash] = cur_pt_idx
        self.neural_points = torch.cat((self.neural_points, added_pt), 0)

        added_orientations = [[1, 0, 0, 0]] * new_point_count
        added_orientations = torch.tensor(added_orientations, dtype=self.dtype, device=self.device) 
        self.point_orientations = torch.cat((self.point_orientations, added_orientations), 0)

        new_points_ts = torch.ones(new_point_count, device=self.device, dtype=torch.long) * cur_ts
        self.point_ts_create = torch.cat((self.point_ts_create, new_points_ts), 0)
        self.point_ts_update = torch.cat((self.point_ts_update, new_points_ts), 0)  
        
        # with padding in the end
        new_fts = self.geo_feature_std*torch.randn(new_point_count+1, self.geo_feature_dim, device=self.device, dtype=self.dtype)
        self.geo_features = torch.cat((self.geo_features[:-1], new_fts),0)

        # with padding in the end
        if self.color_features is not None:
            new_fts = self.color_feature_std*torch.randn(new_point_count+1, self.color_feature_dim, device=self.device, dtype=self.dtype)
            self.color_features = torch.cat((self.color_features[:-1], new_fts),0)

        new_certainty = torch.zeros(new_point_count, device=self.device, dtype=self.dtype, requires_grad=False)
        self.point_certainties = torch.cat((self.point_certainties, new_certainty),0)
        
        self.reset_local_map(sensor_position, sensor_orientation, cur_ts) # no need to recreate hash

    def reset_local_map(self, sensor_position: torch.Tensor, sensor_orientation: torch.Tensor, cur_ts: int, 
                        use_travel_dist: bool = True, diff_ts_local: int = 50):
        
        self.cur_ts = cur_ts
        self.max_ts = max(self.max_ts, cur_ts)
        
        vec2sensor = self.neural_points-sensor_position
        dist2sensor = torch.sum(vec2sensor**2, dim=-1) # dist square
        
        if self.config.use_mid_ts:
            point_ts_used = ((self.point_ts_create + self.point_ts_update) / 2).long() # still as dtype long
        else:
            point_ts_used = self.point_ts_create

        if use_travel_dist:
            delta_travel_dist = torch.abs(self.travel_dist[cur_ts] - self.travel_dist[point_ts_used])
            local_mask = (dist2sensor < self.local_map_radius**2) & (delta_travel_dist < self.diff_travel_dist_local)
        else: # use delta_t
            delta_t = torch.abs(cur_ts - point_ts_used)
            local_mask = (dist2sensor < self.local_map_radius**2) & (delta_t < diff_ts_local)

        self.local_neural_points = self.neural_points[local_mask]
        self.local_point_orientations = self.point_orientations[local_mask]
        self.local_point_certainties = self.point_certainties[local_mask]
        self.local_point_ts_update = self.point_ts_update[local_mask]

        local_mask = torch.cat((local_mask, torch.tensor([True], device=self.device))) # padding with one element in the end
        self.local_mask = local_mask
        
        global2local = torch.full_like(local_mask, -1).long() # if Flase (not in the local map), the mapping get an idx as -1
        local_indices = torch.nonzero(local_mask).flatten()
        local_point_count = local_indices.size(0)
        global2local[local_indices] = torch.arange(local_point_count, device=self.device)
        global2local[-1] = -1 # invalid idx is still invalid after mapping

        self.global2local = global2local 
        
        self.local_geo_features = nn.Parameter(self.geo_features[local_mask])
        if self.color_features is not None:
            self.local_color_features = nn.Parameter(self.color_features[local_mask])

        self.local_orientation = sensor_orientation # not used

    def assign_local_to_global(self):

        local_mask = self.local_mask
        self.neural_points[local_mask[:-1]] = self.local_neural_points
        self.point_orientations[local_mask[:-1]] = self.local_point_orientations
        self.geo_features[local_mask] = self.local_geo_features.data
        if self.color_features is not None:
            self.color_features[local_mask] = self.local_color_features.data
        self.point_certainties[local_mask[:-1]] = self.local_point_certainties
        self.point_ts_update[local_mask[:-1]] = self.local_point_ts_update

        # print("mean certainty for the neural points:", torch.mean(self.point_certainties))

    def query_feature(self, query_points: torch.Tensor, query_ts: torch.Tensor = None,
                      training_mode: bool = True, query_locally: bool = True,
                      query_geo_feature: bool = True, query_color_feature: bool = False): 
        
        if not query_geo_feature and not query_color_feature:
            sys.exit("you need to at least query one kind of feature")
        
        batch_size = query_points.shape[0]
        
        geo_features_vector = None
        color_features_vector = None

        nn_k = self.config.query_nn_k

        # T0 = get_time()

        # the slow part
        dists2, idx = self.radius_neighborhood_search(query_points,
                                                      time_filtering=self.temporal_local_map_on and query_locally)
        
        # [N, K], [N, K]
        # if query globally, we do not have the time filtering
        
        # T10 = get_time()

        # print("K=", idx.shape[-1]) # K
        if query_locally:
            idx = self.global2local[idx] # [N, K] # get the local idx using the global2local mapping
        
        nn_counts = (idx >= 0).sum(dim=-1) # then it could be larger than nn_k because this is before the sorting
        
        # T1 = get_time()
        
        dists2[idx==-1] = 9e3 # invalid, set to large distance
        sorted_dist2, sorted_neigh_idx = torch.sort(dists2, dim=1) # sort according to distance
        sorted_idx = idx.gather(1, sorted_neigh_idx)
        dists2 = sorted_dist2[:,:nn_k] # only take the knn
        idx = sorted_idx[:,:nn_k] # sorted local idx, only take the knn            
        
        # dist2, idx are all with the shape [N, K]

        # T2 = get_time()

        valid_mask = idx >= 0 # [N, K]

        if query_geo_feature:
            geo_features = torch.zeros(batch_size, nn_k, self.geo_feature_dim, device=self.device, dtype=self.dtype) # [N, K, F] 
            if query_locally:
                geo_features[valid_mask] = self.local_geo_features[idx[valid_mask]]
            else:
                geo_features[valid_mask] = self.geo_features[idx[valid_mask]]
            if self.config.layer_norm_on:
                geo_features = F.layer_norm(geo_features, [self.geo_feature_dim])
        if query_color_feature and self.color_features is not None:
            color_features = torch.zeros(batch_size, nn_k, self.color_feature_dim, device=self.device, dtype=self.dtype) # [N, K, F] 
            if query_locally:
                color_features[valid_mask] = self.local_color_features[idx[valid_mask]]
            else:
                color_features[valid_mask] = self.color_features[idx[valid_mask]]
            if self.config.layer_norm_on:
                color_features = F.layer_norm(color_features, [self.color_feature_dim])

        N, K = valid_mask.shape # K = nn_k here

        if query_locally:
            certainty = self.local_point_certainties[idx] # [N, K]
            neighb_vector = query_points.view(-1, 1, 3) - self.local_neural_points[idx] # [N, K, 3]
            quat = self.local_point_orientations[idx]  # [N, K, 4]            
        else:
            certainty = self.point_certainties[idx] # [N, K]
            neighb_vector = query_points.view(-1, 1, 3) - self.neural_points[idx]  # [N, K, 3]
            quat = self.point_orientations[idx]  # [N, K, 4]

        # quat[...,1:] *= -1. # inverse (not needed)
        # This has been doubly checked
        if self.after_pgo:
            neighb_vector = apply_quaternion_rotation(quat, neighb_vector) # [N, K, 3] # passive rotation (axis rotation w.r.t point) 
        neighb_vector[~valid_mask] = torch.zeros(1, 3, device=self.device, dtype=self.dtype)

        if self.config.pos_encoding_band > 0:
            neighb_vector = self.position_encoder_geo(neighb_vector) # [N, K, P]

        if query_geo_feature:
            geo_features_vector = torch.cat((geo_features, neighb_vector), dim=2) # [N, K, F+P]
        if query_color_feature and self.color_features is not None:
            color_features_vector = torch.cat((color_features, neighb_vector), dim=2) # [N, K, F+P]

        eps = 1e-15 # avoid nan (dividing by 0)
        
        weight_vector = 1. / (dists2 + eps) # [N, K] # Inverse distance weighting (IDW), distance square
        
        weight_vector[~valid_mask] = 0. # pad for invalid voxels
        weight_vector[nn_counts == 0] = eps # all 0 would cause NaN during normalization
        
        # apply the normalization of weight 
        weight_row_sums = torch.sum(weight_vector, dim=1).unsqueeze(1)
        weight_vector = torch.div(weight_vector, weight_row_sums) # [N, K] # normalize the weight, to make the sum as 1

        # print(weight_vector)
        weight_vector[~valid_mask] = 0. # invalid has zero weight

        with torch.no_grad():
            # Certainty accumulation for each neural point according to the weight 
            # Use scatter_add_ to accumulate the values for each index
            if training_mode: # only do it during the training mode
                idx[~valid_mask] = 0 # scatter_add don't accept -1 index
                if query_locally:
                    self.local_point_certainties.scatter_add_(dim=0, index=idx.flatten(), src=weight_vector.flatten()) 
                    if query_ts is not None: # update the last update ts for each neural point
                        idx_ts = query_ts.view(-1,1).repeat(1, K) 
                        idx_ts[~valid_mask] = 0
                        self.local_point_ts_update.scatter_reduce_(dim=0, index=idx.flatten(), 
                                                                   src=idx_ts.flatten(), reduce="amax", include_self=True)
                        # print(self.local_point_ts_update)
                else:
                    self.point_certainties.scatter_add_(dim=0, index=idx.flatten(), src=weight_vector.flatten()) 
                # queried_certainty = None

                certainty[~valid_mask] = 0.0
                queried_certainty = torch.sum(certainty * weight_vector, dim=1)

            else: # inference mode
                certainty[~valid_mask] = 0.0 
                queried_certainty = torch.sum(certainty * weight_vector, dim=1)
        
        weight_vector = weight_vector.unsqueeze(-1)  # [N, K, 1]

        if self.config.weighted_first:
            if query_geo_feature:
                geo_features_vector = torch.sum(geo_features_vector * weight_vector, dim=1) # [N, F+P]

            if query_color_feature and self.color_features is not None:
                color_features_vector = torch.sum(color_features_vector * weight_vector, dim=1) # [N, F+P]

        # T3 = get_time()

        # in ms
        # print("time for nn     :", (T1-T0) * 1e3) # ////
        # print("time for sorting:", (T2-T1) * 1e3) # //
        # print("time for feature:", (T3-T2) * 1e3) # ///

        return geo_features_vector, color_features_vector, weight_vector, nn_counts, queried_certainty

    # prune inactive uncertain neural points
    def prune_map(self, prune_certainty_thre):
        
        diff_travel_dist = torch.abs(self.travel_dist[self.cur_ts] - self.travel_dist[self.point_ts_update])
        inactive_mask = diff_travel_dist > self.diff_travel_dist_local

        prune_mask = inactive_mask & (self.point_certainties < prune_certainty_thre) # True for prune
        
        prune_count = torch.sum(prune_mask).item()
        if prune_count > 100:
            if not self.silence:
                print("# Prune neural points: ", prune_count)

            self.neural_points = self.neural_points[~prune_mask]
            self.point_orientations = self.point_orientations[~prune_mask]
            self.point_ts_create = self.point_ts_create[~prune_mask]
            self.point_ts_update = self.point_ts_update[~prune_mask]
            self.point_certainties = self.point_certainties[~prune_mask]

            # with padding
            prune_mask = torch.cat((prune_mask, torch.tensor([False]).to(prune_mask)), dim=0)
            self.geo_features = self.geo_features[~prune_mask]
            if self.config.color_on:
                self.color_features = self.color_features[~prune_mask]
            # recreate hash and local map then
            return True
        return False
            
    def adjust_map(self, pose_diff_torch):
        # for each neural point, use it ts to find the diff between old and new pose, transform the position and rotate the orientation
        # we use the mid_ts for each neural point

        self.after_pgo = True

        if self.config.use_mid_ts:
            used_ts = (self.point_ts_create + self.point_ts_update) / 2 # still as dtype long
        else:
            used_ts = self.point_ts_create 
        
        self.neural_points = transform_batch_torch(self.neural_points, pose_diff_torch[used_ts])
        
        diff_quat_torch = rotmat_to_quat(pose_diff_torch[:,:3,:3]) # rotation part
        
        self.point_orientations = quat_multiply(diff_quat_torch[used_ts], self.point_orientations) # no problem here

    def recreate_hash(self, sensor_position: torch.Tensor, sensor_orientation: torch.Tensor, 
                      kept_points: bool = True, with_ts: bool = True, cur_ts = 0):

        cur_resolution = self.resolution

        self.buffer_pt_index = torch.full((self.buffer_size,), -1, dtype=self.idx_dtype, device=self.device) # reset
        
        # take the point that is closer to the current timestamp (now used)
        # also update the timestep of neural points during merging
        if with_ts:
            if self.config.use_mid_ts:
                ts_used = (self.point_ts_create + self.point_ts_update) / 2 # still as dtype long
            else:
                ts_used = self.point_ts_create
            ts_diff = torch.abs(ts_used - cur_ts).float()
            sample_idx = voxel_down_sample_min_value_torch(self.neural_points, cur_resolution, ts_diff) 
        else:
        # take the point that has a larger certainity
            sample_idx = voxel_down_sample_min_value_torch(self.neural_points, cur_resolution, self.point_certainties.max()-self.point_certainties) 

        original_point_count = self.neural_points.shape[0]
        kept_point_count = sample_idx.shape[0]

        if kept_points:
            # don't filter the neural points (keep them, only merge when neccessary, figure out the better merging method later)
            sample_points = self.neural_points[sample_idx]
            grid_coords = (sample_points / cur_resolution).floor().to(self.primes) 
            hash = torch.fmod((grid_coords * self.primes).sum(-1), int(self.buffer_size))        
            self.buffer_pt_index[hash] = sample_idx

        else:
            if not self.silence:
                print("Filter duplicated neural points")

            # only kept those filtered
            self.neural_points = self.neural_points[sample_idx]
            self.point_orientations = self.point_orientations[sample_idx] # as quaternion
            self.point_ts_create = self.point_ts_create[sample_idx]
            self.point_ts_update = self.point_ts_update[sample_idx]
            self.point_certainties = self.point_certainties[sample_idx]

            sample_idx_pad = torch.cat((sample_idx, torch.tensor([-1]).to(sample_idx)))
            self.geo_features = self.geo_features[sample_idx_pad] # with padding in the end
            if self.color_features is not None:
                self.color_features = self.color_features[sample_idx_pad] # with padding in the end

            new_point_count = self.neural_points.shape[0]

            grid_coords = (self.neural_points / cur_resolution).floor().to(self.primes) 
            hash = torch.fmod((grid_coords * self.primes).sum(-1), int(self.buffer_size))        
            self.buffer_pt_index[hash] = torch.arange(new_point_count, dtype=self.idx_dtype, device=self.device)

        if sensor_position is not None:
            self.reset_local_map(sensor_position, sensor_orientation, cur_ts)

        if not kept_points: # merged
            self.print_memory() # show the updated memory after merging

    def set_search_neighborhood(self, num_nei_cells: int = 1, search_alpha: float = 1.0):

        dx = torch.arange(-num_nei_cells, num_nei_cells+1, 
                          device=self.primes.device, dtype=self.primes.dtype)

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3) # [K,3]

        dx2 = torch.sum(dx**2, dim=-1)
        self.neighbor_dx = dx[dx2<(num_nei_cells+search_alpha)**2] # in the sphere --> smaller K --> faster training
        
        # when num_cells = 3
        # alpha 0.2, K = 147
        # alpha 0.5, K = 179
        # alpha 1.0, K = 251
        
        # when num_cells = 2
        # alpha 0.2, K = 33
        # alpha 0.3, K = 57 
        # alpha 0.5, K = 81
        # alpha 1.0, K = 93
        # alpha 2.0, K = 125

        self.neighbor_K = self.neighbor_dx.shape[0]
        self.max_valid_dist2 = 3*((num_nei_cells+1)*self.resolution)**2
        # print(self.neighbor_K)


    def radius_neighborhood_search(self, points: torch.Tensor, time_filtering: bool = False):
                                        
        # T0 = get_time()
        cur_resolution = self.resolution
        cur_buffer_size = int(self.buffer_size)

        grid_coords = (points / cur_resolution).floor().to(self.primes) # [N,3]
        
        neighbord_cells = grid_coords[..., None, :] + self.neighbor_dx # [N,K,3] # int64

        # T1 = get_time()
    
        # hash = (neighbord_cells * self.primes).sum(-1) % cur_buffer_size  # [N,K] # no negative number
        hash = torch.fmod((neighbord_cells * self.primes).sum(-1), cur_buffer_size) # [N,K] # with negative number (but actually the same)

        # T12 = get_time()

        neighb_idx = self.buffer_pt_index[hash]

        # T2 = get_time()

        if time_filtering: # now is actually travel distance filtering
            diff_travel_dist = torch.abs(self.travel_dist[self.cur_ts] - self.travel_dist[self.point_ts_create[neighb_idx]])
            local_t_window_mask = diff_travel_dist < self.diff_travel_dist_local
            neighb_idx[~local_t_window_mask] = -1 

        # T3 = get_time()

        neighb_pts = self.neural_points[neighb_idx]
        neighb_pts_sub = neighb_pts - points.view(-1, 1, 3) # [N,K,3]

        dist2 = torch.sum(neighb_pts_sub**2, dim=-1)
        dist2[neighb_idx==-1]=self.max_valid_dist2

        # if the dist is too large (indicating a hash collision), also mask the index as invalid
        neighb_idx[dist2>self.max_valid_dist2]=-1

        # T4 = get_time()

        # print("time for get neighbor idx:", (T1-T0) * 1e3)  # |
        # # print("time for hashing func    :", (T12-T1) * 1e3)
        # print("time for hashing         :", (T2-T1) * 1e3)  # ||||
        # print("time for time filtering  :", (T3-T2) * 1e3)  # |
        # print("time for distance        :", (T4-T3) * 1e3)  # |||

        return dist2, neighb_idx
    
    def query_certainty(self, query_points: torch.Tensor): # a faster way to get the certainty at a batch of query points
        
        _, idx = self.radius_neighborhood_search(query_points) # only the self voxel
                                                 
        # idx = self.global2local[0][idx] # [N, K] # get the local idx using the global2local mapping
        # certainty = self.local_hier_certainty[0][idx] # [N, K] # directly global search

        certainty = self.point_certainties[idx]
        certainty[idx < 0] = 0.

        query_points_certainty = torch.max(certainty, dim=-1)[0] 

        # print(query_points_certainty)

        return query_points_certainty
    
    # clear the temp data that is not needed
    def clear_temp(self, clean_more: bool = False):
        self.buffer_pt_index = None
        self.local_neural_points = None
        self.local_point_orientations = None
        self.local_geo_features = nn.Parameter()
        self.local_color_features = nn.Parameter() 
        self.local_point_certainties = None
        self.local_point_ts_update = None
        self.local_mask = None
        self.global2local = None

        # Also only used for debugging, can be removed
        if clean_more:
            self.point_ts_create = None
            self.point_ts_update = None
            self.point_certainties = None 
    
    def get_map_o3d_bbx(self):
        map_min, _ = torch.min(self.neural_points, dim=0)
        map_max, _ = torch.max(self.neural_points, dim=0)

        # print(map_min)

        o3d_bbx = o3d.geometry.AxisAlignedBoundingBox(map_min.cpu().detach().numpy(), map_max.cpu().detach().numpy())

        return o3d_bbx
    
    # def feature_tsne(self):
    #     tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
    #     tsne_result = tsne.fit_transform(self.geo_features[:-1].cpu().detach().numpy())
    
# Borrow from Louis's LocNDF # but the positional encoding is actually not used
class PositionalEncoder(nn.Module):
    # out_dim = in_dimnesionality * (2 * bands + 1)
    def __init__(self, config: Config):
        super().__init__()

        self.freq = torch.tensor(config.pos_encoding_freq)
        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim
        self.base = torch.tensor(config.pos_encoding_base)

        self.out_dim = self.dimensionality * (2 * self.num_bands + 1)

        # self.num_bands = floor(feature_size/dimensionality/2)

    def forward(self, x):
        
        # print(x)
        x = x[..., :self.dimensionality, None]
        device, dtype, orig_x = x.device, x.dtype, x

        scales = torch.logspace(0., torch.log(
            self.freq / 2) / torch.log(self.base), self.num_bands, base=self.base, device=device, dtype=dtype)
        # Fancy reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * torch.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim=-1)
        x = x.flatten(-2, -1)
        # print(x.shape)

        # print(x)

        return x

    def featureSize(self):
        return self.out_dim

# Borrow from Louis's Loc_NDF
class GaussianFourierFeatures(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.freq = torch.tensor(config.pos_encoding_freq)
        self.num_bands = config.pos_encoding_band
        self.dimensionality = config.pos_input_dim

        self.register_buffer('B', torch.randn(
            [self.dimensionality, self.num_bands])*self.freq)

        self.out_dim = self.num_bands*2 + self.dimensionality

    def forward(self, x):
        x_proj = (2.*torch.pi*x) @ self.B
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

    def featureSize(self):
        return self.out_dim
