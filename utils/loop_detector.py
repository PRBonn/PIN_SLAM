#!/usr/bin/env python3
# @file      loop_detector.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import numpy as np
import torch
import torch.nn.functional as F
import math
from rich import print

from utils.config import Config
from utils.mapper import Mapper
from utils.tools import get_time, transform_torch

class NeuralPointMapContextManager:
    def __init__(self, config: Config, mapper: Mapper):
        
        self.mapper = mapper
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.tran_dtype = config.tran_dtype
        self.silence = config.silence

        self.des_shape = config.context_shape
        self.num_candidates = config.context_num_candidates # 1
        self.ringkey_dist_thre = (config.max_z - config.min_z) * 0.25 # m

        self.sc_cosdist_threshold = config.context_cosdist_threshold
        if config.local_map_context:
            self.sc_cosdist_threshold += 0.08
            if config.loop_with_feature:
                self.sc_cosdist_threshold += 0.08
                self.ringkey_dist_thre = 0.25 # use cos distance
                
        self.max_length = config.npmc_max_dist            
            
        self.ENOUGH_LARGE = config.end_frame+1 # capable of up to ENOUGH_LARGE number of nodes 

        self.contexts = [None] * self.ENOUGH_LARGE
        self.ringkeys = [None] * self.ENOUGH_LARGE
        self.contexts_feature = [None] * self.ENOUGH_LARGE
        self.ringkeys_feature = [None] * self.ENOUGH_LARGE

        self.query_contexts = []
        self.tran_from_frame = []
        self.curr_node_idx = 0 

        # needs to cover 10 m
        self.virtual_step_m = config.voxel_size_m * 4.0 # * 5.0
        self.virtual_side_count = config.context_virtual_side_count # 5
        self.virtual_sdf_thre = 0.0
       
    # fast implementation of scan context by torch
    def add_node(self, frame_id, ptcloud, ptfeatures = None):
        
        # ptcloud as torch tensor
        sc, sc_feature = ptcloud2sc_torch(ptcloud, ptfeatures, self.des_shape, self.max_length) # RxS # keep the highest point's height in each bin
        rk = sc2rk(sc) # Rx1 # take the mean for all the sectors of each ring, get r ring circles (to keep the rotation invariance)

        # print("Generate descriptor")
        self.curr_node_idx = frame_id
        self.contexts[frame_id] = sc
        self.ringkeys[frame_id] = rk

        if sc_feature is not None:
            rk_feature = sc2rk(sc_feature)
            self.contexts_feature[frame_id] = sc_feature
            self.ringkeys_feature[frame_id] = rk_feature

        self.query_contexts = []
        self.tran_from_frame = []

    # use virtual node to deal with translation
    def set_virtual_node(self, ptcloud_global, frame_pose, last_frame_pose, ptfeatures = None):

        if last_frame_pose is not None:
            tran_direction = frame_pose[:3,3] - last_frame_pose[:3,3] 
            tran_direction_norm = torch.norm(tran_direction)
            tran_direction_unit = tran_direction / tran_direction_norm
            lat_rot = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device = self.device, dtype=self.dtype)
            lat_direction_unit =  lat_rot @ tran_direction_unit.float() # 3, 1
        else:
            lat_direction_unit = torch.tensor([0, 1, 0], device = self.device, dtype=self.dtype)

        dx = torch.arange(-self.virtual_side_count, self.virtual_side_count+1, device = self.device)*self.virtual_step_m

        lat_tran = dx.view(-1, 1) @ lat_direction_unit.view(1, 3) # N, 3

        virtual_positions = frame_pose[:3,3].float() + lat_tran # N, 3

        # filter the virtual poses using the sdf (negatives are not used)
        # sdf_at_virtual_poses, _ = self.mapper.sdf(virtual_positions)
        # sdf_at_virtual_poses = sdf_at_virtual_poses.detach()
        # sdf_valid_mask = sdf_at_virtual_poses > self.virtual_sdf_thre
        # virtual_positions = virtual_positions[sdf_valid_mask]
        # lat_tran = lat_tran[sdf_valid_mask]

        virtual_pose_count = virtual_positions.shape[0] # M

        if not self.silence:
            print("# Augmented virtual context: ", virtual_pose_count) # can be either local map or scan context

        # encode context for true mask here
        for idx in range(virtual_pose_count):
            
            cur_lat_tran = lat_tran[idx]
            cur_tran_from_frame = torch.eye(4, device = self.device, dtype=torch.float64)
            cur_tran_from_frame[:3,3] = cur_lat_tran

            cur_virtual_pose =  frame_pose @ torch.linalg.inv(cur_tran_from_frame) # T_w<-c' = T_w<-c @ T_c<-c'

            if torch.norm(cur_lat_tran) == 0: # exact pose of this frame
                if ptfeatures is None:
                    cur_sc = self.contexts[self.curr_node_idx]
                else:
                    cur_sc_feature = self.contexts_feature[self.curr_node_idx]
            else:
                ptcloud = transform_torch(ptcloud_global, torch.linalg.inv(cur_virtual_pose))
                cur_sc, cur_sc_feature = ptcloud2sc_torch(ptcloud, ptfeatures, self.des_shape, self.max_length)

            if ptfeatures is None:
                self.query_contexts.append(cur_sc)
            else:
                self.query_contexts.append(cur_sc_feature)
            self.tran_from_frame.append(cur_tran_from_frame)

    # main function for global loop detection
    def detect_global_loop(self, cur_pgo_poses, pgo_poses, dist_thre, loop_candidate_mask, neural_points): 

        dist_to_past = np.linalg.norm(cur_pgo_poses[:,:3,3] - cur_pgo_poses[self.curr_node_idx,:3,3], axis=1)
        dist_search_mask = dist_to_past < dist_thre
        global_loop_candidate_idx = np.where(loop_candidate_mask & dist_search_mask)[0]
        if global_loop_candidate_idx.shape[0] > 0: # candidate exist
            context_pc = neural_points.local_neural_points.detach() # augment virtual poses
            cur_pose = torch.tensor(pgo_poses[self.curr_node_idx], device=self.device, dtype=torch.float64)            
            last_pose = torch.tensor(pgo_poses[self.curr_node_idx-1], device=self.device, dtype=torch.float64) if self.curr_node_idx > 0 else None
            neural_points_feature = neural_points.local_geo_features[:-1].detach() if self.config.loop_with_feature else None
            self.set_virtual_node(context_pc, cur_pose, last_pose, neural_points_feature)
        loop_id, loop_cos_dist, loop_transform = self.detect_loop(global_loop_candidate_idx, use_feature=self.config.loop_with_feature)
        
        local_map_context_loop = False
        if loop_id is not None:
            if self.config.local_map_context: # with the latency
                loop_transform = loop_transform @ np.linalg.inv(cur_pgo_poses[self.curr_node_idx]) @ cur_pgo_poses[-1] # T_l<-c' = T_l<-c @ T_c<-c' = T_l<-c @ T_c<-w @ T_w<-c' 
                local_map_context_loop = True
            if not self.silence:
                print("[bold red]Candidate global loop event detected: [/bold red]", self.curr_node_idx, "---", loop_id, "(" , loop_cos_dist, ")")
            # print("[bold red]Candidate global loop event detected: [/bold red]", self.curr_node_idx, "---", loop_id, "(" , loop_cos_dist, ")")
            
        return loop_id, loop_cos_dist, loop_transform, local_map_context_loop


    def detect_loop(self, candidate_idx, use_feature: bool = False):        
        
        # t1 = get_time()

        if candidate_idx.shape[0] == 0:
            return None, None, None

        if use_feature:
            ringkey_feature_history = torch.stack([self.ringkeys_feature[i] for i in candidate_idx])
            history_count = ringkey_feature_history.shape[0]
        else:
            ringkey_history = torch.stack([self.ringkeys[i] for i in candidate_idx])

        # t2 = get_time()

        min_dist_ringkey = 1e5
        min_loop_idx = None
        min_query_idx = 0

        if len(self.query_contexts) == 0:
            self.tran_from_frame.append(torch.eye(4, device = self.device, dtype=torch.float64))
            if use_feature:
                self.query_contexts.append(self.contexts_feature[self.curr_node_idx])
            else:
                self.query_contexts.append(self.contexts[self.curr_node_idx])

        for query_idx in range(len(self.query_contexts)):
            if use_feature:
                query_context_fearure = self.query_contexts[query_idx] # R,K,D
                query_ringkey_feature = sc2rk(query_context_fearure) # R,D
                dist_to_history = 1.0 - F.cosine_similarity(query_ringkey_feature.view(1,-1), ringkey_feature_history.view(history_count, -1), dim=1) # cosine similarity # RxD dim
                # print(dist_to_history)
            else:    
                query_context = self.query_contexts[query_idx]
                query_ringkey = sc2rk(query_context)
                diff_to_history = query_ringkey - ringkey_history # brute force nn 
                dist_to_history = torch.norm(diff_to_history, p=1, dim=1) # l1 norm

            min_idx = torch.argmin(dist_to_history)
            cur_min_idx_in_candidates = candidate_idx[min_idx].item()
            cur_dist_ringkey = dist_to_history[min_idx].item()

            # print(cur_min_idx_in_candidates)
            # print(cur_dist_ringkey)
            
            if cur_dist_ringkey < min_dist_ringkey:
                min_dist_ringkey = cur_dist_ringkey
                min_loop_idx = cur_min_idx_in_candidates
                min_query_idx = query_idx
        
        if not self.silence:
            print("min ringkey dist:", min_dist_ringkey)

        if min_dist_ringkey > self.ringkey_dist_thre:
            return None, None, None

        # t3 = get_time()

        if use_feature:
            query_sc_feature = self.query_contexts[min_query_idx]
            candidate_sc_feature = self.contexts_feature[min_loop_idx]
            cosdist, yaw_diff = distance_sc_feature_torch(candidate_sc_feature, query_sc_feature)
        else:
            query_sc = self.query_contexts[min_query_idx] 
            candidate_sc = self.contexts[min_loop_idx]
            cosdist, yaw_diff = distance_sc_torch(candidate_sc, query_sc)
            # use aligning key to get the yaw_diff, to further speed up the process

        if not self.silence:
            print("min context cos dist:", cosdist)

        # t4 = get_time()

        # print("stack time  :", (t2-t1) * 1e3)
        # print("rk dist time:", (t3-t2) * 1e3)
        # print("sc dist time:", (t4-t3) * 1e3)

        # find the best match (sector shifted) scan context in the candidates
        # get the yaw angle and the match frame idx at the same time
        if (cosdist < self.sc_cosdist_threshold): # the smaller the sc_cosdist_threshold, the more strict
            yawdiff_deg = yaw_diff * (360.0/self.des_shape[1])

            if not self.silence:
                print("yaw diff deg:", yawdiff_deg)

            yawdiff_rad = math.radians(yawdiff_deg)
            cos_yaw = math.cos(yawdiff_rad)
            sin_yaw = math.sin(yawdiff_rad)
            transformation = np.eye(4)
            transformation[0,0]=cos_yaw
            transformation[0,1]=sin_yaw
            transformation[1,0]=-sin_yaw
            transformation[1,1]=cos_yaw # T_l<-c'

            transformation = transformation @ (self.tran_from_frame[min_query_idx].detach().cpu().numpy())
            # T_l<-c = T_l<-c' @ T_c'<-c

            return min_loop_idx, cosdist, transformation  
            # loop detected!, transformation in numpy (should be  T_l<-c)
        else:
            return None, None, None

class GTLoopManager:
    def __init__(self, config: Config):
        self.max_loop_dist = config.max_loop_dist
        self.min_travel_dist_ratio = 2.5
        
        self.ENOUGH_LARGE = config.end_frame+1
        self.gt_position = [None] * self.ENOUGH_LARGE
        self.gt_pose = [None] * self.ENOUGH_LARGE
        self.travel_dist = [0.] * self.ENOUGH_LARGE

        self.min_loop_idx = self.ENOUGH_LARGE

        self.curr_node_idx = 0
       
    def add_node(self, node_idx: int, gt_pose: np.array):
        # print("LOOP --- ", node_idx)
        self.curr_node_idx = node_idx
        self.gt_position[node_idx] = gt_pose[:3,3]
        self.gt_pose[node_idx] = gt_pose
        if node_idx > 0:
            travel_dist_in_frame = np.linalg.norm(self.gt_position[node_idx] - self.gt_position[node_idx-1])
            self.travel_dist[node_idx] = self.travel_dist[node_idx-1] + travel_dist_in_frame

    def detect_loop(self):      

        exclude_recent_nodes = 30
        valid_recent_node_idx = self.curr_node_idx - exclude_recent_nodes

        if valid_recent_node_idx > 0:
            dist_to_past = np.linalg.norm(self.gt_position[self.curr_node_idx] - np.array(self.gt_position[:valid_recent_node_idx]), axis=1)
            # print(dist_to_past)
            travel_dist_to_past = self.travel_dist[self.curr_node_idx] - np.array(self.travel_dist[:valid_recent_node_idx])
            
            # 0 to valid_recent_node_idx
            candidate_mask = (travel_dist_to_past > self.min_travel_dist_ratio * dist_to_past) & (travel_dist_to_past > 30.0) 

            candidate_idx = np.where(candidate_mask)[0]
            candidate_dist = dist_to_past[candidate_mask]

            if np.shape(candidate_dist)[0] > 0:
                min_index_in_cand = np.argmin(candidate_dist)
                loop_dist = candidate_dist[min_index_in_cand]
                loop_index = candidate_idx[min_index_in_cand]

                if loop_dist < self.max_loop_dist:
                    # T_l<-c = T_l<-w @ T_w<-c
                    loop_trans = np.linalg.inv(self.gt_pose[loop_index]) @ self.gt_pose[self.curr_node_idx] 
                    return loop_index, loop_dist, loop_trans
            
        return None, None, None

def detect_local_loop(dist_to_past, loop_candidate_mask, pgo_poses, cur_drift, cur_frame_id, loop_reg_failed_count=0, dist_thre=1.0, silence=False):
    min_dist = np.min(dist_to_past[loop_candidate_mask])
    min_index = np.where(dist_to_past == min_dist)[0]
    if min_dist < dist_thre and cur_drift < dist_thre*2 and loop_reg_failed_count < 3: # local loop
        loop_id, loop_dist = min_index[0], min_dist # a candidate found
        loop_transform = np.linalg.inv(pgo_poses[loop_id]) @ pgo_poses[-1] 
        if not silence:
            print("[bold red]Candidate local loop event detected: [/bold red]", cur_frame_id, "---", loop_id, "(" , loop_dist, ")")
        return loop_id, loop_dist, loop_transform
    else: 
        return None, None, None


def ptcloud2sc_torch(ptcloud, pt_feature, sc_shape, max_length):
    # pt_cloud in torch

    # filter pt_cloud (x,y) with max_length
    # Calculate the radius (r) using the Euclidean distance formula
    r = torch.norm(ptcloud, dim=1)

    kept_mask = (r < max_length)
    
    points = ptcloud[kept_mask]
    r = r[kept_mask]
        
    num_ring = sc_shape[0] # ring num (radial) # 20
    num_sector = sc_shape[1] # 60

    gap_ring = max_length/num_ring # radial
    gap_sector = 360.0/num_sector # yaw angle

    sc = torch.zeros(num_ring * num_sector, dtype=points.dtype, device=points.device)
    sc_feature = None

    if pt_feature is not None:
        pt_feature_kept = (pt_feature.clone())[kept_mask]
        sc_feature = torch.zeros(num_ring * num_sector, pt_feature.shape[1], dtype=points.dtype, device=points.device)
    
    # Calculate the angle (θ) using the arctan2 function
    theta = torch.atan2(points[:, 1], points[:, 0]) # [-pi, pi] # rad

    # Convert the angle from radians to degrees if needed
    theta_degrees = theta * 180.0 / math.pi + 180.0 # [0, 360]

    # we have the ring, sector coordinate for each point
    idx_ring = torch.clamp((r // gap_ring).long(), 0, num_ring-1)      
    idx_sector = torch.clamp((theta_degrees // gap_sector).long(), 0, num_sector-1)

    grid_indices = idx_ring * num_sector + idx_sector

    sc = sc.scatter_reduce_(dim=0, index=grid_indices, src=points[:, 2], reduce="amax", include_self=False) # record the max value of z value
    # scatter_reduce_, This operation may behave nondeterministically when given tensors on a CUDA device
    # be careful
    sc = sc.view(num_ring, num_sector) # R, S
    
    if pt_feature is not None:
        grid_indices = grid_indices.view(-1, 1).repeat(1, pt_feature.shape[1])
        sc_feature = sc_feature.scatter_reduce_(dim=0, index=grid_indices, src=pt_feature_kept, reduce="mean", include_self=False)
        sc_feature = sc_feature.view(num_ring, num_sector, pt_feature.shape[1]) # R, S, D
        # print(sc_feature) 
        
    return sc, sc_feature

def sc2rk(sc):
    return torch.mean(sc, dim=1)

# the (consine) distance between two sc (input as torch tensors)
def distance_sc_torch(sc1, sc2): # RxS
    num_sectors = sc1.shape[1]

    # repeate to move 1 columns
    _one_step = 1 # const
    sim_for_each_cols = torch.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = torch.roll(sc1, _one_step, 1) # columne shift (one column sector each time)

        # each sector's cosine similarity
        cossim = F.cosine_similarity(sc1, sc2, dim=0)

        sim_for_each_cols[i] = torch.mean(cossim) # average cosine similarity

    # rotate (shift sector) to find the best match yaw_shift and the similarity
    yaw_diff = torch.argmax(sim_for_each_cols) + 1 # starts with 0 
    sim = torch.max(sim_for_each_cols)
    
    dist = 1 - sim

    return dist.item(), yaw_diff.item()

# the distance between two sc (input as torch tensors)
def distance_sc_feature_torch(sc1, sc2): # RxSxD
    
    num_rings = sc1.shape[0]
    num_sectors = sc1.shape[1]

    # repeate to move 1 columns
    _one_step = 1 # const
    sim_for_each_cols = torch.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = torch.roll(sc1, _one_step, 1) # columne shift (one column sector each time)

        # each sector's cosine similarity
        cossim = F.cosine_similarity(sc1.view(num_rings, -1), sc2.view(num_rings, -1), dim=0)

        sim_for_each_cols[i] = torch.mean(cossim) # average cosine similarity

    # rotate (shift sector) to find the best match yaw_shift and the similarity
    yaw_diff = torch.argmax(sim_for_each_cols) + 1 # starts with 0 
    sim = torch.max(sim_for_each_cols)
    dist = 1 - sim

    return dist.item(), yaw_diff.item()