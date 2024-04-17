#!/usr/bin/env python3
# @file      tools.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

from typing import List
import sys
import os
import random
import multiprocessing
import getpass
import time
from pathlib import Path
from datetime import datetime
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.autograd import grad
import math
import roma
import numpy as np
import wandb
import json
import open3d as o3d
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
# 'plasma', 'inferno', 'magma', Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',  'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
                  
from utils.config import Config

# setup this run
def setup_experiment(config: Config, argv = None, debug_mode: bool = False): 

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    
    run_name = config.name + "_" + ts  # modified to a name that is easier to index
        
    run_path = os.path.join(config.output_root, run_name)

    cuda_available = torch.cuda.is_available()
    if not cuda_available: 
        print('No CUDA device available, use CPU instead')
        config.device = 'cpu'

    if not debug_mode:
        access = 0o755
        os.makedirs(run_path, access, exist_ok=True)
        assert os.access(run_path, os.W_OK)
        if not config.silence:
            print(f"Start {run_path}")

        config.run_path = run_path

        mesh_path = os.path.join(run_path, "mesh")
        map_path = os.path.join(run_path, "map")
        model_path = os.path.join(run_path, "model")
        log_path = os.path.join(run_path, "log")
        os.makedirs(mesh_path, access, exist_ok=True)
        os.makedirs(map_path, access, exist_ok=True)
        os.makedirs(model_path, access, exist_ok=True)
        os.makedirs(log_path, access, exist_ok=True)
        
        if config.wandb_vis_on:
            # set up wandb
            setup_wandb()
            wandb.init(project="PIN_SLAM", config=vars(config), dir=run_path) # your own worksapce
            wandb.run.name = run_name         

        # config file and reproducable shell script
        if argv is not None:
            shutil.copy2(argv[1], run_path) # copy the config file to the result folder

            git_commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip() # current git commit
            with open(os.path.join(run_path,'run.sh'), 'w') as reproduce_shell:
                reproduce_shell.write(' '.join(["git checkout ", git_commit_id, "\n"]))
                run_str = "python3 "+ ' '.join(argv)
                reproduce_shell.write(run_str)
    
    # set the random seed for all
    os.environ['PYTHONHASHSEED']=str(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed) 
    o3d.utility.random.seed(config.seed)

    torch.set_default_dtype(config.dtype)

    return run_path

def setup_optimizer(config: Config, neural_point_feat, mlp_geo_param = None, 
                    mlp_sem_param = None, mlp_color_param = None, poses = None, lr_ratio = 1.0) -> Optimizer:
    lr_cur = config.lr * lr_ratio
    lr_pose = config.lr_pose
    weight_decay = config.weight_decay
    weight_decay_mlp = 0.0
    opt_setting = []
    # weight_decay is for L2 regularization
    if mlp_geo_param is not None: 
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': weight_decay_mlp} 
        opt_setting.append(mlp_geo_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {'params': mlp_sem_param, 'lr': lr_cur, 'weight_decay': weight_decay_mlp} 
        opt_setting.append(mlp_sem_param_opt_dict)
    if config.color_on and mlp_color_param is not None:
        mlp_color_param_opt_dict = {'params': mlp_color_param, 'lr': lr_cur, 'weight_decay': weight_decay_mlp} 
        opt_setting.append(mlp_color_param_opt_dict)
    if poses is not None:
        poses_opt_dict = {'params': poses, 'lr': lr_pose, 'weight_decay': weight_decay}
        opt_setting.append(poses_opt_dict)
    feat_opt_dict = {'params': neural_point_feat, 'lr': lr_cur, 'weight_decay': weight_decay} 
    opt_setting.append(feat_opt_dict)
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt    

# set up weight and bias
def setup_wandb():
    print("Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file.")
    username = getpass.getuser()
    # print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):"
        )
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')

def step_lr_decay(
    optimizer: Optimizer,
    learning_rate: float,
    iteration_number: int,
    steps: List,
    reduce: float = 1.0):

    if reduce > 1.0 or reduce <= 0.0:
        sys.exit(
            "The decay reta should be between 0 and 1."
        )

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate

# calculate the analytical gradient by pytorch auto diff
def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad

def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True


def freeze_decoders(geo_decoder, sem_decoder, color_decoder, config):
    if not config.silence:
        print("Freeze the decoder")
    freeze_model(geo_decoder) # fixed the geo decoder
    if config.semantic_on:
        freeze_model(sem_decoder) # fixed the sem decoder
    if config.color_on:
        freeze_model(color_decoder) # fixed the color decoder

def save_checkpoint(
    neural_points, geo_decoder, color_decoder, sem_decoder, optimizer, run_path, checkpoint_name, iters
):
    torch.save(
        {
            "iters": iters,
            "neural_points": neural_points, # save the whole NN module
            "geo_decoder": geo_decoder.state_dict(),
            "color_decoder": color_decoder.state_dict(),
            "sem_decoder": sem_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_path, f"{checkpoint_name}.pth"),
    )
    print(f"save the model to {run_path}/{checkpoint_name}.pth")

def save_implicit_map(run_path, neural_points, geo_decoder, color_decoder=None, sem_decoder=None):
    
    map_dict = {"neural_points": neural_points, "geo_decoder": geo_decoder.state_dict()}
    if color_decoder is not None:
        map_dict["color_decoder"] = color_decoder.state_dict()
    if sem_decoder is not None:
        map_dict["sem_decoder"] = sem_decoder.state_dict()

    model_save_path = os.path.join(run_path, "model", "pin_map.pth") # end with .pth
    torch.save(map_dict, model_save_path) 

    print(f"save the map to {model_save_path}")

    np.save(os.path.join(run_path, "memory_footprint.npy"), 
            np.array(neural_points.memory_footprint)) # save detailed memory table

def load_decoder(config, geo_mlp, sem_mlp, color_mlp):
    loaded_model = torch.load(config.model_path)
    geo_mlp.load_state_dict(loaded_model["geo_decoder"])
    print("Pretrained decoder loaded")
    freeze_model(geo_mlp) # fixed the decoder
    if config.semantic_on:
        sem_mlp.load_state_dict(loaded_model["sem_decoder"])
        freeze_model(sem_mlp) # fixed the decoder
    if config.color_on:
        color_mlp.load_state_dict(loaded_model["color_decoder"])
        freeze_model(color_mlp) # fixed the decoder

def get_time():
    """
    :return: get timing statistics
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available: # issue #10
        torch.cuda.synchronize()
    return time.time()

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.
    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.
    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)

def color_to_intensity(colors: torch.tensor):

    intensity = 0.144 * colors[:, 0] + 0.299 * colors[:, 1] + 0.587 * colors[:, 2]

    return intensity.unsqueeze(1)

def create_axis_aligned_bounding_box(center, size):
    # Calculate the min and max coordinates based on the center and size
    min_coords = center - (size / 2)
    max_coords = center + (size / 2)

    # Create an Open3D axis-aligned bounding box
    bounding_box = o3d.geometry.OrientedBoundingBox()
    bounding_box.center = center
    bounding_box.R = np.identity(3)  # Identity rotation matrix for axis-aligned box
    # bounding_box.extent = (max_coords - min_coords) / 2
    bounding_box.extent = (max_coords - min_coords)

    return bounding_box

def apply_quaternion_rotation(quat: torch.tensor, points: torch.tensor) -> torch.tensor:
    # apply passive rotation: coordinate system rotation w.r.t. the points
    # p' = qpq^-1
    quat_w = quat[..., 0].unsqueeze(-1)
    quat_xyz = -quat[..., 1:]
    t = 2 * torch.linalg.cross(quat_xyz, points)
    points_t = points + quat_w * t + torch.linalg.cross(quat_xyz, t)
    return points_t

# pytorch implementations
def rotmat_to_quat(rot_matrix: torch.tensor):
    """
    Convert a batch of 3x3 rotation matrices to quaternions.
    """
    qw = torch.sqrt(1.0 + rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]) / 2.0
    qx = (rot_matrix[:, 2, 1] - rot_matrix[:, 1, 2]) / (4.0 * qw)
    qy = (rot_matrix[:, 0, 2] - rot_matrix[:, 2, 0]) / (4.0 * qw)
    qz = (rot_matrix[:, 1, 0] - rot_matrix[:, 0, 1]) / (4.0 * qw)
    return torch.stack((qw, qx, qy, qz), dim=1)

def quat_to_rotmat(quaternions: torch.tensor):
    # Ensure quaternions are normalized
    quaternions /= torch.norm(quaternions, dim=1, keepdim=True)
    
    # Extract quaternion components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Calculate rotation matrix elements
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    rotation_matrix = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)
    ], dim=1).view(-1, 3, 3)
    
    return rotation_matrix

def quat_multiply(q1: torch.tensor, q2: torch.tensor):
    """
    Perform quaternion multiplication for batches.
    q' = q1 @ q2
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=1) # quaternion representing the rotation
    w2, x2, y2, z2 = torch.unbind(q2, dim=1) # quaternion to be rotated

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=1)

def torch2o3d(points_torch):
    pc_o3d = o3d.geometry.PointCloud()
    points_np = points_torch.cpu().detach().numpy().astype(np.float64)
    pc_o3d.points = o3d.utility.Vector3dVector(points_np)
    return pc_o3d

def o3d2torch(o3d, device='cpu', dtype=torch.float32 ):
    return torch.tensor(np.asarray(o3d.points), dtype=dtype, device=device)

def transform_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [4, 4]
    # Add a homogeneous coordinate to each point in the point cloud
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1)

    # Apply the transformation by matrix multiplication
    transformed_points_homo = torch.matmul(points_homo, transformation.to(points).T)

    # Remove the homogeneous coordinate from each transformed point
    transformed_points = transformed_points_homo[:, :3]

    return transformed_points

def transform_batch_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [N, 4, 4]
    # N,3,3 @ N,3,1 -> N,3,1 + N,3,1 -> N,3,1 -> N,3
    points = torch.matmul(transformation[:, :3, :3].to(points), points.unsqueeze(-1)) + transformation[:, :3, 3:].to(points)

    return points.squeeze(-1)

def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers. 
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`  

    Reference: Louis Wiesmann
    """
    _quantization = 1000 # if change to 1, then it would take the first (smallest) index lie in the voxel

    offset = torch.floor(points.min(dim=0)[0]/voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1)**0.5
    dist = (dist / dist.max() * (_quantization - 1)).long() # for speed up # [0-_quantization]

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
       
    offset = 10**len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset

    idx = torch.empty(unique.shape, dtype=inverse.dtype,
                      device=inverse.device).scatter_reduce_(dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on 
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx

def voxel_down_sample_min_value_torch(points: torch.tensor, voxel_size: float, value: torch.tensor):
    """
        voxel based downsampling. Returns the indices of the points which has the minimum value in the voxel. 
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`  
    """
    _quantization = 1000

    offset = torch.floor(points.min(dim=0)[0]/voxel_size).long()
    grid = torch.floor(points / voxel_size)
    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
       
    offset = 10**len(str(idx_d.max().item()))
    
    # not same value, taker the smaller value, same value, consider the smaller index
    value = (value / value.max() * (_quantization - 1)).long() # [0-_quantization]
    idx_d = idx_d + value * offset 

    idx = torch.empty(unique.shape, dtype=inverse.dtype,
                      device=inverse.device).scatter_reduce_(dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on 
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx

# split a large point cloud into bounding box chunks
def split_chunks(pc: o3d.geometry.PointCloud(), aabb: o3d.geometry.AxisAlignedBoundingBox(), chunk_m: float = 100.0):

    if not pc.has_points():
        return None
    
    # aabb = pc.get_axis_aligned_bounding_box()
    chunk_aabb = []
    
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound() + 1e-5 # just to gurantee there's a zero on one side
    bbx_range = max_bound-min_bound
    if  bbx_range[0] > bbx_range[1]:
        axis_split = 0
        axis_kept = 1
    else:
        axis_split = 1
        axis_kept = 0
    chunk_split = np.arange(min_bound[axis_split], max_bound[axis_split], chunk_m)

    chunk_count = 0

    for i in range(len(chunk_split)):
        cur_min_bound = np.copy(min_bound) # you need to clone, otherwise value changed
        cur_max_bound = np.copy(max_bound)
        cur_min_bound[axis_split] = chunk_split[i]
        if i < len(chunk_split)-1:
            cur_max_bound[axis_split] = chunk_split[i+1]
        
        cur_aabb = o3d.geometry.AxisAlignedBoundingBox(cur_min_bound, cur_max_bound)
        # pc_clone = copy.deepcopy(pc)

        pc_chunk = pc.crop(cur_aabb) # crop as clone, original one would not be changed
        cur_pc_aabb = pc_chunk.get_axis_aligned_bounding_box()

        chunk_min_bound = cur_pc_aabb.get_min_bound()
        chunk_max_bound = cur_pc_aabb.get_max_bound()
        chunk_range = chunk_max_bound - chunk_min_bound
        if chunk_range[axis_kept] > chunk_m * 3:
            chunk_split_2 = np.arange(chunk_min_bound[axis_kept], chunk_max_bound[axis_kept], chunk_m)
            for j in range(len(chunk_split_2)):
                cur_chunk_min_bound = np.copy(chunk_min_bound) # you need to clone, otherwise value changed
                cur_chunk_max_bound = np.copy(chunk_max_bound)
                cur_chunk_min_bound[axis_kept] = chunk_split_2[j]
                if j < len(chunk_split_2)-1:
                    cur_chunk_max_bound[axis_kept] = chunk_split_2[j+1]
                cur_aabb = o3d.geometry.AxisAlignedBoundingBox(cur_chunk_min_bound, cur_chunk_max_bound)   
                pc_chunk = pc.crop(cur_aabb) # crop as clone, original one would not be changed
                cur_pc_aabb = pc_chunk.get_axis_aligned_bounding_box()
                chunk_count += 1
                chunk_aabb.append(cur_pc_aabb)        
        else:
            chunk_count += 1
            chunk_aabb.append(cur_pc_aabb)

    # print("# Chunk for meshing: ", chunk_count)
    return chunk_aabb

# torch version of lidar undistortion (deskewing)
def deskewing(points: torch.tensor, ts: torch.tensor, pose: torch.tensor, ts_mid_pose = 0.5):

    if ts is None:
        return points # no deskewing
    
    # pose as T_last<-cur
    # ts is from 0 to 1 as the ratio
    ts = ts.squeeze(-1)

    # Normalize the tensor to the range [0, 1] 
    # NOTE: you need to figure out the begin and end of a frame because 
    # sometimes there's only partial measurements, some part are blocked by some occlussions
    min_ts = torch.min(ts)
    max_ts = torch.max(ts)
    ts = (ts - min_ts) / (max_ts - min_ts)

    ts -= ts_mid_pose

    rotmat_slerp = roma.rotmat_slerp(torch.eye(3).to(points), pose[:3,:3].to(points), ts)

    tran_lerp = ts[:, None] * pose[:3, 3].to(points) 

    points_deskewd = points
    points_deskewd[:,:3] = (rotmat_slerp @ points[:,:3].unsqueeze(-1)).squeeze(-1) + tran_lerp

    return points_deskewd

def tranmat_close_to_identity(mats: np.ndarray, rot_thre: float, tran_thre: float):

    rot_diff = np.abs(mats[:3,:3] - np.identity(3))

    rot_close_to_identity = np.all(rot_diff < rot_thre)

    tran_diff = mats[:3,3]

    tran_close_to_identity = np.all(tran_diff < tran_thre)

    if rot_close_to_identity and tran_close_to_identity:
        return True
    else:
        return False

def plot_timing_detail(time_table: np.ndarray, saving_path:str, with_loop=False):
    
    frame_count = time_table.shape[0]
    time_table_ms = time_table*1e3

    for i in range(time_table.shape[1]-1): # accumulated time
        time_table_ms[:, i+1] += time_table_ms[:, i]

    # font1 = {'family': 'Times New Roman', 'weight' : 'normal', 'size': 16}
    # font2 = {'family': 'Times New Roman', 'weight' : 'normal', 'size': 18}
    font2 = {'weight' : 'normal', 'size': 18}

    color_values = np.linspace(0, 1, 6)
    # Get the colors from the "viridis" colormap at the specified values
    # plasma, Pastel1, tab10
    colors = [viridis(x) for x in color_values]
    
    fig = plt.figure(figsize=(12.0, 4.0))

    frame_array = np.arange(frame_count)
    realtime_limit = 100.0 * np.ones([frame_count,1])
    ax1 = fig.add_subplot(111)

    line_width_1 = 0.6
    line_width_2 = 1.0
    alpha_value = 1.0
    
    ax1.fill_between(frame_array, time_table_ms[:, 0], facecolor=colors[0], edgecolor='face', where=time_table_ms[:, 0]>0, alpha=alpha_value, interpolate=True)
    ax1.fill_between(frame_array, time_table_ms[:, 0], time_table_ms[:, 1], facecolor=colors[1], edgecolor='face', where=time_table_ms[:, 1]>time_table_ms[:, 0], alpha=alpha_value, interpolate=True)
    ax1.fill_between(frame_array, time_table_ms[:, 1], time_table_ms[:, 2], facecolor=colors[2], edgecolor='face', where=time_table_ms[:, 2]>time_table_ms[:, 1], alpha=alpha_value, interpolate=True)
    ax1.fill_between(frame_array, time_table_ms[:, 2], time_table_ms[:, 3], facecolor=colors[3], edgecolor='face', where=time_table_ms[:, 3]>time_table_ms[:, 2], alpha=alpha_value, interpolate=True)
    if with_loop:
        ax1.fill_between(frame_array, time_table_ms[:, 3], time_table_ms[:, 4], facecolor=colors[4], edgecolor='face', where=time_table_ms[:, 4]>time_table_ms[:, 3], alpha=alpha_value, interpolate=True)
    
    ax1.plot(frame_array, realtime_limit, "--", linewidth=line_width_2, color = 'k')

    plt.tick_params(labelsize=12)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]

    plt.xlim((0,frame_count-1))
    plt.ylim((0,200))

    plt.xlabel('Frame ID', font2)
    plt.ylabel('Runtime (ms)', font2)
    plt.tight_layout()
    #plt.title('Timing table')

    if with_loop:
        legend = plt.legend(('Pre-processing', 'Odometry', "Mapping preparation", "Map optimization", "Loop closures"), prop=font2, loc=2)
    else:
        legend = plt.legend(('Pre-processing', 'Odometry', "Mapping preparation", "Map optimization"), prop=font2, loc=2)

    plt.savefig(saving_path, dpi=500)
    # plt.show()