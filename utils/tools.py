#!/usr/bin/env python3
# @file      tools.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import getpass
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
import yaml
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
from rich import print
import roma
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from torch import optim
from torch.autograd import grad
from torch.optim.optimizer import Optimizer

from utils.config import Config


# setup this run
def setup_experiment(config: Config, argv=None, debug_mode: bool = False):

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    warnings.filterwarnings("ignore", category=FutureWarning) 

    config.run_name = config.name + "_" + ts  # modified to a name that is easier to index
    run_path = os.path.join(config.output_root, config.run_name)

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("No CUDA device available, use CPU instead")
        config.device = "cpu"
    else:
        torch.cuda.empty_cache()
    if config.device == "cpu":
        print("Using the pure CPU mode, this would be slow")
        
    # set X service (FIXME)
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    # set the random seed for all
    seed_anything(config.seed)

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
        meta_data_path = os.path.join(run_path, "meta")
        os.makedirs(mesh_path, access, exist_ok=True)
        os.makedirs(map_path, access, exist_ok=True)
        os.makedirs(model_path, access, exist_ok=True)
        os.makedirs(log_path, access, exist_ok=True)
        os.makedirs(meta_data_path, access, exist_ok=True)

        if config.wandb_vis_on:
            # set up wandb
            setup_wandb()
            wandb.init(
                project="PIN_SLAM", config=vars(config), dir=run_path
            )  # your own worksapce
            wandb.run.name = config.run_name

        # config file and reproducable shell script
        if argv is not None:
            if len(argv) > 1 and os.path.exists(argv[1]):
                config_path = argv[1]
            else:
                config_path = "config/lidar_slam/run.yaml"
            # copy the config file to the result folder
            shutil.copy2(config_path, run_path)  

            git_commit_id = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )  # current git commit
            with open(os.path.join(run_path, "run.sh"), "w") as reproduce_shell:
                reproduce_shell.write(" ".join(["git checkout ", git_commit_id, "\n"]))
                run_str = "python3 " + " ".join(argv)
                reproduce_shell.write(run_str)


        # disable lidar deskewing when not input per frame 
        if config.step_frame > 1:
            config.deskew = False

        # write the full configs to yaml file
        config_dict = vars(config)
        config_out_path = os.path.join(meta_data_path, "config_all.yaml")
        with open(config_out_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)

    # set up dtypes, note that torch stuff cannot be write to yaml, so we set it up after write out the yaml for the whole config
    config.setup_dtype()
    torch.set_default_dtype(config.dtype)

    return run_path


def seed_anything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    o3d.utility.random.seed(seed)

def remove_gpu_cache():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()

def get_gpu_memory_usage_gb(return_cached: bool = True):
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        if return_cached:
            return torch.cuda.memory_cached() / (1024 ** 3)
        else:
            return torch.cuda.memory_allocated() / (1024 ** 3)
    else:
        return 0.0

def setup_optimizer(
    config: Config,
    neural_point_feat,
    mlp_geo_param=None,
    mlp_sem_param=None,
    mlp_color_param=None,
    poses=None,
    lr_ratio=1.0,
) -> Optimizer:
    lr_cur = config.lr * lr_ratio
    lr_pose = config.lr_pose
    weight_decay = config.weight_decay
    weight_decay_mlp = 0.0
    opt_setting = []
    # weight_decay is for L2 regularization
    if mlp_geo_param is not None:
        mlp_geo_param_opt_dict = {
            "params": mlp_geo_param,
            "lr": lr_cur,
            "weight_decay": weight_decay_mlp,
        }
        opt_setting.append(mlp_geo_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {
            "params": mlp_sem_param,
            "lr": lr_cur,
            "weight_decay": weight_decay_mlp,
        }
        opt_setting.append(mlp_sem_param_opt_dict)
    if config.color_on and mlp_color_param is not None:
        mlp_color_param_opt_dict = {
            "params": mlp_color_param,
            "lr": lr_cur,
            "weight_decay": weight_decay_mlp,
        }
        opt_setting.append(mlp_color_param_opt_dict)
    if poses is not None:
        poses_opt_dict = {"params": poses, "lr": lr_pose, "weight_decay": weight_decay}
        opt_setting.append(poses_opt_dict)
    feat_opt_dict = {
        "params": neural_point_feat,
        "lr": lr_cur,
        "weight_decay": weight_decay,
    }
    opt_setting.append(feat_opt_dict)
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9, 0.99), eps=config.adam_eps)
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)

    return opt


# set up weight and bias
def setup_wandb():
    print(
        "Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file."
    )
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
    reduce: float = 1.0,
):

    if reduce > 1.0 or reduce <= 0.0:
        sys.exit("The decay reta should be between 0 and 1.")

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate


def get_gradient(inputs, outputs):
    """
    Calculate the analytical gradient by pytorch auto diff
    """
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


def freeze_decoders(mlp_dict, config):
    if not config.silence:
        print("Freeze the decoders")
    
    keys = list(mlp_dict.keys())
    for key in keys:
        mlp = mlp_dict[key]
        if mlp is not None:
            freeze_model(mlp)

def unfreeze_decoders(mlp_dict, config):
    if not config.silence:
        print("Unfreeze the decoders")
    keys = list(mlp_dict.keys())
    for key in keys:
        mlp = mlp_dict[key]
        if mlp is not None:
            unfreeze_model(mlp)


def save_implicit_map(
    run_path, neural_points, mlp_dict, with_footprint: bool = True
):
    # together with the mlp decoders

    map_model = {"neural_points": neural_points}

    for key in list(mlp_dict.keys()):
        if mlp_dict[key] is not None:
            map_model[key] = mlp_dict[key].state_dict()
        else:
            map_model[key] = None

    model_save_path = os.path.join(run_path, "model", "pin_map.pth")  # end with .pth
    torch.save(map_model, model_save_path)

    print(f"save the map to {model_save_path}")

    if with_footprint:
        np.save(
            os.path.join(run_path, "memory_footprint.npy"),
            np.array(neural_points.memory_footprint),
        )  # save detailed memory table


def load_decoders(loaded_model, mlp_dict, freeze_decoders: bool = True):

    for key in list(loaded_model.keys()):
        if key != "neural_points":
            if loaded_model[key] is not None:
                mlp_dict[key].load_state_dict(loaded_model[key])
                if freeze_decoders:
                    freeze_model(mlp_dict[key])

    print("Pretrained decoders loaded")

def create_bbx_o3d(center, half_size):
    return o3d.geometry.AxisAlignedBoundingBox(center - half_size, center + half_size)

def get_time():
    """
    :return: get timing statistics
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:  # issue #10
        torch.cuda.synchronize()
    return time.time()

def track_progress():
    progress_bar = tqdm(desc="Processing", total=0, unit="calls")

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            wrapper.calls += 1
            progress_bar.update(1)
            progress_bar.set_description("Processing point cloud frame")
            return result
        wrapper.calls = 0
        return wrapper
    return decorator

def is_prime(n):
    """Helper function to check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_closest_prime(n):
    """Find the closest prime number to n."""
    if n < 2:
        return 2
    
    if is_prime(n):
        return n
        
    # Check numbers both above and below n
    lower = n - 1
    upper = n + 1
    
    while True:
        if is_prime(lower):
            return lower
        if is_prime(upper):
            return upper
        lower -= 1
        upper += 1


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
    intensity = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2] # thanks @zhSlamer (issue #46)
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
    bounding_box.extent = max_coords - min_coords

    return bounding_box


def apply_quaternion_rotation(quat: torch.tensor, points: torch.tensor) -> torch.tensor:
    """
    Apply passive rotation: coordinate system rotation w.r.t. the points
    p' = qpq^-1
    """
    quat_w = quat[..., 0].unsqueeze(-1)
    quat_xyz = -quat[..., 1:]
    t = 2 * torch.linalg.cross(quat_xyz, points)
    points_t = points + quat_w * t + torch.linalg.cross(quat_xyz, t)
    return points_t


# pytorch implementations
def rotmat_to_quat(rot_matrix: torch.tensor):
    """
    Convert a batch of 3x3 rotation matrices to quaternions.
    rot_matrix: N,3,3
    return N,4
    """
    qw = (
        torch.sqrt(
            1.0 + rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]
        )
        / 2.0
    )
    qx = (rot_matrix[:, 2, 1] - rot_matrix[:, 1, 2]) / (4.0 * qw)
    qy = (rot_matrix[:, 0, 2] - rot_matrix[:, 2, 0]) / (4.0 * qw)
    qz = (rot_matrix[:, 1, 0] - rot_matrix[:, 0, 1]) / (4.0 * qw)
    return torch.stack((qw, qx, qy, qz), dim=1)


def quat_to_rotmat(quaternions: torch.tensor):
    """
    Convert a batch of quaternions to rotation matrices.
    quaternions: N,4
    return N,3,3
    """
    # Ensure quaternions are normalized
    quaternions /= torch.norm(quaternions, dim=1, keepdim=True)

    # Extract quaternion components
    w, x, y, z = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )

    # Calculate rotation matrix elements
    w2, x2, y2, z2 = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation_matrix = torch.stack(
        [
            1 - 2 * (y2 + z2),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (x2 + z2),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (x2 + y2),
        ],
        dim=1,
    ).view(-1, 3, 3)

    return rotation_matrix


def quat_multiply(q1: torch.tensor, q2: torch.tensor):
    """
    Perform quaternion multiplication for batches.
    q' = q1 @ q2
    apply rotation q1 to quat q2
    both in the shape of N, 4
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=1)  # quaternion representing the rotation
    w2, x2, y2, z2 = torch.unbind(q2, dim=1)  # quaternion to be rotated

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=1) # N, 4


def torch2o3d(points_torch):
    """
    Convert a batch of points from torch to o3d
    """
    pc_o3d = o3d.geometry.PointCloud()
    points_np = points_torch.cpu().detach().numpy().astype(np.float64)
    pc_o3d.points = o3d.utility.Vector3dVector(points_np)
    return pc_o3d


def o3d2torch(o3d, device="cpu", dtype=torch.float32):
    """
    Convert a batch of points from o3d to torch
    """
    return torch.tensor(np.asarray(o3d.points), dtype=dtype, device=device)


def transform_torch(points: torch.tensor, transformation: torch.tensor):
    """
    Transform a batch of points by a transformation matrix
    Args:
        points: N,3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
        transformation: 4,4 torch tensor, the transformation matrix
    Returns:
        transformed_points: N,3 torch tensor, the transformed coordinates
    """
    # Add a homogeneous coordinate to each point in the point cloud
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1)

    # Apply the transformation by matrix multiplication
    transformed_points_homo = torch.matmul(points_homo, transformation.to(points).T)

    # Remove the homogeneous coordinate from each transformed point
    transformed_points = transformed_points_homo[:, :3]

    return transformed_points


def transform_batch_torch(points: torch.tensor, transformation: torch.tensor):
    """
    Transform a batch of points by a batch of transformation matrices
    Args:
        points: N,3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
        transformation: N,4,4 torch tensor, the transformation matrices
    Returns:
        transformed_points: N,3 torch tensor, the transformed coordinates
    """

    # Extract rotation and translation components
    rotation = transformation[:, :3, :3].to(points)
    translation = transformation[:, :3, 3:].to(points)

    # Reshape points to match dimensions for batch matrix multiplication
    points = points.unsqueeze(-1)

    # Perform batch matrix multiplication using torch.bmm(), instead of memory hungry matmul
    transformed_points = torch.bmm(rotation, points) + translation

    # Squeeze to remove the last dimension
    transformed_points = transformed_points.squeeze(-1)

    return transformed_points


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
    _quantization = 1000  # if change to 1, then it would take the first (smallest) index lie in the voxel

    offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1) ** 0.5
    dist = (
        dist / dist.max() * (_quantization - 1)
    ).long()  # for speed up # [0-_quantization]

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx


def voxel_down_sample_min_value_torch(
    points: torch.tensor, voxel_size: float, value: torch.tensor
):
    """
        voxel based downsampling. Returns the indices of the points which has the minimum value in the voxel.
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`
    """
    _quantization = 1000

    offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
    grid = torch.floor(points / voxel_size)
    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    # not same value, taker the smaller value, same value, consider the smaller index
    value = (value / value.max() * (_quantization - 1)).long()  # [0-_quantization]
    idx_d = idx_d + value * offset

    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
    # This operation may behave nondeterministically when given tensors on
    # a CUDA device. consider to change a more stable implementation

    idx = idx % offset
    return idx


# split a large point cloud into bounding box chunks
def split_chunks(
    pc: o3d.geometry.PointCloud(),
    aabb: o3d.geometry.AxisAlignedBoundingBox(),
    chunk_m: float = 100.0
):
    """
    Split a large point cloud into bounding box chunks
    """
    if not pc.has_points():
        return None

    # aabb = pc.get_axis_aligned_bounding_box()
    chunk_aabb = []

    min_bound = aabb.get_min_bound()
    max_bound = (
        aabb.get_max_bound() + 1e-5
    )  # just to gurantee there's a zero on one side
    bbx_range = max_bound - min_bound
    if bbx_range[0] > bbx_range[1]:
        axis_split = 0
        axis_kept = 1
    else:
        axis_split = 1
        axis_kept = 0
    chunk_split = np.arange(min_bound[axis_split], max_bound[axis_split], chunk_m)

    chunk_count = 0

    for i in range(len(chunk_split)):
        cur_min_bound = np.copy(min_bound)  # you need to clone, otherwise value changed
        cur_max_bound = np.copy(max_bound)
        cur_min_bound[axis_split] = chunk_split[i]
        if i < len(chunk_split) - 1:
            cur_max_bound[axis_split] = chunk_split[i + 1]

        cur_aabb = o3d.geometry.AxisAlignedBoundingBox(cur_min_bound, cur_max_bound)
        # pc_clone = copy.deepcopy(pc)

        pc_chunk = pc.crop(cur_aabb)  # crop as clone, original one would not be changed
        cur_pc_aabb = pc_chunk.get_axis_aligned_bounding_box()

        chunk_min_bound = cur_pc_aabb.get_min_bound()
        chunk_max_bound = cur_pc_aabb.get_max_bound()
        chunk_range = chunk_max_bound - chunk_min_bound
        if chunk_range[axis_kept] > chunk_m * 3:
            chunk_split_2 = np.arange(
                chunk_min_bound[axis_kept], chunk_max_bound[axis_kept], chunk_m
            )
            for j in range(len(chunk_split_2)):
                cur_chunk_min_bound = np.copy(
                    chunk_min_bound
                )  # you need to clone, otherwise value changed
                cur_chunk_max_bound = np.copy(chunk_max_bound)
                cur_chunk_min_bound[axis_kept] = chunk_split_2[j]
                if j < len(chunk_split_2) - 1:
                    cur_chunk_max_bound[axis_kept] = chunk_split_2[j + 1]
                cur_aabb = o3d.geometry.AxisAlignedBoundingBox(
                    cur_chunk_min_bound, cur_chunk_max_bound
                )
                pc_chunk = pc.crop(
                    cur_aabb
                )  # crop as clone, original one would not be changed
                cur_pc_aabb = pc_chunk.get_axis_aligned_bounding_box()
                chunk_count += 1
                chunk_aabb.append(cur_pc_aabb)
        else:
            chunk_count += 1
            chunk_aabb.append(cur_pc_aabb)

    # print("# Chunk for meshing: ", chunk_count)
    return chunk_aabb


# torch version of lidar undistortion (deskewing)
def deskewing(
    points: torch.tensor, ts: torch.tensor, pose: torch.tensor, ts_mid_pose=0.5
):
    """
    Deskew a batch of points at timestamp ts by a relative transformation matrix
    """
    if ts is None:
        return points  # no deskewing

    # pose as T_last<-cur
    # ts is from 0 to 1 as the ratio
    ts = ts.squeeze(-1)

    # Normalize the tensor to the range [0, 1]
    # NOTE: you need to figure out the begin and end of a frame because
    # sometimes there's only partial measurements, some part are blocked by some occlussions
    min_ts = torch.min(ts)
    max_ts = torch.max(ts)
    ts = (ts - min_ts) / (max_ts - min_ts)

    # this is related to: https://github.com/PRBonn/kiss-icp/issues/299
    ts -= ts_mid_pose 

    rotmat_slerp = roma.rotmat_slerp(
        torch.eye(3).to(points), pose[:3, :3].to(points), ts
    )

    tran_lerp = ts[:, None] * pose[:3, 3].to(points)

    points_deskewd = points
    points_deskewd[:, :3] = (rotmat_slerp @ points[:, :3].unsqueeze(-1)).squeeze(-1) + tran_lerp

    return points_deskewd


def tranmat_close_to_identity(mats: np.ndarray, rot_thre: float, tran_thre: float):
    """
    Check if a batch of transformation matrices is close to identity
    """
    rot_diff = np.abs(mats[:3, :3] - np.identity(3))

    rot_close_to_identity = np.all(rot_diff < rot_thre)

    tran_diff = np.abs(mats[:3, 3])

    tran_close_to_identity = np.all(tran_diff < tran_thre)

    if rot_close_to_identity and tran_close_to_identity:
        return True
    else:
        return False

def feature_pca_torch(data, principal_components = None,
                     principal_dim: int = 3,
                     down_rate: int = 1,
                     project_data: bool = True,
                     normalize: bool = True,
                     chunk_size: int = 16384):
    """
        do PCA to a NxD torch tensor to get the data along the K principle dimensions
        N is the data count, D is the dimension of the data

        We can also use a pre-computed principal_components for only the projection of input data
    """

    N, D = data.shape

    # Step 1: Center the data (subtract the mean of each dimension)
    data_centered = data - data.mean(dim=0)

    if principal_components is None:
        data_centered_for_compute = data_centered[::down_rate]

        assert data_centered_for_compute.shape[0] > principal_dim, "not enough data for PCA computation, down_rate might be too large or original data count is too small"

        # Step 2: Compute the covariance matrix (D x D)
        cov_matrix = torch.matmul(data_centered_for_compute.T, data_centered_for_compute) / (N - 1)

        # Step 3: Perform eigen decomposition of the covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        eigenvalues_r = eigenvalues.real.to(data)
        eigenvectors_r = eigenvectors.real.to(data)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues_r, descending=True)
        principal_components = eigenvectors_r[:, sorted_indices[:principal_dim]]  # First 3 principal components

    data_pca = None
    if project_data:
        # Step 5: Project data onto the top 3 principal components
        data_pca_chunks = []
        for i in range(0, N, chunk_size):
            chunk = data_centered[i:i + chunk_size]
            chunk_pca = torch.matmul(chunk, principal_components)  # N, D @ D, P
            data_pca_chunks.append(chunk_pca)

        data_pca = torch.cat(data_pca_chunks, dim=0)

        # normalize to show as rgb
        if normalize: 

            # # deal with outliers
            quantile_down_rate = 37 # quantile has count limit, downsample the data to avoid the limit
            min_vals = torch.quantile(data_pca[::quantile_down_rate], 0.02, dim=0, keepdim=True)
            max_vals = torch.quantile(data_pca[::quantile_down_rate], 0.98, dim=0, keepdim=True)

            # Normalize to range [0, 1]
            data_pca.sub_(min_vals).div_(max_vals - min_vals)
            # data_pca = data_pca.clamp(0, 1)

    return data_pca, principal_components

def plot_timing_detail(time_table: np.ndarray, saving_path: str, with_loop=False):
    """
    Plot the timing detail for processing per frame
    """
    frame_count = time_table.shape[0]
    time_table_ms = time_table * 1e3

    for i in range(time_table.shape[1] - 1):  # accumulated time
        time_table_ms[:, i + 1] += time_table_ms[:, i]

    # font1 = {'family': 'Times New Roman', 'weight' : 'normal', 'size': 16}
    # font2 = {'family': 'Times New Roman', 'weight' : 'normal', 'size': 18}
    font2 = {"weight": "normal", "size": 18}

    color_values = np.linspace(0, 1, 6)
    # Get the colors from the "viridis" colormap at the specified values
    # plasma, Pastel1, tab10
    colors = [viridis(x) for x in color_values]

    fig = plt.figure(figsize=(12.0, 4.0))

    frame_array = np.arange(frame_count)
    realtime_limit = 100.0 * np.ones([frame_count, 1])
    ax1 = fig.add_subplot(111)

    line_width_1 = 0.6
    line_width_2 = 1.0
    alpha_value = 1.0

    ax1.fill_between(
        frame_array,
        time_table_ms[:, 0],
        facecolor=colors[0],
        edgecolor="face",
        where=time_table_ms[:, 0] > 0,
        alpha=alpha_value,
        interpolate=True,
    )
    ax1.fill_between(
        frame_array,
        time_table_ms[:, 0],
        time_table_ms[:, 1],
        facecolor=colors[1],
        edgecolor="face",
        where=time_table_ms[:, 1] > time_table_ms[:, 0],
        alpha=alpha_value,
        interpolate=True,
    )
    ax1.fill_between(
        frame_array,
        time_table_ms[:, 1],
        time_table_ms[:, 2],
        facecolor=colors[2],
        edgecolor="face",
        where=time_table_ms[:, 2] > time_table_ms[:, 1],
        alpha=alpha_value,
        interpolate=True,
    )
    ax1.fill_between(
        frame_array,
        time_table_ms[:, 2],
        time_table_ms[:, 3],
        facecolor=colors[3],
        edgecolor="face",
        where=time_table_ms[:, 3] > time_table_ms[:, 2],
        alpha=alpha_value,
        interpolate=True,
    )
    if with_loop:
        ax1.fill_between(
            frame_array,
            time_table_ms[:, 3],
            time_table_ms[:, 4],
            facecolor=colors[4],
            edgecolor="face",
            where=time_table_ms[:, 4] > time_table_ms[:, 3],
            alpha=alpha_value,
            interpolate=True,
        )

    ax1.plot(frame_array, realtime_limit, "--", linewidth=line_width_2, color="k")

    plt.tick_params(labelsize=12)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]

    plt.xlim((0, frame_count - 1))
    plt.ylim((0, 200))

    plt.xlabel("Frame ID", font2)
    plt.ylabel("Runtime (ms)", font2)
    plt.tight_layout()
    # plt.title('Timing table')

    if with_loop:
        legend = plt.legend(
            (
                "Pre-processing",
                "Odometry",
                "Mapping preparation",
                "Map optimization",
                "Loop closures",
            ),
            prop=font2,
            loc=2,
        )
    else:
        legend = plt.legend(
            ("Pre-processing", "Odometry", "Mapping preparation", "Map optimization"),
            prop=font2,
            loc=2,
        )

    plt.savefig(saving_path, dpi=500)
    # plt.show()
