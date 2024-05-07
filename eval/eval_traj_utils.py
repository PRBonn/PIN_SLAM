#!/usr/bin/env python3
# @file      eval_traj_utils.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# our implmentation
def absolute_error(
    poses_gt: np.ndarray, poses_result: np.ndarray, align_on: bool = True
):
    assert poses_gt.shape[0] == poses_result.shape[0], "poses length should be identical"
    align_mat = np.eye(4)
    if align_on:
        align_rot, align_tran, _ = align_traj(poses_result, poses_gt)
        align_mat[:3, :3] = align_rot
        align_mat[:3, 3] = np.squeeze(align_tran)

    frame_count = poses_gt.shape[0]

    rot_errors = []
    tran_errors = []

    for i in range(frame_count):
        cur_results_pose_aligned = align_mat @ poses_result[i]
        cur_gt_pose = poses_gt[i]
        delta_rot = (
            np.linalg.inv(cur_gt_pose[:3, :3]) @ cur_results_pose_aligned[:3, :3]
        ) 
        delta_tran = cur_gt_pose[:3, 3] - cur_results_pose_aligned[:3, 3]

        # the one used for kiss-icp
        # delta_tran = cur_gt_pose[:3,3] - delta_rot @ cur_results_pose_aligned[:3,3]

        delta_rot_theta = rotation_error(delta_rot)
        delta_t = np.linalg.norm(delta_tran)

        rot_errors.append(delta_rot_theta)
        tran_errors.append(delta_t)

    rot_errors = np.array(rot_errors)
    tran_errors = np.array(tran_errors)

    rot_rmse = (
        np.sqrt(np.dot(rot_errors, rot_errors) / frame_count) * 180.0 / np.pi
    )  # this seems to have some problem
    tran_rmse = np.sqrt(np.dot(tran_errors, tran_errors) / frame_count)

    # rot_mean = np.mean(rot_errors)
    # tran_mean = np.mean(tran_errors)

    # rot_median = np.median(rot_errors)
    # tran_median = np.median(tran_errors)

    # rot_std = np.std(rot_errors)
    # tran_std = np.std(tran_errors)

    return rot_rmse, tran_rmse, align_mat


def align_traj(poses_np_1, poses_np_2):

    traj_1 = poses_np_1[:,:3,3].squeeze().T
    traj_2 = poses_np_2[:,:3,3].squeeze().T
    
    return align(traj_1, traj_2)


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    Borrowed from NICE-SLAM
    """
    model_zerocentered = model - model.mean(1, keepdims=True)
    data_zerocentered = data - data.mean(1, keepdims=True)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1, keepdims=True) - rot * model.mean(1, keepdims=True)

    model_aligned = rot * model + trans

    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[
        0
    ]  # as RMSE

    return rot, trans, trans_error


def relative_error(poses_gt, poses_result):
    """calculate sequence error (kitti metric, relative drifting error)
    Args:
        poses_gt, kx4x4 np.array, ground truth poses
        poses_result, kx4x4 np.array, predicted poses
    Returns:
        err (list list): [first_frame, rotation error, translation error, length, speed]
            - first_frame: frist frame index
            - rotation error: rotation error per length
            - translation error: translation error per length
            - length: evaluation trajectory length
            - speed: car speed (#FIXME: 10FPS is assumed)
    """
    assert poses_gt.shape[0] == poses_result.shape[0], "poses length should be identical"
    err = []
    dist = trajectory_distances(poses_gt)
    step_size = 10

    lengths = [100, 200, 300, 400, 500, 600, 700, 800]  # unit: m
    num_lengths = len(lengths)

    for first_frame in range(0, poses_gt.shape[0], step_size):
        for i in range(num_lengths):
            len_ = lengths[i]
            last_frame = last_frame_from_segment_length(dist, first_frame, len_)

            # Continue if sequence not long enough
            if last_frame == -1:
                continue

            # compute rotational and translational errors
            pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) @ poses_gt[last_frame]
            pose_delta_result = (
                np.linalg.inv(poses_result[first_frame]) @ poses_result[last_frame]
            )

            pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt

            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)

            # compute speed
            num_frames = last_frame - first_frame + 1.0
            speed = len_ / (0.1 * num_frames)

            err.append([first_frame, r_err / len_, t_err / len_, len_, speed])

    t_err = 0
    r_err = 0

    if len(err) == 0:  # the case when the trajectory is not long enough
        return 0, 0

    for i in range(len(err)):
        r_err += err[i][1]
        t_err += err[i][2]

    r_err /= len(err)
    t_err /= len(err)
    drift_ate = t_err * 100.0
    drift_are = r_err / np.pi * 180.0

    return drift_ate, drift_are


def trajectory_distances(poses_np):
    """Compute distance for each pose w.r.t frame-0
    Args:
        poses kx4x4 np.array
    Returns:
        dist (float list): distance of each pose w.r.t frame-0
    """
    dist = [0]

    for i in range(poses_np.shape[0] - 1):
        rela_dist = np.linalg.norm(poses_np[i+1] - poses_np[i])
        dist.append(dist[i] + rela_dist)
        
    return dist


def rotation_error(pose_error):
    """Compute rotation error
    From a rotation matrix to the axis angle, use the angle as the result
    Args:
        pose_error (4x4 or 3x3 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    # 0.5 * (trace - 1)
    d = 0.5 * (a + b + c - 1.0)
    # make sure the rot_mat is valid (trace < 3, det = 1)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))  # in rad
    return rot_error


def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error


def last_frame_from_segment_length(dist, first_frame, length):
    """Find frame (index) that away from the first_frame with
    the required distance
    Args:
        dist (float list): distance of each pose w.r.t frame-0
        first_frame (int): start-frame index
        length (float): required distance
    Returns:
        i (int) / -1: end-frame index. if not found return -1
    """
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + length):
            return i
    return -1


def plot_trajectories(
    traj_plot_path: str,
    poses_est,
    poses_ref,
    poses_est_2=None,
    plot_3d: bool = True,
    grid_on: bool = True,
    plot_start_end_markers: bool = True,
    vis_now: bool = False,
    close_all: bool = True,
) -> None:
    # positions_est, positions_ref, positions_est_2 as list of numpy array

    from evo.core.trajectory import PosePath3D
    from evo.tools import plot as evoplot
    from evo.tools.settings import SETTINGS

    # without alignment

    if close_all:
        plt.close("all")

    poses = PosePath3D(poses_se3=poses_est)
    gt_poses = PosePath3D(poses_se3=poses_ref)
    if poses_est_2 is not None:
        poses_2 = PosePath3D(poses_se3=poses_est_2)

    if plot_3d:
        plot_mode = evoplot.PlotMode.xyz
    else:
        plot_mode = evoplot.PlotMode.xy

    fig = plt.figure(f"Trajectory results")
    ax = evoplot.prepare_axis(fig, plot_mode)
    evoplot.traj(
        ax=ax,
        plot_mode=plot_mode,
        traj=gt_poses,
        label="ground truth",
        style=SETTINGS.plot_reference_linestyle,
        color=SETTINGS.plot_reference_color,
        alpha=SETTINGS.plot_reference_alpha,
        plot_start_end_markers=False,
    )
    evoplot.traj(
        ax=ax,
        plot_mode=plot_mode,
        traj=poses,
        label="PIN-SLAM",
        style=SETTINGS.plot_trajectory_linestyle,
        color="#4c72b0bf",
        alpha=SETTINGS.plot_trajectory_alpha,
        plot_start_end_markers=plot_start_end_markers,
    )
    if poses_est_2 is not None:  # better to change color (or the alpha)
        evoplot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=poses_2,
            label="PIN-Odom",
            style=SETTINGS.plot_trajectory_linestyle,
            color="#FF940E",
            alpha=SETTINGS.plot_trajectory_alpha / 3.0,
            plot_start_end_markers=False,
        )

    plt.tight_layout()
    ax.legend(frameon=grid_on)

    if traj_plot_path is not None:
        plt.savefig(traj_plot_path, dpi=600)

    if vis_now:
        plt.show()


def read_kitti_format_calib(filename: str):
    """
    read calibration file (with the kitti format)
    returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)

    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib


def read_kitti_format_poses(filename: str) -> List[np.ndarray]:
    """
    read pose file (with the kitti format)
    returns -> list, transformation before calibration transformation
    """
    pose_file = open(filename)

    poses = []

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(pose)

    pose_file.close()
    return poses


# copyright: Nacho et al. KISS-ICP
def apply_kitti_format_calib(poses: List[np.ndarray], calib_T_cl) -> List[np.ndarray]:
    """Converts from Velodyne to Camera Frame (# T_camera<-lidar)"""
    poses_calib = []
    for pose in poses:
        poses_calib.append(calib_T_cl @ pose @ np.linalg.inv(calib_T_cl))
    return poses_calib


# copyright: Nacho et al. KISS-ICP
def write_kitti_format_poses(filename: str, poses: List[np.ndarray]):
    def _to_kitti_format(poses: np.ndarray) -> np.ndarray:
        return np.array([np.concatenate((pose[0], pose[1], pose[2])) for pose in poses])

    np.savetxt(fname=f"{filename}_kitti.txt", X=_to_kitti_format(poses))


# for LiDAR dataset
def get_metrics(seq_result: List[Dict]):
    odom_ate = (seq_result[0])["Average Translation Error [%]"]
    odom_are = (seq_result[0])["Average Rotational Error [deg/m]"] * 100.0
    slam_rmse = (seq_result[1])["Absoulte Trajectory Error [m]"]
    metrics_dict = {
        "Odometry ATE [%]": odom_ate,
        "Odometry ARE [deg/100m]": odom_are,
        "SLAM RMSE [m]": slam_rmse,
    }
    return metrics_dict


def mean_metrics(seq_metrics: List[Dict]):
    sums = defaultdict(float)
    counts = defaultdict(int)

    for seq_metric in seq_metrics:
        for key, value in seq_metric.items():
            sums[key] += value
            counts[key] += 1

    mean_metrics = {key: sums[key] / counts[key] for key in sums}
    return mean_metrics
