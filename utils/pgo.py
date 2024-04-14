#!/usr/bin/env python3
# @file      pgo.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import numpy as np
from numpy.linalg import inv
import gtsam
import matplotlib.pyplot as plt
from rich import print

from utils.config import Config
    
class PoseGraphManager:
    def __init__(self, config: Config):

        self.config = config

        self.silence = config.silence

        self.fixed_cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9])) # fixed

        tran_std = config.pgo_tran_std # m
        rot_std = config.pgo_rot_std # degree # better to be small

        self.const_cov = np.array([np.radians(rot_std), np.radians(rot_std), np.radians(rot_std), tran_std, tran_std, tran_std]) # first rotation, then translation
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)
        
        # not used (figure it out)
        mEst = gtsam.noiseModel.mEstimator.GemanMcClure(1.0)
        self.robust_loop_cov = gtsam.noiseModel.Robust(mEst, self.loop_cov)
        self.robust_odom_cov = gtsam.noiseModel.Robust(mEst, self.odom_cov)

        self.graph_factors = gtsam.NonlinearFactorGraph() # edges # with pose and pose covariance
        self.graph_initials = gtsam.Values() # initial guess # as pose

        self.cur_pose = None
        self.curr_node_idx = None
        self.graph_optimized = None

        self.init_poses = []
        self.pgo_poses = []
        self.loop_edges_vis = []
        self.loop_edges = []
        self.loop_trans = []

        self.min_loop_idx = config.end_frame+1
        self.last_loop_idx = 0
        self.drift_radius = 0.0 # m
        self.pgo_count = 0
        self.last_error = 0.0

    def add_frame_node(self, frame_id, init_pose):
        """create frame pose node and set pose initial guess  
        Args:
            frame_id: int
            init_pose (np.array): 4x4, as T_world<-cur
        """
        self.curr_node_idx = frame_id # make start with 0
        if not self.graph_initials.exists(gtsam.symbol('x', frame_id)): # create if not yet exists
            self.graph_initials.insert(gtsam.symbol('x', frame_id), gtsam.Pose3(init_pose))
        
    def add_pose_prior(self, frame_id: int, prior_pose: np.ndarray, fixed: bool = False):
        """add pose prior unary factor  
        Args:
            frame_id: int
            prior_pose (np.array): 4x4, as T_world<-cur
            dist_ratio: float , use to determine the covariance, the std is porpotional to this dist_ratio
            fixed: bool, if True, this frame is fixed with very low covariance
        """

        if fixed:
            cov_model = self.fixed_cov
        else:
            tran_sigma = self.drift_radius+1e-4 # avoid divide by 0
            rot_sigma = self.drift_radius * np.radians(10.0)
            cov_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([rot_sigma, rot_sigma, rot_sigma, tran_sigma, tran_sigma, tran_sigma]))
        
        self.graph_factors.add(gtsam.PriorFactorPose3(
                                            gtsam.symbol('x', frame_id), 
                                            gtsam.Pose3(prior_pose), 
                                            cov_model))

    def add_odometry_factor(self, cur_id: int, last_id: int, odom_transform: np.ndarray, cov = None):
        """add a odometry factor between two adjacent pose nodes
        Args:
            cur_id: int
            last_id: int
            odom_transform (np.array): 4x4 , as T_prev<-cur
            cov (np.array): 6x6 covariance matrix, if None, set to the default value
        """
        
        if cov is None:
            cov_model = self.odom_cov
        else:
            cov_model = gtsam.noiseModel.Gaussian.Covariance(cov)

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', last_id), #t-1
                                                gtsam.symbol('x', cur_id),  #t
                                                gtsam.Pose3(odom_transform),  # T_prev<-cur
                                                cov_model))  # NOTE: add robust kernel

    def add_loop_factor(self, cur_id: int, loop_id: int, loop_transform: np.ndarray, cov = None, reject_outlier = True):
        """add a loop closure factor between two pose nodes
        Args:
            cur_id: int
            loop_id: int
            loop_transform (np.array): 4x4 , as T_loop<-cur
            cov (np.array): 6x6 covariance matrix, if None, set to the default value
        """

        if cov is None:
            cov_model = self.loop_cov
        else:
            cov_model = gtsam.noiseModel.Gaussian.Covariance(cov)

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', loop_id), #l
                                                gtsam.symbol('x', cur_id),  #t 
                                                gtsam.Pose3(loop_transform),  # T_loop<-cur
                                                cov_model))  # NOTE: add robust kernel
        
        cur_error = self.graph_factors.error(self.graph_initials)
        valid_error_thre = self.last_error + (cur_id - self.last_loop_idx) * self.config.pgo_error_thre_frame
        if reject_outlier and cur_error > valid_error_thre:
            if not self.silence:
                print("[bold yellow]A loop edge rejected due to too large error[/bold yellow]")
            self.graph_factors.remove(self.graph_factors.size()-1)
            return False
        return True

    def optimize_pose_graph(self):
        
        if self.config.pgo_with_lm:
            opt_param = gtsam.LevenbergMarquardtParams()
            opt_param.setMaxIterations(self.config.pgo_max_iter)
            opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_initials, opt_param)
        else: # pgo with dogleg
            opt_param = gtsam.DoglegParams()
            opt_param.setMaxIterations(self.config.pgo_max_iter)
            opt = gtsam.DoglegOptimizer(self.graph_factors, self.graph_initials, opt_param)  
          
        error_before = self.graph_factors.error(self.graph_initials)

        self.graph_optimized = opt.optimizeSafely()

        # Calculate marginal covariances for all variables
        # marginals = gtsam.Marginals(self.graph_factors, self.graph_optimized)
        # try to even visualize the covariance
        # cov = get_node_cov(marginals, 50)
        # print(cov)

        error_after = self.graph_factors.error(self.graph_optimized)
        if not self.silence:
            print("[bold red]PGO done[/bold red]")
            print("error %f --> %f:" % (error_before, error_after))

        self.graph_initials = self.graph_optimized # update the initial guess

        # update the pose of each frame after pgo
        frame_count = self.curr_node_idx+1
        self.pgo_poses = [None] * frame_count # start from 0
        for idx in range(frame_count):
            self.pgo_poses[idx] = get_node_pose(self.graph_optimized, idx)

        self.cur_pose = self.pgo_poses[self.curr_node_idx] 

        self.pgo_count += 1
        self.last_error = error_after

    # write the pose graph as the g2o format
    def write_g2o(self, out_file):
        gtsam.writeG2o(self.graph_factors, self.graph_initials, out_file)

    # write loop data into txt file
    def write_loops(self, out_file):
        with open(out_file, 'w') as file:
            for indices, array_data in zip(self.loop_edges, self.loop_trans):
                # Write the indices to the file
                file.write(f"{indices[0]} {indices[1]}\n")

                # Write the array data to the file with 4 numbers per row
                reshaped_array = array_data.reshape(-1, 4)
                np.savetxt(file, reshaped_array, delimiter=' ', fmt='%f')

    # read loop data from txt file
    def read_loops(self, in_file, subsample_rate=1):
        self.loop_edges = []
        self.loop_trans = []
        try:
            with open(in_file, 'r') as file:
                lines = file.readlines()  # Read all lines from the file

                # Process each line in the file
                # Assuming each array data is followed by 5 lines (indices + array data)
                for i in range(0, len(lines), 5*subsample_rate):  
                    # Extract indices from the current line
                    indices = np.array([int(num) for num in lines[i].strip().split(' ')])
                    self.loop_edges.append(indices)

                    # Extract array data from the next 4 lines
                    array_data_lines = [line.strip().split() for line in lines[i+1:i+4]]
                    tran_array = np.array([[float(num) for num in row] for row in array_data_lines])

                    # add noise for robsutness check (only for debug)
                    # tran_array[:3,3]+= ((np.random.rand(3)-0.5)*1.0)

                    self.loop_trans.append(tran_array)
                return True
        except IOError:
            return False

    # do pgo with known odom and loop data offline (for debugging)
    def offline_pgo(self, odom_poses):
        # initialize 
        self.graph_factors = gtsam.NonlinearFactorGraph() # edges # with pose and pose covariance
        self.graph_initials = gtsam.Values() # initial guess # as pose

        # create nodes
        for i in range(len(odom_poses)):
            self.add_frame_node(i, odom_poses[i]) 

        # pose prior
        self.add_pose_prior(0, odom_poses[0], fixed=True)

        # create odom edges
        for i in range(len(odom_poses)-1):
            odom_tran = inv(odom_poses[i]) @ odom_poses[i+1]
            self.add_odometry_factor(i+1, i, odom_tran)

        # create loop edges
        for i in range(len(self.loop_edges)):
            self.add_loop_factor(self.loop_edges[i][1], self.loop_edges[i][0], self.loop_trans[i], reject_outlier=False)
        
        # optimize 
        self.optimize_pose_graph()
        # poses after optimization: pgo_poses

    # get the difference of poses before and after pgo        
    def get_pose_diff(self):
        assert len(self.pgo_poses) == len(self.init_poses), "Lists of poses must have the same size."
        pose_diff = np.array([(pgo_pose @ np.linalg.inv(init_pose)) for pgo_pose, init_pose in zip(self.pgo_poses, self.init_poses)])
        return pose_diff
    
    def estimate_drift(self, travel_dist_list, used_frame_id, drfit_ratio = 0.01, correct_ratio = 0.01):
        # estimate the current drift # better to calculate according to residual
        self.drift_radius = (travel_dist_list[used_frame_id] - travel_dist_list[self.last_loop_idx])*drfit_ratio
        if self.min_loop_idx < self.last_loop_idx: # the loop has been corrected previously
            self.drift_radius += (travel_dist_list[self.min_loop_idx] + travel_dist_list[used_frame_id]*correct_ratio)*drfit_ratio
        # print("Estimated drift (m):", self.drift_radius)
    
    def plot_loops(self, loop_plot_path, vis_now = False):
    
        pose_count = len(self.pgo_poses)
        z_ratio = 0.002
        ts = np.arange(0, pose_count, 1) * z_ratio

        traj_est = np.vstack([pose[:3, 3] for pose in self.pgo_poses])
        # print(poses_est)

        # Create a 3D plot
        fig = plt.figure() # facecolor='white'
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory
        ax.plot(traj_est[:,0], traj_est[:,1], ts, 'k')

        for loop in self.loop_edges:
            node_0 = loop[0]
            node_1 = loop[1]
            ax.plot([traj_est[node_0, 0], traj_est[node_1, 0]], [traj_est[node_0, 1], traj_est[node_1, 1]], [ts[node_0], ts[node_1]], color='green')

        ax.grid(False)
        ax.set_axis_off()
        # turn of the gray background
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_box_aspect([1, 1, 1])

        plt.tight_layout()

        if loop_plot_path is not None:
            plt.savefig(loop_plot_path, dpi=600)

        # Show the plot
        if vis_now:
            plt.show()


def get_node_pose(graph, idx):

    pose = graph.atPose3(gtsam.symbol('x', idx))
    # print(pose)
    pose_se3 = np.eye(4)
    pose_se3[:3, 3] = np.array([pose.x(), pose.y(), pose.z()])
    pose_se3[:3, :3] = pose.rotation().matrix()

    return pose_se3

def get_node_cov(marginals, idx):

    cov = marginals.marginalCovariance(gtsam.symbol('x', idx))
    # print(cov)
    # pose_se3 = np.eye(4)
    # pose_se3[:3, 3] = np.array([pose.x(), pose.y(), pose.z()])
    # pose_se3[:3, :3] = pose.rotation().matrix()
    
    return cov