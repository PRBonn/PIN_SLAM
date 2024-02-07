#!/usr/bin/env python3
# @file      config.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import yaml
import os
import torch

class Config:
    def __init__(self):

        # Default values (most of the parameters would be kept as default or adaptive)

        # settings
        self.name: str = "dummy"  # experiment name

        self.run_path: str = ""

        self.output_root: str = ""  # output root folder
        self.pc_path: str = ""  # input point cloud folder
        self.pose_path: str = ""  # input pose file
        self.calib_path: str = ""  # input calib file (to sensor frame)

        self.label_path: str = "" # input point-wise label path, for semantic mapping

        self.closed_pose_path = None

        self.load_model: bool = False  # load the pre-trained model or not
        self.model_path: str = "/"  # pre-trained model path

        self.first_frame_ref: bool = False  # if false, we directly use the world
        # frame as the reference frame
        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = 0  # end at this frame
        self.every_frame: int = 1  # process every x frame

        self.seed: int = 42 # random seed for the experiment
        self.num_workers: int = 12 # number of worker for the dataloader
        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id
        self.dtype = torch.float32 # default torch tensor data type
        self.tran_dtype = torch.float64 # dtype used for all the transformation and poses

        # dataset specific
        self.kitti_correction_on: bool = False
        self.correction_deg: float = 0.0

        # motion undistortion
        self.deskew: bool = False
        self.valid_ts_in_points: bool = True
        self.lidar_type_guess: str = "velodyne"

        # process
        self.min_range: float = 2.5 # filter too-close points (and 0 artifacts)
        self.max_range: float = 60.0 # filter far-away points

        self.adaptive_range_on: bool = False # use an adpative range (used for NCD)

        # block with such radius, but actually a square (unit: m)
        self.min_z: float = -4.0  # filter for z coordinates (unit: m)
        self.max_z: float = 100.0

        self.rand_downsample: bool = True  # apply random or voxel downsampling to input original point clcoud
        self.vox_down_m: float = 0.05 # the voxel size if using voxel downsampling (unit: m)
        self.rand_down_r: float = 1.0 # the decimation ratio if using random downsampling (0-1)
   
        self.filter_noise: bool = False  # use SOR to remove the noise or not
        self.sor_nn: int = 25  # SOR neighborhood size
        self.sor_std: float = 2.5  # SOR std threshold

        self.estimate_normal: bool = True  # estimate surface normal or not

        # semantic related
        self.semantic_on: bool = False # semantic shine mapping on [semantic]
        self.sem_class_count: int = 20 # semantic class count: 20 for semantic kitti
        self.sem_label_decimation: int = 1 # use only 1/${sem_label_decimation} of the available semantic labels for training (fitting)
        self.freespace_label_on: bool = False
        self.filter_moving_object: bool = True

        # color (intensity) related 
        self.color_map_on: bool = False # texture mapping default on
        self.color_on: bool = False
        self.color_channel: int = 0 # For RGB, channel=3, For Lidar with intensity, channel=1

        # neural points
        self.weighted_first: bool = True # weighted the neighborhood feature before decoding to sdf or do the weighting of the decoded sdf afterwards

        self.layer_norm_on: bool = False # apply layer norm to the features

        self.voxel_size_m: float = 0.3 # we use the voxel hashing structure to maintain the neural points, the voxel size is set as this value
        self.max_points_per_voxel: int = 1 # we keep the maximum this number of neural feature points in each voxel
        
        self.num_nei_cells: int = 2 # the neighbor searching padding voxel # NOTE: can even be 3 when the motion is dramastic

        self.query_nn_k: int = 6 # query the point's k nearest neural points

        self.use_mid_ts: bool = False # use the middle of the created and last updated timestamp for adjusting or just use the created timestamp

        # NOTE: this is a very sophisticated parameter
        # the larger this value is, the larger neighborhood region would be, the more robust to the highly dynamic motion and also the more time-consuming
        self.search_alpha: float = 0.2

        self.idw_index: int = 2 # the index for IDW (inverse distance weighting)

        # self.query_nn_radius_m: float = 0.5 # only use the neural points within this radius (set to 1.0 maybe), this is not used

        # self.point_level_num: int = 1 # multi-res hashing # level
        self.buffer_size: int = int(5e7) # 1e8

        # shared by both kinds of feature 
        self.feature_dim: int = 8  # length of the feature for each grid feature
        self.feature_std: float = 0.0  # grid feature initialization standard deviation

        # Use all the surface samples or just the exact measurements to build the neural points map
        # If True may lead to larger memory, but is more robust while the reconstruction.
        self.from_sample_points: bool = True
        self.from_all_samples: bool = False  # even use the freespace samples (for better ESDF mapping at a cost of larger memory consumption)
        self.map_surface_ratio: float = 0.5 # 3.0 # ratio * surface sample std

        # for local map
        self.diff_ts_local: float = 400. # deprecated (use travel distance instead)
        self.local_map_travel_dist_ratio: float = 5.0
        self.local_map_radius: float = 50.0

        # map management
        self.prune_map_on: bool = False
        self.max_prune_certainty: float = 1.0 # neural point pruning threshold

        # positional encoding related [not used]
        self.use_gaussian_pe: bool = False # use Gaussian Fourier or original log positional encoding
        self.pos_encoding_freq: int = 200 # 200
        self.pos_encoding_band: int = 0 # if 0, without encoding
        self.pos_input_dim: int = 3
        self.pos_encoding_base: int = 2

        # sampler
        # spilt into 3 parts for sampling
        self.surface_sample_range_m: float = 0.25 # better to be set according to the noise level (actually a std for a gaussian distribution)
        self.surface_sample_n: int = 3
        self.free_sample_begin_ratio: float = 0.3
        self.free_sample_end_dist_m: float = 1.0 # maximum distance after the surface (unit: m)
        self.free_front_n: int = 2
        self.free_behind_n: int = 1

        # replay based (data pool related)
        self.window_radius: float = 50.0 # unit: m
        self.pool_capacity: int = int(1e7)
        self.bs_new_sample: int = 2048 # number of the sample per batch for the current frame's data, half of all the data
        self.new_certainty_thre: float = 1.0
        self.pool_filter_freq: int = 5 

        # tracking
        self.track_on: bool = True
        self.photometric_loss_on: bool = False # add the color (or intensity) [photometric loss] to the tracking loss
        self.photometric_loss_weight: float = 0.01 # weight for the photometric loss in tracking
        self.source_vox_down_m: float = 0.8
        self.uniform_motion_on: bool = True # use uniform motion (constant velocity) model for the transformation inital guess
        self.reg_min_grad_norm: float = 0.5
        self.reg_max_grad_norm: float = 2.0
        self.max_sdf_ratio: float = 5.0 # ratio * surface_sample sigma
        self.max_sdf_std_ratio: float = 1.0 # ratio * surface_sample sigma # 1.0
        self.reg_dist_div_grad_norm: bool = False # divide the sdf by the sdf gradient's norm for fixing overshoting or not
        self.reg_GM_dist_m: float = 0.5 # 0.3 # default value changed
        self.reg_GM_grad: float = 0.2 # 0.1 # default value changed, the smaller the value, the smaller the weight would be (give even smaller weight to the outliers)
        self.reg_lm_lambda: float = 1e-4 # 1e-3
        self.reg_iter_n: int = 50 # maximum iteration number for registration
        self.reg_term_thre_deg: float = 0.01
        self.reg_term_thre_m: float = 0.0005
        self.eigenvalue_check: bool = True
        self.consist_wieght_on: bool = True

        self.stop_frame_thre: int = 20 # 20, determine if the robot is stopped when there's almost no motion in a time peroid

        # decoder
        self.mlp_bias_on: bool = True
        
        self.geo_mlp_level: int = 1
        self.geo_mlp_hidden_dim: int = 64

        self.sem_mlp_level: int = 1
        self.sem_mlp_hidden_dim: int = 64

        self.color_mlp_level: int = 1
        self.color_mlp_hidden_dim: int = 64
        
        self.freeze_after_frame: int = 40  # 10, if the decoder model is not loaded , it would be trained and freezed after such frame number

        # loss
        # the main loss type, select from the sample sdf loss ('bce', 'l1', 'l2', 'zhong') 
        self.main_loss_type: str = 'bce'
        
        self.sigma_sigmoid_m: float = 0.1 # better to be set according to the noise level
        self.sigma_scale_constant: float = 0.0 # scale factor adding to the constant sigma value (linear with the distance) [deprecated]
        self.logistic_gaussian_ratio: float = 0.55 # the factor ratio for approximize a Gaussian distribution using the derivative of logistic function

        # conduct projective distance correction based on the sdf gradient or not
        self.proj_correction_on: bool = False # This does not work well 

        self.predict_sdf: bool = False
        self.loss_weight_on: bool = False  # if True, the weight would be given to the loss, if False, the weight would be used to change the sigmoid's shape
        self.behind_dropoff_on: bool = False  # behind surface drop off weight
        self.dist_weight_on: bool = True  
        self.dist_weight_scale: float = 0.8 # weight range [0.6, 1.4]
        
        self.dropoff_min_sigma: float = 1.0
        self.dropoff_max_sigma: float = 8.0
        self.normal_loss_on: bool = False
        self.weight_n: float = 0.01
        
        self.numerical_grad: bool = True # use numerical gradient as in the paper Neuralangelo
        self.gradient_decimation: int = 10 # use just a part of the points for the ekional loss when using the numerical grad, save computing time
        self.num_grad_step_ratio: float = 0.2 # step as a ratio of the surface sample sigma

        self.ekional_loss_on: bool = True
        self.ekional_add_to: str = 'all' # select from 'all', 'surface', 'freespace'
        self.weight_e: float = 0.5

        self.consistency_loss_on: bool = False
        self.weight_c: float = 0.5 # weight for consistency loss, don't mix with the color weight 
        self.consistency_count: int = 1000
        self.consistency_range: float = 0.05 # the neighborhood points would be randomly select within the radius of xxx m

        self.weight_s: float = 1.0  # weight for semantic classification loss
        self.weight_i: float = 1.0  # weight for color or intensity loss

        self.dynamic_filter_on: bool = False
        self.dynamic_certainty_thre: float = 4.0
        self.dynamic_sdf_ratio_thre: float = 1.5

        # optimizer
        self.mapping_freq_frame: int = 1
        self.ba_freq_frame: int = 0 # 10
        self.ba_frame: int = 50 # window size for ba

        # to have a better reconstruction results, you need to set a larger iters, a smaller lr
        self.iters: int = 15
        self.init_iter_ratio: int = 40
        self.opt_adam: bool = True  # use adam or sgd
        self.bs: int = 16384
        self.lr: float = 0.01
        self.lr_pose: float = 1e-3
        self.weight_decay: float = 0. # 1e-7
        self.adam_eps: float = 1e-15

        # loop closure detection
        self.global_loop_on: bool = True # global loop detection using context
        self.local_map_context: bool = False # use local map or scan context for loop closure description
        self.loop_with_feature: bool = False # encode neural point feature in the context
        self.min_loop_travel_dist_ratio: float = 3.5 # accumulated travel distance should be larger than theis ratio * local map radius to be considered as an valid candidate
        self.local_map_context_latency: int = 0 # 10
        self.loop_local_map_time_window: int = 100
        self.context_shape = [20, 60] # [20, 60] 
        self.context_num_candidates: int = 1
        self.context_cosdist_threshold: float = 0.2 # 0.15
        self.context_virtual_side_count: int = 4 # 6

        self.use_gt_loop: bool = False # use the gt loop closure derived from the gt pose or not (only used for debugging)
        self.max_loop_dist: float = 8.0
    
        # pose graph optimization
        self.pgo_on: bool = False
        self.pgo_freq: int = 30 # frequency for detecting loop closure
        self.pgo_with_lm: bool = True # use lm or dogleg # lm seems to be better (why)
        self.pgo_max_iter: int = 50
        self.pgo_with_pose_prior: bool = False # use the pose prior or not during the pgo
        self.pgo_tran_std: float = 0.04 # m
        self.pgo_rot_std: float = 0.01 # deg
        self.use_reg_cov_mat: bool = False # use the covariance matrix directly calculated by the registration for pgo edges or not

        self.pgo_merge_map: bool = False # merge the map (neural points) or not after the pgo (or we keep all the history neural points) don't merge them till the end, always keep those neural points in the memory until the end
        self.rehash_with_time: bool = True # Do the rehashing based on time difference or higher certainty

        # eval
        self.wandb_vis_on: bool = False
        self.silence: bool = False # with log or not
        self.o3d_vis_on: bool = True # visualize the mesh in-the-fly using o3d visualzier or not [press space to pasue/resume]
        self.o3d_vis_raw: bool = False # visualize the raw point cloud or the weight source point cloud
        self.eval_on: bool = False
        self.eval_outlier_thre = 0.5  # unit:m
        self.eval_freq_iters: int = 100
        self.vis_freq_iters: int = 100
        self.save_freq_iters: int = 100
        self.mesh_freq_frame: int = 10  # do the reconstruction per x frames
        self.sdfslice_freq_frame: int = 1 # do the sdf slice visulization per x frames
        self.vis_sdf_slice_v: bool = False
        self.sdf_slice_height: float = -1.0 # (m)

        self.eval_traj_align: bool = True # do the SE3 alignment of the trajectory when evaluating the absolute error
        
        # mesh reconstruction, marching cubes related
        self.mc_res_m: float = 0.1
        self.pad_voxel: int = 2
        self.skip_top_voxel: int = 2 

        self.mc_mask_on: bool = True # use mask for marching cubes to avoid the artifacts
        self.mc_local: bool = False # for the visualization of incremental mapping, we only show the local mesh sourrounding the current frame or not
        self.mesh_min_nn: int = 8  # The minimum number of the neighbor neural points for meshing, too small would cause some artifacts, too large would lead to lots of holes

        self.min_cluster_vertices: int = 200 # if a connected's vertices number is smaller than this value, it would get filtered (as a postprocessing)

        self.keep_local_mesh: bool = False # keep the local mesh in the visualizer or not (don't delete them could cause a too large memory consumption)
        
        self.infer_bs: int = 4096
        self.mesh_vis_on: bool = True
        self.mesh_vis_normal: bool = False
        self.vis_frame_axis_len: float = 0.8 # unit: m
        self.vis_point_size: int = 2

        self.sensor_cad_path = None # the path to the sensor cad file, "./cad/ipb_car.ply"

        self.save_map: bool = False # save the neural point map model and decoders or not
        self.save_merged_pc: bool = False # save the merged point cloud pc or not
        self.save_mesh: bool = False # save the reconstructed mesh map or not

        # ROS related 
        self.publish_np_map: bool = True # only for Rviz visualization
        self.publish_np_map_down_rate_list = [11, 23, 37, 53, 71, 89, 97, 113, 131, 151] # prime number list, downsampling for boosting pubishing speed 
        self.republish_raw_input: bool = False
        self.timeout_duration_s: int = 30 # in seconds


    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # common settings
        if "setting" in config_args:
            self.name = config_args["setting"].get("name", "pin_slam")
            
            self.output_root = config_args["setting"].get("output_root", "./experiments")
            self.pc_path = config_args["setting"].get("pc_path", "") 
            self.pose_path = config_args["setting"].get("pose_path", "")
            self.calib_path = config_args["setting"].get("calib_path", "")

            # optional, when semantic mapping is on [semantic]
            self.semantic_on = config_args["setting"].get("semantic_on", False) 
            if self.semantic_on:
                self.label_path = config_args["setting"].get("label_path", "./demo_data/labels")

            self.color_map_on = config_args["setting"].get("color_map_on", True)
            self.color_channel = config_args["setting"].get("color_channel", 0)
            if (self.color_channel == 1 or self.color_channel == 3) and self.color_map_on:
                self.color_on = True
            else:
                self.color_on = False

            self.load_model = config_args["setting"].get("load_model", False)
            if self.load_model:
                self.model_path = config_args["setting"].get("model_path", "")
            
            self.first_frame_ref = config_args["setting"].get("first_frame_ref", False)
            self.begin_frame = config_args["setting"].get("begin_frame", 0)
            self.end_frame = config_args["setting"].get("end_frame", 999999)
            self.every_frame = config_args["setting"].get("every_frame", 1)

            self.seed = config_args["setting"].get("random_seed", self.seed)
            self.device = config_args["setting"].get("device", "cuda") # or cpu, on cpu it's about 5 times slower 
            self.gpu_id = config_args["setting"].get("gpu_id", "0")

            self.kitti_correction_on = config_args["setting"].get("kitti_correct", False)
            if self.kitti_correction_on:
                self.correction_deg = config_args["setting"].get("correct_deg", self.correction_deg)

            self.deskew = config_args["setting"].get("deskew", False) # apply motion undistortion or not
            self.valid_ts_in_points = config_args["setting"].get("valid_ts", True)

        # process
        if "process" in config_args:
            self.min_range = config_args["process"].get("min_range_m", self.min_range)
            self.max_range = config_args["process"].get("max_range_m", self.max_range)
            self.min_z = config_args["process"].get("min_z_m", self.min_z)
            self.max_z = config_args["process"].get("max_z_m", self.max_z)
            
            self.rand_downsample = config_args["process"].get("rand_downsample", False)
            if self.rand_downsample:
                self.rand_down_r = config_args["process"].get("rand_down_r", self.rand_down_r)
            else:
                self.vox_down_m = config_args["process"].get("vox_down_m", self.max_range*1e-3)
            
            self.estimate_normal = config_args["process"].get("estimate_normal", False)
            self.dynamic_filter_on = config_args["process"].get("dynamic_filter_on", False)
            self.adaptive_range_on = config_args["process"].get("adaptive_range_on", False)

        # sampler
        if "sampler" in config_args:
            self.surface_sample_range_m = config_args["sampler"].get("surface_sample_range_m", self.vox_down_m*3.0) 
            self.free_sample_begin_ratio = config_args["sampler"].get("free_sample_begin_ratio", self.free_sample_begin_ratio)
            self.free_sample_end_dist_m = config_args["sampler"].get("free_sample_end_dist_m", self.surface_sample_range_m*4.0) # this value should be at least 2 times of surface_sample_range_m
            
            self.surface_sample_n = config_args["sampler"].get("surface_sample_n", self.surface_sample_n)
            self.free_front_n = config_args["sampler"].get("free_front_sample_n", self.free_front_n)
            self.free_behind_n = config_args["sampler"].get("free_behind_sample_n", self.free_behind_n)

        # neural point map
        if "neuralpoints" in config_args:
            self.voxel_size_m = config_args["neuralpoints"].get("voxel_size_m", self.vox_down_m*5.0)
            self.query_nn_k = config_args["neuralpoints"].get("query_nn_k", self.query_nn_k)

            self.num_nei_cells = config_args["neuralpoints"].get("num_nei_cells", self.num_nei_cells)
            self.search_alpha = config_args["neuralpoints"].get("search_alpha", self.search_alpha)
            
            self.feature_dim = config_args["neuralpoints"].get("feature_dim", self.feature_dim)

            # number of band for positional embedding, if 0, then there's no positional embedding
            self.use_gaussian_pe = config_args["neuralpoints"].get("pos_encoding_gaussian", False) 
            self.pos_encoding_band = config_args["neuralpoints"].get("pos_encoding_band", 0) # default without encoding

            # weighted the neighborhood feature before decoding to sdf or do the weighting of the decoded 
            # sdf afterwards, weighted first is faster, but may have some problem during the neural point map update after pgo
            self.weighted_first = config_args["neuralpoints"].get("weighted_first", self.weighted_first) 

            # build the neural point map from the surface samples or only the measurement points
            self.from_sample_points = config_args["neuralpoints"].get("from_sample_points", self.from_sample_points)
            if self.from_sample_points:
                self.map_surface_ratio = config_args["neuralpoints"].get("map_surface_ratio", self.map_surface_ratio)

            self.max_prune_certainty = config_args["neuralpoints"].get("max_prune_certainty", self.max_prune_certainty)
            self.use_mid_ts = config_args["neuralpoints"].get("use_mid_ts", self.use_mid_ts)

            self.local_map_travel_dist_ratio = config_args["neuralpoints"].get("local_map_travel_dist_ratio", self.local_map_travel_dist_ratio)

        # decoder
        if "decoder" in config_args: # only on if indicated
            # number of the level of the mlp decoder
            self.geo_mlp_level = config_args["decoder"].get("mlp_level", self.geo_mlp_level)
            # dimension of the mlp's hidden layer
            self.geo_mlp_hidden_dim = config_args["decoder"].get("mlp_hidden_dim", self.geo_mlp_hidden_dim) 
            # freeze the decoder after runing for x frames (used for incremental mapping to avoid forgeting)
            self.freeze_after_frame = config_args["decoder"].get("freeze_after_frame", self.freeze_after_frame)

        # TODO, now set to the same as geo mlp, but actually can be different
        self.color_mlp_level = self.geo_mlp_level
        self.color_mlp_hidden_dim = self.geo_mlp_hidden_dim

        self.sem_mlp_level = self.geo_mlp_level
        self.sem_mlp_hidden_dim = self.geo_mlp_hidden_dim

        # loss
        if "loss" in config_args:
            self.main_loss_type = config_args["loss"].get("main_loss_type", "bce")

            self.sigma_sigmoid_m = config_args["loss"].get("sigma_sigmoid_m", self.vox_down_m)

            self.loss_weight_on = config_args["loss"].get("loss_weight_on", self.loss_weight_on)

            if self.loss_weight_on:
                self.dist_weight_scale = config_args["loss"].get("dist_weight_scale", self.dist_weight_scale)
                # apply "behind the surface" loss weight drop-off or not
                self.behind_dropoff_on = config_args["loss"].get("behind_dropoff_on", self.behind_dropoff_on)
            
            self.ekional_loss_on = config_args["loss"].get("ekional_loss_on", self.ekional_loss_on) # use ekional loss (norm(gradient) = 1 loss)
            self.weight_e = float(config_args["loss"].get("weight_e", self.weight_e))

            self.numerical_grad = config_args["loss"].get("numerical_grad_on", self.numerical_grad)
            if not self.numerical_grad:
                self.gradient_decimation = 1
            else:
                self.gradient_decimation = config_args["loss"].get("grad_decimation", self.gradient_decimation)
                self.num_grad_step_ratio = config_args["loss"].get("num_grad_step_ratio", self.num_grad_step_ratio)

            self.consistency_loss_on = config_args["loss"].get("consistency_loss_on", self.consistency_loss_on)

        # rehersal (replay) based method
        if "continual" in config_args:
            self.pool_capacity = int(float(config_args["continual"].get("pool_capacity", self.pool_capacity)))
            self.bs_new_sample = int(config_args["continual"].get("batch_size_new_sample", 0))
            self.new_certainty_thre = float(config_args["continual"].get("new_certainty_thre", self.new_certainty_thre))
            self.pool_filter_freq = config_args["continual"].get("pool_filter_freq", 1)
        
        # tracker
        self.track_on = config_args.get("tracker", False) # only on if indicated
        if self.track_on:
            if self.color_on:
                self.photometric_loss_on = config_args["tracker"].get("photo_loss", self.photometric_loss_on)
                if self.photometric_loss_on:
                    self.photometric_loss_weight = float(config_args["tracker"].get("photo_weight", self.photometric_loss_weight))
                self.consist_wieght_on = config_args["tracker"].get("consist_wieght", self.consist_wieght_on)
            self.uniform_motion_on = config_args["tracker"].get("uniform_motion_on", self.uniform_motion_on)
            self.source_vox_down_m = config_args["tracker"].get("source_vox_down_m", self.vox_down_m*10)
            self.reg_iter_n = config_args["tracker"].get("iter_n", self.reg_iter_n)
            self.reg_min_grad_norm = config_args["tracker"].get("min_grad_norm", self.reg_min_grad_norm)
            self.reg_max_grad_norm = config_args["tracker"].get("max_grad_norm", self.reg_max_grad_norm)
            self.reg_GM_grad = config_args["tracker"].get("GM_grad", self.reg_GM_grad)
            self.reg_GM_dist_m = config_args["tracker"].get("GM_dist", self.reg_GM_dist_m)
            self.reg_lm_lambda = float(config_args["tracker"].get("lm_lambda", self.reg_lm_lambda))
            self.reg_term_thre_deg = float(config_args["tracker"].get("term_deg", self.reg_term_thre_deg))
            self.reg_term_thre_m = float(config_args["tracker"].get("term_m", self.reg_term_thre_m))
            self.eigenvalue_check = config_args["tracker"].get("eigenvalue_check", True)

        # pgo
        if self.track_on:
            self.pgo_on = config_args.get("pgo", False) # only on if indicated

        if self.pgo_on: 
            # loop detection mode
            self.use_gt_loop = config_args["pgo"].get("gt_loop", False) # only for debugging, not used for real cases
            self.local_map_context = config_args["pgo"].get("map_context", True)
            if self.local_map_context:
                self.loop_with_feature = config_args["pgo"].get("loop_with_feature", self.loop_with_feature)
            self.context_virtual_side_count = config_args["pgo"].get("virtual_side_count", self.context_virtual_side_count)
            self.pgo_freq = config_args["pgo"].get("pgo_freq_frame", self.pgo_freq)
            self.pgo_with_pose_prior = config_args["pgo"].get("with_pose_prior", self.pgo_with_pose_prior)
            # default cov (constant for all the edges)
            self.pgo_tran_std = config_args["pgo"].get("tran_std", self.pgo_tran_std)
            self.pgo_rot_std = config_args["pgo"].get("rot_std", self.pgo_rot_std)
            # use default or estimated cov
            self.use_reg_cov_mat = config_args["pgo"].get("use_reg_cov", False)
            # merge the neural point map or not after the loop
            # merge the map may lead to some holes
            self.pgo_merge_map = config_args["pgo"].get("merge_map", False) 
            self.context_cosdist_threshold = config_args["pgo"].get("context_cosdist", self.context_cosdist_threshold) 
            self.min_loop_travel_dist_ratio = config_args["pgo"].get("min_loop_travel_ratio", self.min_loop_travel_dist_ratio) 
            self.closed_pose_path = self.pose_path

        # mapping optimizer
        if "optimizer" in config_args:
            self.mapping_freq_frame = config_args["optimizer"].get("mapping_freq_frame", 1)
            self.iters = config_args["optimizer"].get("iters", self.iters) # mapping iters per frame
            self.init_iter_ratio = config_args["optimizer"].get("init_iter_ratio", self.init_iter_ratio) # iteration count ratio for the first frame (a kind of warm-up) #iter = init_iter_ratio*iter
            self.bs = config_args["optimizer"].get("batch_size", self.bs)
            self.lr = float(config_args["optimizer"].get("learning_rate", self.lr))

            # bundle adjustment
            self.ba_freq_frame = config_args["optimizer"].get("ba_freq_frame", 0) # default off
            self.ba_frame = config_args["optimizer"].get("ba_local_frame", 50)

            if self.ba_freq_frame > 0: # dirty fix to resolve the conflict
                self.stop_frame_thre = self.end_frame

        # vis and eval
        if "eval" in config_args:
            # use weight and bias to monitor the experiment or not
            self.wandb_vis_on = config_args["eval"].get("wandb_vis_on", False)
            
            self.silence = config_args["eval"].get("silence_log", self.silence)

            # turn on the open3d visualizer to visualize the mapping progress or not
            self.o3d_vis_on = config_args["eval"].get("o3d_vis_on", True)
            # path to the sensor cad file
            self.sensor_cad_path = config_args["eval"].get('sensor_cad_path', None)
            
            # frequency for mesh reconstruction for incremental mode (per x frame)
            self.mesh_freq_frame = config_args["eval"].get('mesh_freq_frame', self.mesh_freq_frame)
            # keep the previous reconstructed mesh in the visualizer or not
            self.keep_local_mesh = config_args["eval"].get('keep_local_mesh', self.keep_local_mesh)
            
            # frequency for sdf slice visualization for incremental mode (per x frame)
            self.sdfslice_freq_frame = config_args["eval"].get('sdf_freq_frame', 1)
            self.sdf_slice_height = config_args["eval"].get('sdf_slice_height', self.sdf_slice_height) # accoridng to sensor height
            
            # mesh masking
            self.mesh_min_nn = config_args["eval"].get('mesh_min_nn', self.mesh_min_nn)
            self.skip_top_voxel = config_args["eval"].get('skip_top_voxel', self.skip_top_voxel)
            self.min_cluster_vertices = config_args["eval"].get('min_cluster_vertices', self.min_cluster_vertices)
            
            # initial marching cubes grid sampling interval (unit: m)
            self.mc_res_m = config_args["eval"].get('mc_res_m', self.voxel_size_m)
            
            # save the map or not
            self.save_map = config_args["eval"].get('save_map', False)
            self.save_merged_pc = config_args["eval"].get('save_merged_pc', False)
            self.save_mesh = config_args["eval"].get('save_mesh', False)

        # associated parameters
        self.infer_bs = self.bs * 64
        self.consistency_count = int(self.bs / 4)

        self.window_radius = max(self.max_range, 6.0) # for the sampling data poo, should not be too small

        self.local_map_radius = self.max_range + 2.0 # for the local neural points
        # self.local_map_radius = 1.05 * self.max_range
        
        self.vis_frame_axis_len = self.max_range/50.0

        if self.local_map_context:
            self.local_map_context_latency = config_args["pgo"].get('local_map_latency', 10) # 10
            self.context_cosdist_threshold += 0.1
            if self.loop_with_feature:
                # self.context_shape = [10, 60]
                self.context_cosdist_threshold += 0.05
        else:
            self.loop_with_feature = False