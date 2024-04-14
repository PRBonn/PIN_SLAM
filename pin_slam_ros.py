#!/usr/bin/env python3
# @file      pin_slam_ros.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import os
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, TransformStamped
import nav_msgs.msg
from nav_msgs.msg import Odometry
import std_msgs.msg
from std_srvs.srv import Empty, EmptyResponse
import tf
import tf2_ros
import sys
import numpy as np
import wandb
import torch
from rich import print

from utils.config import Config
from utils.tools import *
from utils.loss import *
from utils.pgo import PoseGraphManager
from utils.loop_detector import NeuralPointMapContextManager, detect_local_loop
from utils.mesher import Mesher
from utils.tracker import Tracker
from utils.mapper import Mapper
from model.neural_points import NeuralPoints
from model.decoder import Decoder
from dataset.slam_dataset import SLAMDataset


'''
    📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency
     Y. Pan et al.
'''

class PINSLAMer:
    def __init__(self, config_path, point_cloud_topic, ts_field_name):

        rospy.init_node("pin_slam")
        print("[bold green]PIN-SLAM starts[/bold green]","📍" )

        self.global_frame_name = rospy.get_param('~global_frame_name', 'map') # odom 
        self.body_frame_name = rospy.get_param('~body_frame_name', "base_link") 
        self.sensor_frame_name = rospy.get_param('~sensor_frame_name', "range_sensor") # child
                                            
        self.config = Config()
        self.config.load(config_path)
        argv = ["pin_slam_ros.py", config_path, point_cloud_topic, ts_field_name]
        self.run_path = setup_experiment(self.config, argv)

        self.ts_field_name = ts_field_name
        
        # initialize the mlp decoder
        self.geo_mlp = Decoder(self.config, self.config.geo_mlp_hidden_dim, self.config.geo_mlp_level, 1)
        self.sem_mlp = Decoder(self.config, self.config.sem_mlp_hidden_dim, self.config.sem_mlp_level, self.config.sem_class_count + 1) if self.config.semantic_on else None
        self.color_mlp = Decoder(self.config, self.config.color_mlp_hidden_dim, self.config.color_mlp_level, self.config.color_channel) if self.config.color_on else None

        # initialize the feature octree
        self.neural_points = NeuralPoints(self.config)

        # Load the decoder model
        if self.config.load_model: # not used
            load_decoder(self.config, self.geo_mlp, self.sem_mlp, self.color_mlp)

        # dataset
        self.dataset = SLAMDataset(self.config)

        # odometry tracker
        self.tracker = Tracker(self.config, self.neural_points, self.geo_mlp, self.sem_mlp, self.color_mlp)

        # mapper
        self.mapper = Mapper(self.config, self.dataset, self.neural_points, self.geo_mlp, self.sem_mlp, self.color_mlp)

        # mesh reconstructor
        self.mesher = Mesher(self.config, self.neural_points, self.geo_mlp, self.sem_mlp, self.color_mlp)

        # pose graph manager (for back-end optimization) initialization
        self.pgm = PoseGraphManager(self.config)
        if self.config.pgo_on:     
            self.pgm.add_pose_prior(0, np.eye(4), fixed=True)

        # loop closure detector
        self.lcd_npmc = NeuralPointMapContextManager(self.config, self.mapper) # npmc: neural point map context
        self.loop_corrected = False
        self.loop_reg_failed_count = 0

        # service mesh
        self.mesh_min_nn = self.config.mesh_min_nn
        self.mc_res_m = self.config.mc_res_m

        # publisher
        queue_size_ = 10  # Replace with your actual queue size
        self.traj_pub = rospy.Publisher("~pin_path", nav_msgs.msg.Path, queue_size=queue_size_)
        self.path_msg = nav_msgs.msg.Path()
        self.path_msg.header.frame_id = self.global_frame_name
        self.odom_pub = rospy.Publisher("~odometry", Odometry, queue_size=queue_size_)
        self.frame_input_pub = rospy.Publisher("~frame/input", PointCloud2, queue_size=queue_size_)
        self.frame_map_pub = rospy.Publisher("~frame/mapping", PointCloud2, queue_size=queue_size_)
        self.frame_reg_pub = rospy.Publisher("~frame/registration", PointCloud2, queue_size=queue_size_)
        self.map_pub = rospy.Publisher("~map/neural_points", PointCloud2, queue_size=queue_size_)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # self.pose_pub = rospy.Publisher("~pose", PoseStamped, queue_size=100)
        # self.neural_points_pub = rospy.Publisher("~neural_points", PointCloud2, queue_size=10)

        self.last_message_time = time.time()
        self.begin = False
        
        # ros service
        rospy.Service('~save_results', Empty, self.save_slam_result_service_callback)
        rospy.Service('~save_mesh', Empty, self.save_mesh_service_callback)

        # for each frame
        rospy.Subscriber(point_cloud_topic, PointCloud2, self.frame_callback)

    def save_slam_result_service_callback(self, request):
        # Do something when the service is called
        rospy.loginfo("Service called, save results")
        self.save_results(terminate=False)

        return EmptyResponse()
    
    def save_mesh_service_callback(self, request):

        rospy.loginfo("Service called, save mesh")

        # update map bbx
        global_neural_pcd_down = self.neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23) # prime number
        self.dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
        
        mc_cm_str = str(round(self.mc_res_m*1e2))
        mesh_path = os.path.join(self.run_path, "mesh", 'mesh_frame_' + str(self.dataset.processed_frame) + "_" + mc_cm_str + "cm.ply")
        
        # figure out how to do it efficiently
        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, self.mc_res_m*300) # reconstruct in chunks
        cur_mesh = self.mesher.recon_aabb_collections_mesh(chunks_aabb, self.mc_res_m, mesh_path, False, self.config.semantic_on, self.config.color_on, filter_isolated_mesh=True, mesh_min_nn=self.mesh_min_nn)    

        return EmptyResponse()


    def frame_callback(self, msg):
        
        # I. Load data and preprocessing
        T0 = get_time()
        self.dataset.read_frame_ros(msg, ts_field_name=self.ts_field_name)

        T1 = get_time()
        self.dataset.preprocess_frame() 

        T2 = get_time()

        # II. Odometry
        if self.dataset.processed_frame > 0: 
            tracking_result = self.tracker.tracking(self.dataset.cur_source_points, self.dataset.cur_pose_guess_torch, 
                                                self.dataset.cur_source_colors, self.dataset.cur_source_normals)
            
            cur_pose_torch, _, _, valid_flag = tracking_result

            self.dataset.lose_track = not valid_flag 
            self.mapper.lose_track = not valid_flag

            self.dataset.update_odom_pose(cur_pose_torch) # update dataset.cur_pose_torch
            self.begin = True
        
        self.neural_points.travel_dist = torch.tensor(np.array(self.dataset.travel_dist), device = self.config.device, dtype=self.config.dtype) 
        
        T3 = get_time()

        # III. Loop detection and pgo 
        if self.config.pgo_on: 
            self.loop_corrected = self.detect_correct_loop()

        T4 = get_time()

        # IV: Mapping and bundle adjustment

        # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
        # if the robot stop, also don't process this frame, since there's no new oberservations
        if not self.mapper.lose_track and not self.dataset.stop_status:
            self.mapper.process_frame(self.dataset.cur_point_cloud_torch, self.dataset.cur_sem_labels_torch,
                                      self.dataset.cur_pose_torch, self.dataset.processed_frame)
        else: # lose track, still need to set back the local map
            self.neural_points.reset_local_map(self.dataset.cur_pose_torch[:3,3], None, self.dataset.processed_frame)
            self.mapper.static_mask = None
                                
        T5 = get_time()

        # for the first frame, we need more iterations to do the initialization (warm-up)
        cur_iter_num = self.config.iters * self.config.init_iter_ratio if self.dataset.processed_frame == 0 else self.config.iters
        if self.config.adaptive_iters and self.dataset.stop_status:
            cur_iter_num = max(1, cur_iter_num-10)
        if self.dataset.processed_frame == self.config.freeze_after_frame: # freeze the decoder after certain frame 
            freeze_decoders(self.geo_mlp, self.sem_mlp, self.color_mlp, self.config)
        
        # conduct local bundle adjustment (with lower frequency)
        if self.config.ba_freq_frame > 0 and (self.dataset.processed_frame+1) % self.config.ba_freq_frame == 0:
            self.mapper.bundle_adjustment(self.config.iters*4, self.config.ba_frame)

        # mapping with fixed poses (every frame)
        if self.dataset.processed_frame % self.config.mapping_freq_frame == 0:
            self.mapper.mapping(cur_iter_num)

        T6 = get_time()

        # publishing
        self.publish_msg(msg)

        T7 = get_time()

        if not self.config.silence:
            print("Frame (", self.dataset.processed_frame, ")")
            print("time for frame reading          (ms):", (T1-T0)*1e3)
            print("time for frame preprocessing    (ms):", (T2-T1)*1e3)
            print("time for odometry               (ms):", (T3-T2)*1e3)
            if self.config.pgo_on:
                print("time for loop detection and PGO (ms):", (T4-T3)*1e3)
            print("time for mapping preparation    (ms):", (T5-T4)*1e3)
            print("time for training               (ms):", (T6-T5)*1e3)
            print("time for publishing             (ms):", (T7-T6)*1e3)
        
        cur_frame_process_time = np.array([T2-T1, T3-T2, T5-T4, T6-T5, T4-T3]) # loop & pgo in the end, visualization and I/O time excluded
        self.dataset.time_table.append(cur_frame_process_time) # in s

        if self.config.wandb_vis_on:
            wandb_log_content = {'frame': self.dataset.processed_frame, 'timing(s)/preprocess': T2-T1, 'timing(s)/tracking': T3-T2, 'timing(s)/pgo': T4-T3, 'timing(s)/mapping': T6-T4} 
            wandb.log(wandb_log_content)

        self.dataset.processed_frame += 1
        self.last_message_time = time.time()

        # For Livox LIDAR, you need to Livox driver to convert the point cloud to the desired type

    def check_exit(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            # Check if the timeout has occurred
            delta_t_s = time.time() - self.last_message_time
            # print(delta_t_s)

            if delta_t_s > self.config.timeout_duration_s and self.begin:
                # Save results and stop the program
                self.save_results(terminate=True)
                rospy.signal_shutdown('Timeout reached. Save results and eiit.')
            rate.sleep()

    def save_results(self, terminate: bool = False):
        
        print("Mission completed")
        
        self.dataset.write_results(self.run_path)
        if self.config.pgo_on and self.pgm.pgo_count>0:
            print("# Loop corrected: ", self.pgm.pgo_count)
            self.pgm.write_g2o(os.path.join(self.run_path, "final_pose_graph.g2o"))
            self.pgm.plot_loops(os.path.join(self.run_path, "loop_plot.png"), vis_now=False)      

        if terminate:
            self.neural_points.recreate_hash(None, None, False, False) # merge the final neural point map
            self.neural_points.prune_map(self.config.max_prune_certainty) # prune uncertain points for the final output 

        if self.config.save_map:
            neural_pcd = self.neural_points.get_neural_points_o3d(query_global=True, color_mode=0)
            o3d.io.write_point_cloud(os.path.join(self.run_path, "map", "neural_points.ply"), neural_pcd)
            if terminate:
                self.neural_points.clear_temp() # clear temp data for output
            save_implicit_map(self.run_path, self.neural_points, self.geo_mlp, self.color_mlp, self.sem_mlp)

    def publish_msg(self, input_pc_msg):
        
        cur_pose = self.dataset.cur_pose_ref
        cur_q = tf.transformations.quaternion_from_matrix(cur_pose)
        cur_t = cur_pose[0:3,3]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.global_frame_name
        pose_msg.pose.orientation.x = cur_q[0]
        pose_msg.pose.orientation.y = cur_q[1]
        pose_msg.pose.orientation.z = cur_q[2]
        pose_msg.pose.orientation.w = cur_q[3]
        pose_msg.pose.position.x = cur_t[0]
        pose_msg.pose.position.y = cur_t[1]
        pose_msg.pose.position.z = cur_t[2]

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id =  self.sensor_frame_name
        odom_msg.pose.pose = pose_msg.pose
        self.odom_pub.publish(odom_msg)

        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = self.global_frame_name
        transform_msg.child_frame_id = self.sensor_frame_name
        transform_msg.transform.rotation.x = cur_q[0]
        transform_msg.transform.rotation.y = cur_q[1]
        transform_msg.transform.rotation.z = cur_q[2]
        transform_msg.transform.rotation.w = cur_q[3]
        transform_msg.transform.translation.x = cur_t[0]
        transform_msg.transform.translation.y = cur_t[1]
        transform_msg.transform.translation.z = cur_t[2]
        self.tf_broadcaster.sendTransform(transform_msg)

        self.path_msg.header.stamp = rospy.Time.now()

        self.path_msg.poses.append(pose_msg)

        if self.loop_corrected: # update traj after pgo
            self.path_msg.poses = []
            for cur_pose in self.dataset.pgo_poses:
                cur_q = tf.transformations.quaternion_from_matrix(cur_pose)
                cur_t = cur_pose[0:3,3]

                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = self.global_frame_name
                pose_msg.pose.orientation.x = cur_q[0]
                pose_msg.pose.orientation.y = cur_q[1]
                pose_msg.pose.orientation.z = cur_q[2]
                pose_msg.pose.orientation.w = cur_q[3]
                pose_msg.pose.position.x = cur_t[0]
                pose_msg.pose.position.y = cur_t[1]
                pose_msg.pose.position.z = cur_t[2]

                self.path_msg.poses.append(pose_msg)
        
        self.traj_pub.publish(self.path_msg) 

        # Publish point cloud
        fields_xyz = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
                
        # Neural Points
        if self.neural_points.neural_points is not None and self.config.publish_np_map:
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.global_frame_name 
            neural_point_count = self.neural_points.count()
            down_rate_level = neural_point_count // 500000
            down_rate_level = min(down_rate_level, len(self.config.publish_np_map_down_rate_list)-1)
            publish_np_map_down_rate = self.config.publish_np_map_down_rate_list[down_rate_level] 
            neural_points_np = self.neural_points.neural_points[::publish_np_map_down_rate].detach().cpu().numpy().astype(np.float32)
            # neural_points_feature_np = self.neural_points.geo_features[:-1,0:3].detach().cpu().numpy().astype(np.float32) # how to convert to rgb that we actually needed
            # neural_features_vis = F.normalize(neural_features_vis, p=2, dim=1)
            # TODO: add rgb (time or feature as rgb)

            neural_points_pc2_msg = pc2.create_cloud(header, fields_xyz, neural_points_np)
            self.map_pub.publish(neural_points_pc2_msg)

        # point cloud for mapping
        if self.dataset.cur_point_cloud_torch is not None:
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.sensor_frame_name 

            frame_mapping_np = self.dataset.cur_point_cloud_torch.detach().cpu().numpy().astype(np.float32)

            frame_mapping_pc2_msg = pc2.create_cloud(header, fields_xyz, frame_mapping_np)
            self.frame_map_pub.publish(frame_mapping_pc2_msg)

        # point cloud for registration
        if self.dataset.cur_source_points is not None:
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.sensor_frame_name 

            frame_registration_np = self.dataset.cur_source_points.detach().cpu().numpy().astype(np.float32)

            # TODO: add rgb (weight as rgb)

            frame_registration_pc2_msg = pc2.create_cloud(header, fields_xyz, frame_registration_np)
            self.frame_reg_pub.publish(frame_registration_pc2_msg)
        
        if self.config.republish_raw_input:
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.sensor_frame_name 
            input_pc_msg.header = header 

            self.frame_input_pub.publish(input_pc_msg)

    def detect_correct_loop(self):

        cur_frame_id = self.dataset.processed_frame
        if self.config.global_loop_on:
            if self.config.local_map_context and cur_frame_id >= self.config.local_map_context_latency: # local map context
                cur_frame = cur_frame_id-self.config.local_map_context_latency
                cur_pose = torch.tensor(self.dataset.pgo_poses[cur_frame], device=self.config.device, dtype=torch.float64)
                self.neural_points.reset_local_map(cur_pose[:3,3], None, cur_frame, False, self.config.loop_local_map_time_window) 
                context_pc_local = transform_torch(self.neural_points.local_neural_points.detach(), torch.linalg.inv(cur_pose)) # transformed back into the local frame
                neural_points_feature = self.neural_points.local_geo_features[:-1].detach() if self.config.loop_with_feature else None
                self.lcd_npmc.add_node(cur_frame, context_pc_local, neural_points_feature)
            else: # first frame not yet have local map, use scan context
                self.lcd_npmc.add_node(0, self.dataset.cur_point_cloud_torch)
            
        self.pgm.add_frame_node(cur_frame_id, self.dataset.pgo_poses[cur_frame_id]) # add new node and pose initial guess
        self.pgm.init_poses = self.dataset.pgo_poses

        if cur_frame_id > 0:     
            self.pgm.add_odometry_factor(cur_frame_id, cur_frame_id-1, self.dataset.last_odom_tran) # T_p<-c
            self.pgm.estimate_drift(self.dataset.travel_dist, cur_frame_id) # estimate the current drift
            if self.config.pgo_with_pose_prior: # add pose prior
                self.pgm.add_pose_prior(cur_frame_id, self.dataset.pgo_poses[cur_frame_id])

            if cur_frame_id - self.pgm.last_loop_idx > self.config.pgo_freq and not self.dataset.stop_status:
                cur_pgo_poses = np.stack(self.dataset.pgo_poses)
                dist_to_past = np.linalg.norm(cur_pgo_poses[:,:3,3] - cur_pgo_poses[-1,:3,3], axis=1)
                loop_candidate_mask = (self.dataset.travel_dist[-1] - self.dataset.travel_dist > self.config.min_loop_travel_dist_ratio*self.config.local_map_radius)
                loop_id = None
                local_map_context_loop = False
                if loop_candidate_mask.any(): # have at least one candidate
                    # firstly try to detect the local loop
                    loop_id, loop_dist, loop_transform = detect_local_loop(dist_to_past, loop_candidate_mask, self.dataset.pgo_poses, self.pgm.drift_radius, cur_frame_id, self.loop_reg_failed_count, dist_thre=self.config.voxel_size_m*5.0, silence=self.config.silence)
                    if loop_id is None and self.config.global_loop_on: # global loop detection (large drift)
                        loop_id, loop_cos_dist, loop_transform, local_map_context_loop = self.lcd_npmc.detect_global_loop(cur_pgo_poses, self.dataset.pgo_poses, self.pgm.drift_radius*3.0, loop_candidate_mask, self.neural_points)

                if loop_id is not None: # if a loop is found, we refine loop closure transform initial guess with a scan-to-map registration                    
                    if self.config.loop_z_check_on and abs(loop_transform[2,3]) > self.config.voxel_size_m*3.0: # for multi-floor buildings, z may cause ambiguilties
                        loop_id = None
                        if not self.config.silence:
                            print("[bold red]Delta z check failed, reject the loop[/bold red]")
                        return False 
                    pose_init_np = self.dataset.pgo_poses[loop_id] @ loop_transform # T_w<-c = T_w<-l @ T_l<-c 
                    pose_init_torch = torch.tensor(pose_init_np, device=self.config.device, dtype=torch.float64)
                    self.neural_points.recreate_hash(pose_init_torch[:3,3], None, True, True, loop_id) # recreate hash and local map for registration, this is the reason why we'd better to keep the duplicated neural points until the end
                    pose_refine_torch, _, _, reg_valid_flag = self.tracker.tracking(self.dataset.cur_source_points, pose_init_torch, loop_reg=True)
                    pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                    loop_transform = np.linalg.inv(self.dataset.pgo_poses[loop_id]) @ pose_refine_np # T_l<-c = T_l<-w @ T_w<-c
                    if not self.config.silence:
                        print("[bold green]Refine loop transformation succeed [/bold green]")
                    # only conduct pgo when the loop and loop constraint is correct
                    if reg_valid_flag: # refine succeed
                        reg_valid_flag = self.pgm.add_loop_factor(cur_frame_id, loop_id, loop_transform)
                    if reg_valid_flag:   
                        self.pgm.optimize_pose_graph() # conduct pgo
                        cur_loop_vis_id = cur_frame_id-self.config.local_map_context_latency if local_map_context_loop else cur_frame_id
                        self.pgm.loop_edges.append(np.array([loop_id, cur_loop_vis_id],dtype=np.uint32)) # only for vis
                        # update the neural points and poses
                        pose_diff_torch = torch.tensor(self.pgm.get_pose_diff(), device=self.config.device, dtype=self.config.dtype)
                        self.dataset.cur_pose_torch = torch.tensor(self.pgm.cur_pose, device=self.config.device, dtype=self.config.dtype)
                        self.neural_points.adjust_map(pose_diff_torch)
                        self.neural_points.recreate_hash(self.dataset.cur_pose_torch[:3,3], None, (not self.config.pgo_merge_map), self.config.rehash_with_time, cur_frame_id) # recreate hash from current time
                        self.mapper.transform_data_pool(pose_diff_torch) # transform global pool
                        self.dataset.update_poses_after_pgo(self.pgm.cur_pose, self.pgm.pgo_poses)
                        self.pgm.last_loop_idx = cur_frame_id
                        self.pgm.min_loop_idx = min(self.pgm.min_loop_idx, loop_id)
                        self.loop_reg_failed_count = 0
                        return True
                    else:
                        if not self.config.silence:
                            print("[bold red]Registration failed, reject the loop candidate [/bold red]")
                        self.neural_points.recreate_hash(self.dataset.cur_pose_torch[:3,3], None, True, True, cur_frame_id) # if failed, you need to reset the local map back to current frame
                        self.loop_reg_failed_count += 1

        return False
   
                    
if __name__ == "__main__":

    config_path = rospy.get_param('~config_path', "./config/lidar_slam/run_ncd_128.yaml")
    point_cloud_topic = rospy.get_param('~point_cloud_topic', "/os_cloud_node/points")
    ts_field_name = rospy.get_param('~point_timestamp_field_name', "time")

    # If you would like to directly run the python script without including it in a ROS package
    # python pin_slam_ros_node.py [path_to_your_config_file] [point_cloud_topic]
    print("If you would like to directly run the python script without including it in a ROS package\n\
           python pin_slam_ros_node.py (path_to_your_config_file) (point_cloud_topic) (point_timestamp_field_name)")

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    if len(sys.argv) > 2:
        point_cloud_topic = sys.argv[2]

    if len(sys.argv) > 3:
        ts_field_name = sys.argv[3]

    slamer = PINSLAMer(config_path, point_cloud_topic, ts_field_name)
    slamer.check_exit()




