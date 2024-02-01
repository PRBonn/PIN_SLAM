# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [SHINEMapping] by Yue Pan.

import open3d as o3d
import numpy as np

def eval_mesh(file_pred, file_trgt, down_sample_res=0.02, threshold=0.05, truncation_acc=0.50, truncation_com=0.50, gt_bbx_mask_on= True, 
              mesh_sample_point=10000000, possion_sample_init_factor=5):
    """ Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction (should be mesh)
        file_trgt: file path of target (shoud be point cloud)
        down_sample_res: use voxel_downsample to uniformly sample mesh points
        threshold: distance threshold used to compute precision/recall
        truncation_acc: points whose nearest neighbor is farther than the distance would not be taken into account (take pred as reference)
        truncation_com: points whose nearest neighbor is farther than the distance would not be taken into account (take trgt as reference)
        gt_bbx_mask_on: use the bounding box of the trgt as a mask of the pred mesh
        mesh_sample_point: number of the sampling points from the mesh
        possion_sample_init_factor: used for possion uniform sampling, check open3d for more details (deprecated)
    Returns:

    Returns:
        Dict of mesh metrics (chamfer distance, precision, recall, f1 score, etc.)
    """

    mesh_pred = o3d.io.read_triangle_mesh(file_pred)

    pcd_trgt = o3d.io.read_point_cloud(file_trgt)

    # (optional) filter the prediction outside the gt bounding box (since gt sometimes is not complete enough)
    if gt_bbx_mask_on: 
        trgt_bbx = pcd_trgt.get_axis_aligned_bounding_box()
        min_bound = trgt_bbx.get_min_bound()
        min_bound[2]-=down_sample_res
        max_bound = trgt_bbx.get_max_bound()
        max_bound[2]+=down_sample_res
        trgt_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound) 
        mesh_pred = mesh_pred.crop(trgt_bbx)
        # pcd_sample_pred = pcd_sample_pred.crop(trgt_bbx)

    # pcd_sample_pred = mesh_pred.sample_points_poisson_disk(number_of_points=mesh_sample_point, init_factor=possion_sample_init_factor)
    # mesh uniform sampling
    pcd_sample_pred = mesh_pred.sample_points_uniformly(number_of_points=mesh_sample_point)

    if down_sample_res > 0:
        pred_pt_count_before = len(pcd_sample_pred.points)
        pcd_pred = pcd_sample_pred.voxel_down_sample(down_sample_res)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample_res)
        pred_pt_count_after = len(pcd_pred.points)
        print("Predicted mesh unifrom sample: ", pred_pt_count_before, " --> ", pred_pt_count_after, " (", down_sample_res, "m)")
    
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist_p = nn_correspondance(verts_trgt, verts_pred, truncation_acc, True) # find nn in ground truth samples for each predict sample -> precision related
    _, dist_r = nn_correspondance(verts_pred, verts_trgt, truncation_com, False) # find nn in predict samples for each ground truth sample -> recall related
    
    dist_p = np.array(dist_p)
    dist_r = np.array(dist_r)

    dist_p_s = np.square(dist_p)
    dist_r_s = np.square(dist_r)

    dist_p_mean = np.mean(dist_p)
    dist_r_mean = np.mean(dist_r) 

    dist_p_s_mean = np.mean(dist_p_s)
    dist_r_s_mean = np.mean(dist_r_s) 

    chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)
    chamfer_l2 = np.sqrt(0.5 * (dist_p_s_mean + dist_r_s_mean))

    precision = np.mean((dist_p < threshold).astype('float')) * 100.0 # %
    recall = np.mean((dist_r < threshold).astype('float')) * 100.0 # %
    fscore = 2 * precision * recall / (precision + recall) # %
    
    metrics = {'MAE_accuracy (m)': dist_p_mean,
               'MAE_completeness (m)': dist_r_mean,
               'Chamfer_L1 (m)': chamfer_l1,
               'Chamfer_L2 (m)': chamfer_l2, 
               'Precision [Accuracy] (%)': precision, 
               'Recall [Completeness] (%)': recall,
               'F-score (%)': fscore, 
               'Spacing (m)': down_sample_res,  # evlaution setup
               'Inlier_threshold (m)': threshold,  # evlaution setup
               'Outlier_truncation_acc (m)': truncation_acc, # evlaution setup
               'Outlier_truncation_com (m)': truncation_com  # evlaution setup
               }
    return metrics


def nn_correspondance(verts1, verts2, truncation_dist, ignore_outlier=True):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
        scalar truncation_dist: points whose nearest neighbor is farther than the distance would not be taken into account
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    truncation_dist_square = truncation_dist**2

    for vert in verts2:
        _, inds, dist_square = kdtree.search_knn_vector_3d(vert, 1)
        
        if dist_square[0] < truncation_dist_square:
            indices.append(inds[0])
            distances.append(np.sqrt(dist_square[0]))
        else:
            if not ignore_outlier:
                indices.append(inds[0])
                distances.append(truncation_dist)

    return indices, distances


def eval_depth(depth_pred, depth_trgt):
    """ Computes 2d metrics between two depth maps
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth
    Returns:
        Dict of metrics
    """
    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25 ** 2).astype('float')
    r3 = (thresh < 1.25 ** 3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics

def crop_intersection(file_gt, files_pred, out_file_crop, dist_thre=0.1, mesh_sample_point=1000000):
    """ Get the cropped ground truth point cloud according to the intersection of the predicted
    mesh by different methods
    Args:
        file_gt: file path of the ground truth (shoud be point cloud)
        files_pred: a list of the paths of different methods's reconstruction (shoud be mesh)
        out_file_crop: output path of the cropped ground truth point cloud
        dist_thre: nearest neighbor distance threshold in meter
        mesh_sample_point: number of the sampling points from the mesh
    """
    print("Load the original ground truth point cloud from:", file_gt)
    pcd_gt = o3d.io.read_point_cloud(file_gt)
    pcd_gt_pts = np.asarray(pcd_gt.points)
    dist_square_thre = dist_thre**2
    for i in range(len(files_pred)):
        cur_file_pred = files_pred[i]
        print("Process", cur_file_pred)
        cur_mesh_pred = o3d.io.read_triangle_mesh(cur_file_pred)

        cur_sample_pred = cur_mesh_pred.sample_points_uniformly(number_of_points=mesh_sample_point)
        
        cur_kdtree = o3d.geometry.KDTreeFlann(cur_sample_pred)
        
        crop_pcd_gt_pts = []
        for pt in pcd_gt_pts:
            _, _, dist_square = cur_kdtree.search_knn_vector_3d(pt, 1)
            
            if dist_square[0] < dist_square_thre:
                crop_pcd_gt_pts.append(pt)
        
        pcd_gt_pts = np.asarray(crop_pcd_gt_pts, dtype=np.float64)

    crop_pcd_gt = o3d.geometry.PointCloud()
    crop_pcd_gt.points = o3d.utility.Vector3dVector(pcd_gt_pts)
    
    print("Output the croped ground truth to:", out_file_crop)
    o3d.io.write_point_cloud(out_file_crop, crop_pcd_gt)

    
