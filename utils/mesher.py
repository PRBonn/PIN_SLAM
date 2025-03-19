#!/usr/bin/env python3
# @file      mesher.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import math

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import skimage.measure
import torch
from tqdm import tqdm

from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.semantic_kitti_utils import sem_kitti_color_map
from utils.tools import remove_gpu_cache

class Mesher:
    def __init__(
        self,
        config: Config,
        neural_points: NeuralPoints,
        decoders: dict,
    ):

        self.config = config
        self.silence = config.silence
        self.neural_points = neural_points
        self.sdf_mlp = decoders["sdf"]
        self.sem_mlp = decoders["semantic"]
        self.color_mlp = decoders["color"]
        self.device = config.device
        self.cur_device = self.device
        self.dtype = config.dtype
        self.global_transform = np.eye(4)

    def query_points(
        self,
        coord,
        bs,
        query_sdf=True,
        query_sem=False,
        query_color=False,
        query_mask=True,
        query_locally=False,
        mask_min_nn_count: int = 4,
        out_torch: bool = False,
    ):
        """query the sdf value, semantic label and marching cubes mask for points
        Args:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            bs: batch size for the inference
        Returns:
            sdf_pred: Ndim numpy array or torch tensor, signed distance value (scaled) at each query point
            sem_pred: Ndim numpy array or torch tenosr, semantic label prediction at each query point
            mc_mask:  Ndim bool numpy array or torch tensor, marching cubes mask at each query point
        """
        # the coord torch tensor is already scaled in the [-1,1] coordinate system
        sample_count = coord.shape[0]
        iter_n = math.ceil(sample_count / bs)
        if query_sdf:
            if out_torch:
                sdf_pred = torch.zeros(sample_count)
            else:
                sdf_pred = np.zeros(sample_count)
        else:
            sdf_pred = None
        if query_sem:
            if out_torch:
                sem_pred = torch.zeros(sample_count)
            else:
                sem_pred = np.zeros(sample_count)
        else:
            sem_pred = None
        if query_color:
            if out_torch:
                color_pred = torch.zeros((sample_count, self.config.color_channel))
            else:
                color_pred = np.zeros((sample_count, self.config.color_channel))
        else:
            color_pred = None
        if query_mask:
            if out_torch:
                mc_mask = torch.zeros(sample_count)
            else:
                mc_mask = np.zeros(sample_count)
        else:
            mc_mask = None

        with torch.no_grad():  # eval step
            for n in tqdm(range(iter_n), disable=self.silence):
                head = n * bs
                tail = min((n + 1) * bs, sample_count)
                batch_coord = coord[head:tail, :]
                batch_size = batch_coord.shape[0]
                if self.cur_device == "cpu" and self.device == "cuda":
                    batch_coord = batch_coord.cuda()
                (
                    batch_geo_feature,
                    batch_color_feature,
                    weight_knn,
                    nn_count,
                    _,
                ) = self.neural_points.query_feature(
                    batch_coord,
                    training_mode=False,
                    query_locally=query_locally,  # inference mode, query globally
                    query_geo_feature=query_sdf or query_sem,
                    query_color_feature=query_color,
                )

                pred_mask = nn_count >= 1  # only query sdf here
                if query_sdf:
                    if self.config.weighted_first:
                        batch_sdf = torch.zeros(batch_size, device=self.device)
                    else:
                        batch_sdf = torch.zeros(
                            batch_size,
                            batch_geo_feature.shape[1],
                            1,
                            device=self.device,
                        )
                    # predict the sdf with the feature, only do for the unmasked part (not in the unknown freespace)
                    batch_sdf[pred_mask] = self.sdf_mlp.sdf(
                        batch_geo_feature[pred_mask]
                    )

                    if not self.config.weighted_first:
                        batch_sdf = torch.sum(batch_sdf * weight_knn, dim=1).squeeze(1)
                    if out_torch:
                        sdf_pred[head:tail] = batch_sdf.detach()
                    else:
                        sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()
                if query_sem:
                    batch_sem_prob = self.sem_mlp.sem_label_prob(batch_geo_feature)
                    if not self.config.weighted_first:
                        batch_sem_prob = torch.sum(batch_sem_prob * weight_knn, dim=1)
                    batch_sem = torch.argmax(batch_sem_prob, dim=1)
                    if out_torch:
                        sem_pred[head:tail] = batch_sem.detach()
                    else:
                        sem_pred[head:tail] = batch_sem.detach().cpu().numpy()
                if query_color:
                    batch_color = self.color_mlp.regress_color(batch_color_feature)
                    if not self.config.weighted_first:
                        batch_color = torch.sum(batch_color * weight_knn, dim=1)  # N, C
                    if out_torch:
                        color_pred[head:tail] = batch_color.detach()
                    else:
                        color_pred[head:tail] = (
                            batch_color.detach().cpu().numpy().astype(dtype=np.float64)
                        )
                if query_mask:
                    # do marching cubes only when there are at least K near neural points
                    mask_mc = nn_count >= mask_min_nn_count
                    if out_torch:
                        mc_mask[head:tail] = mask_mc.detach()
                    else:
                        mc_mask[head:tail] = mask_mc.detach().cpu().numpy()

        return sdf_pred, sem_pred, color_pred, mc_mask

    def get_query_from_bbx(self, bbx, voxel_size, pad_voxel=0, skip_top_voxel=0):
        """
        get grid query points inside a given bounding box (bbx)
        Args:
            bbx: open3d bounding box, in world coordinate system, with unit m
            voxel_size: scalar, marching cubes voxel size with unit m
        Returns:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            voxel_num_xyz: 3dim numpy array, the number of voxels on each axis for the bbx
            voxel_origin: 3dim numpy array the coordinate of the bottom-left corner of the 3d grids
                for marching cubes, in world coordinate system with unit m
        """
        # bbx and voxel_size are all in the world coordinate system
        min_bound = bbx.get_min_bound()
        max_bound = bbx.get_max_bound()
        len_xyz = max_bound - min_bound
        voxel_num_xyz = (np.ceil(len_xyz / voxel_size) + pad_voxel * 2).astype(np.int_)
        voxel_origin = min_bound - pad_voxel * voxel_size
        # pad an additional voxel underground to gurantee the reconstruction of ground
        voxel_origin[2] -= voxel_size
        voxel_num_xyz[2] += 1
        voxel_num_xyz[2] -= skip_top_voxel

        voxel_count_total = voxel_num_xyz[0] * voxel_num_xyz[1] * voxel_num_xyz[2]
        if voxel_count_total > 5e8:  # this value is determined by your gpu memory
            print("too many query points, use smaller chunks")
            return None, None, None
            # self.cur_device = "cpu" # firstly save in cpu memory (which would be larger than gpu's)
            # print("too much query points, use cpu memory")
        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16, device=self.cur_device)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16, device=self.cur_device)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16, device=self.cur_device)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        # get the vector of all the grid point's 3D coordinates
        coord = (
            torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        )
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype, device=self.cur_device)

        return coord, voxel_num_xyz, voxel_origin

    def get_query_from_hor_slice(self, bbx, slice_z, voxel_size):
        """
        get grid query points inside a given bounding box (bbx) at slice height (slice_z)
        """
        # bbx and voxel_size are all in the world coordinate system
        min_bound = bbx.get_min_bound()
        max_bound = bbx.get_max_bound()
        len_xyz = max_bound - min_bound
        voxel_num_xyz = (np.ceil(len_xyz / voxel_size)).astype(np.int_)
        voxel_num_xyz[2] = 1
        voxel_origin = min_bound
        voxel_origin[2] = slice_z

        query_count_total = voxel_num_xyz[0] * voxel_num_xyz[1]
        if query_count_total > 1e8:  # avoid gpu memory issue, dirty fix
            self.cur_device = (
                "cpu"  # firstly save in cpu memory (which would be larger than gpu's)
            )
            print("too much query points, use cpu memory")
        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16, device=self.cur_device)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16, device=self.cur_device)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16, device=self.cur_device)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        # get the vector of all the grid point's 3D coordinates
        coord = (
            torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        )
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype, device=self.cur_device)

        return coord, voxel_num_xyz, voxel_origin

    def get_query_from_ver_slice(self, bbx, slice_x, voxel_size):
        """
        get grid query points inside a given bounding box (bbx) at slice position (slice_x)
        """
        # bbx and voxel_size are all in the world coordinate system
        min_bound = bbx.get_min_bound()
        max_bound = bbx.get_max_bound()
        len_xyz = max_bound - min_bound
        voxel_num_xyz = (np.ceil(len_xyz / voxel_size)).astype(np.int_)
        voxel_num_xyz[0] = 1
        voxel_origin = min_bound
        voxel_origin[0] = slice_x

        query_count_total = voxel_num_xyz[1] * voxel_num_xyz[2]
        if query_count_total > 1e8:  # avoid gpu memory issue, dirty fix
            self.cur_device = (
                "cpu"  # firstly save in cpu memory (which would be larger than gpu's)
            )
            print("too much query points, use cpu memory")
        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16, device=self.cur_device)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16, device=self.cur_device)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16, device=self.cur_device)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        # get the vector of all the grid point's 3D coordinates
        coord = (
            torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        )
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype, device=self.cur_device)

        return coord, voxel_num_xyz, voxel_origin

    def generate_sdf_map(self, coord, sdf_pred, mc_mask):
        """
        Generate the SDF map for saving
        """
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        sdf_map_pc = o3d.t.geometry.PointCloud(device)

        coord_np = coord.detach().cpu().numpy()

        # the sdf (unit: m) would be saved in the intensity channel
        sdf_map_pc.point["positions"] = o3d.core.Tensor(coord_np, dtype, device)
        sdf_map_pc.point["intensities"] = o3d.core.Tensor(
            np.expand_dims(sdf_pred, axis=1), dtype, device
        )  # scaled sdf prediction
        if mc_mask is not None:
            # the marching cubes mask would be saved in the labels channel
            sdf_map_pc.point["labels"] = o3d.core.Tensor(
                np.expand_dims(mc_mask, axis=1), o3d.core.int32, device
            )  # mask

        # global transform (to world coordinate system) before output
        if not np.array_equal(self.global_transform, np.eye(4)):
            sdf_map_pc.transform(self.global_transform)

        return sdf_map_pc

    def generate_sdf_map_for_vis(
        self, coord, sdf_pred, mc_mask, min_sdf=-1.0, max_sdf=1.0, cmap="bwr"
    ):  # 'jet','bwr','viridis'
        """
        Generate the SDF map for visualization
        """
        # do the masking or not
        if mc_mask is not None:
            coord = coord[mc_mask > 0]
            sdf_pred = sdf_pred[mc_mask > 0]

        coord_np = coord.detach().cpu().numpy().astype(np.float64)

        sdf_pred_show = np.clip((sdf_pred - min_sdf) / (max_sdf - min_sdf), 0.0, 1.0)

        color_map = cm.get_cmap(cmap)  # or 'jet'
        colors = color_map(1.0 - sdf_pred_show)[:, :3].astype(np.float64) # change to blue (+) --> red (-)

        sdf_map_pc = o3d.geometry.PointCloud()
        sdf_map_pc.points = o3d.utility.Vector3dVector(coord_np)
        sdf_map_pc.colors = o3d.utility.Vector3dVector(colors)
        if not np.array_equal(self.global_transform, np.eye(4)):
            sdf_map_pc.transform(self.global_transform)

        return sdf_map_pc

    def assign_to_bbx(self, sdf_pred, sem_pred, color_pred, mc_mask, voxel_num_xyz):
        """assign the queried sdf, semantic label and marching cubes mask back to the 3D grids in the specified bounding box
        Args:
            sdf_pred: Ndim np.array/torch.tensor
            sem_pred: Ndim np.array/torch.tensor
            mc_mask:  Ndim bool np.array/torch.tensor
            voxel_num_xyz: 3dim numpy array/torch.tensor, the number of voxels on each axis for the bbx
        Returns:
            sdf_pred:  a*b*c np.array/torch.tensor, 3d grids of sign distance values
            sem_pred:  a*b*c np.array/torch.tensor, 3d grids of semantic labels
            mc_mask:   a*b*c np.array/torch.tensor, 3d grids of marching cube masks, marching cubes only on where
                the mask is true
        """
        if sdf_pred is not None:
            sdf_pred = sdf_pred.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        if sem_pred is not None:
            sem_pred = sem_pred.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        if color_pred is not None:
            color_pred = color_pred.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        if mc_mask is not None:
            mc_mask = mc_mask.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        return sdf_pred, sem_pred, color_pred, mc_mask

    def mc_mesh(self, mc_sdf, mc_mask, voxel_size, mc_origin):
        """use the marching cubes algorithm to get mesh vertices and faces
        Args:
            mc_sdf:  a*b*c np.array, 3d grids of sign distance values
            mc_mask: a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where
                the mask is true
            voxel_size: scalar, marching cubes voxel size with unit m
            mc_origin: 3*1 np.array, the coordinate of the bottom-left corner of the 3d grids for
                marching cubes, in world coordinate system with unit m
        Returns:
            ([verts], [faces]), mesh vertices and triangle faces
        """
        if not self.silence:
            print("Marching cubes ...")
        # the input are all already numpy arraies
        verts, faces = np.zeros((0, 3)), np.zeros((0, 3))
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                mc_sdf, level=0.0, allow_degenerate=False, mask=mc_mask
            )
            #  Whether to allow degenerate (i.e. zero-area) triangles in the
            # end-result. Default True. If False, degenerate triangles are
            # removed, at the cost of making the algorithm slower.
        except:
            pass

        verts = mc_origin + verts * voxel_size

        return verts, faces

    def estimate_vertices_sem(self, mesh, verts, filter_free_space_vertices=True):
        """
        Predict the semantic label of the vertices
        """
        if len(verts) == 0:
            return mesh

        # print("predict semantic labels of the vertices")
        verts_torch = torch.tensor(verts, dtype=self.dtype, device=self.device)
        _, verts_sem, _, _ = self.query_points(
            verts_torch, self.config.infer_bs, False, True, False, False
        )
        verts_sem_list = list(verts_sem)
        verts_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in verts_sem_list]
        verts_sem_rgb = np.asarray(verts_sem_rgb, dtype=np.float64) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(verts_sem_rgb)

        # filter the freespace vertices
        if filter_free_space_vertices:
            non_freespace_idx = verts_sem <= 0
            mesh.remove_vertices_by_mask(non_freespace_idx)

        return mesh

    def estimate_vertices_color(self, mesh, verts):
        """
        Predict the color of the vertices
        """
        if len(verts) == 0:
            return mesh

        # print("predict color labels of the vertices")
        verts_torch = torch.tensor(verts, dtype=self.dtype, device=self.device)
        _, _, verts_color, _ = self.query_points(
            verts_torch, self.config.infer_bs, False, False, True, False
        )

        if self.config.color_channel == 1:
            verts_color = np.repeat(verts_color * 2.0, 3, axis=1)

        mesh.vertex_colors = o3d.utility.Vector3dVector(verts_color)

        return mesh

    def filter_isolated_vertices(self, mesh, filter_cluster_min_tri=300):
        """
        Cluster connected triangles and remove the small clusters
        """
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        # print("Remove the small clusters")
        triangles_to_remove = (
            cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
        )
        mesh.remove_triangles_by_mask(triangles_to_remove)

        return mesh

    def generate_bbx_sdf_hor_slice(
        self, bbx, slice_z, voxel_size, query_locally=False, min_sdf=-1.0, max_sdf=1.0, mask_min_nn_count=5
    ):
        """
        Generate the SDF slice at height (slice_z)
        """
        # print("Generate the SDF slice at heright %.2f (m)" % (slice_z))
        coord, _, _ = self.get_query_from_hor_slice(bbx, slice_z, voxel_size)
        sdf_pred, _, _, mc_mask = self.query_points(
            coord,
            self.config.infer_bs,
            True,
            False,
            False,
            self.config.mc_mask_on,
            query_locally=query_locally,
            mask_min_nn_count=mask_min_nn_count,
        )
        sdf_map_pc = self.generate_sdf_map_for_vis(
            coord, sdf_pred, mc_mask, min_sdf, max_sdf
        )

        return sdf_map_pc

    def generate_bbx_sdf_ver_slice(
        self, bbx, slice_x, voxel_size, query_locally=False, min_sdf=-1.0, max_sdf=1.0, mask_min_nn_count=5
    ):
        """
        Generate the SDF slice at x position (slice_x)
        """
        # print("Generate the SDF slice at x position %.2f (m)" % (slice_x))
        coord, _, _ = self.get_query_from_ver_slice(bbx, slice_x, voxel_size)
        sdf_pred, _, _, mc_mask = self.query_points(
            coord,
            self.config.infer_bs,
            True,
            False,
            False,
            self.config.mc_mask_on,
            query_locally=query_locally,
            mask_min_nn_count=mask_min_nn_count,
        )
        sdf_map_pc = self.generate_sdf_map_for_vis(
            coord, sdf_pred, mc_mask, min_sdf, max_sdf
        )

        return sdf_map_pc

    # reconstruct the mesh from a the map defined by a collection of bounding boxes
    def recon_aabb_collections_mesh(
        self,
        aabbs,
        voxel_size,
        mesh_path=None,
        query_locally=False,
        estimate_sem=False,
        estimate_color=False,
        mesh_normal=True,
        filter_isolated_mesh=False,
        filter_free_space_vertices=True,
        mesh_min_nn=10,
        use_torch_mc=False,
    ):
        """
        Reconstruct the mesh from a collection of bounding boxes
        """
        if not self.silence:
            print("# Chunk for meshing: ", len(aabbs))
            
        mesh_merged = o3d.geometry.TriangleMesh()
        for bbx in tqdm(aabbs, disable=self.silence):
            cur_mesh = self.recon_aabb_mesh(
                bbx,
                voxel_size,
                None,
                query_locally,
                estimate_sem,
                estimate_color,
                mesh_normal,
                filter_isolated_mesh,
                filter_free_space_vertices,
                mesh_min_nn,
                use_torch_mc,
            )
            mesh_merged += cur_mesh

            remove_gpu_cache() # deal with high GPU memory consumption when meshing (TODO)

        mesh_merged.remove_duplicated_vertices()

        if mesh_normal:
            mesh_merged.compute_vertex_normals()

        if mesh_path is not None:
            o3d.io.write_triangle_mesh(mesh_path, mesh_merged)
            if not self.silence:
                print("save the mesh to %s\n" % (mesh_path))

        return mesh_merged

    def recon_aabb_mesh(
        self,
        bbx,
        voxel_size,
        mesh_path=None,
        query_locally=False,
        estimate_sem=False,
        estimate_color=False,
        mesh_normal=True,
        filter_isolated_mesh=False,
        filter_free_space_vertices=True,
        mesh_min_nn=10,
        use_torch_mc=False,
    ):
        """
        Reconstruct the mesh from a given bounding box
        """
        # reconstruct and save the (semantic) mesh from the feature octree the decoders within a
        # given bounding box.  bbx and voxel_size all with unit m, in world coordinate system
        coord, voxel_num_xyz, voxel_origin = self.get_query_from_bbx(
            bbx, voxel_size, self.config.pad_voxel, self.config.skip_top_voxel
        )
        if coord is None:  # use chunks in this case
            return None

        sdf_pred, _, _, mc_mask = self.query_points(
            coord,
            self.config.infer_bs,
            True,
            False,
            False,
            self.config.mc_mask_on,
            query_locally,
            mesh_min_nn,
            out_torch=use_torch_mc,
        )

        mc_sdf, _, _, mc_mask = self.assign_to_bbx(
            sdf_pred, None, None, mc_mask, voxel_num_xyz
        )
        if use_torch_mc:
            # torch version
            verts, faces = self.mc_mesh_torch(
                mc_sdf, mc_mask, voxel_size, torch.tensor(voxel_origin).to(mc_sdf)
            )  # has some double faces problem
            mesh = o3d.t.geometry.TriangleMesh(device=o3d.core.Device("cuda:0"))
            mesh.vertex.positions = o3d.core.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(verts)
            )
            mesh.triangle.indices = o3d.core.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(faces)
            )
            mesh = mesh.to_legacy()
            mesh.remove_duplicated_vertices()
            mesh.compute_vertex_normals()
        else:
            # np cpu version
            verts, faces = self.mc_mesh(
                mc_sdf, mc_mask.astype(bool), voxel_size, voxel_origin
            )  # too slow ? (actually not, the slower part is the querying)
            # directly use open3d to get mesh
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts.astype(np.float64)),
                o3d.utility.Vector3iVector(faces),
            )

        # if not self.silence:
        #     print("Marching cubes done")

        if estimate_sem:
            mesh = self.estimate_vertices_sem(mesh, verts, filter_free_space_vertices)
        else:
            if estimate_color:
                mesh = self.estimate_vertices_color(mesh, verts)

        mesh.remove_duplicated_vertices()

        if mesh_normal:
            mesh.compute_vertex_normals()

        if filter_isolated_mesh:
            mesh = self.filter_isolated_vertices(mesh, self.config.min_cluster_vertices)

        # global transform (to world coordinate system) before output
        if not np.array_equal(self.global_transform, np.eye(4)):
            mesh.transform(self.global_transform)

        # write the mesh to ply file
        if mesh_path is not None:
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            if not self.silence:
                print("save the mesh to %s\n" % (mesh_path))

        return mesh
