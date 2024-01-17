<p align="center">

  <h1 align="center">📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency</h1>
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/"><strong>Yue Pan</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/xingguang-zhong/"><strong>Xingguang Zhong</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/index.php/people/louis-wiesmann/"><strong>Louis Wiesmann</strong></a>
    .
    <a href=""><strong>Thorbjörn Posewsky</strong></a>
    .
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong></a>
  <h3 align="center"><a href="https://arxiv.org/abs/2210.02299">Preprint</a> | Video</a></h3>
  <div align="center"></div>
</p>



----
## Abstract
Accurate and robust localization and mapping are
essential components for most autonomous robots. In this paper,
we propose a SLAM system for building globally consistent maps,
called PIN-SLAM, that is based on an elastic and compact
point-based implicit neural map representation. Taking range
measurements as input, our approach alternates between incremental learning of the local implicit signed distance field
and the pose estimation given the current local map using a
correspondence-free, point-to-implicit model registration. Our
implicit map is based on sparse optimizable neural points,
which are inherently elastic and deformable with the global pose
adjustment when closing a loop. Loops are also detected using the
neural point features. Extensive experiments validate that PIN-SLAM is robust to various environments and versatile to different
range sensors such as LiDAR and RGB-D cameras. PIN-SLAM
achieves pose estimation accuracy better or on par with the state-of-the-art LiDAR odometry or SLAM systems and outperforms
the recent neural implicit SLAM approaches while maintaining
a more consistent, and highly compact implicit map that can be
reconstructed as accurate and complete meshes. Finally, thanks to
the voxel hashing for efficient neural points indexing and the fast
implicit map-based registration without closest point association,
PIN-SLAM can run at the sensor frame rate on a moderate GPU.

----
## Codes 
Coming soon.