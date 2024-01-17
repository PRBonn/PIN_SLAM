<p align="center">

  <h1 align="center">📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency</h1>
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/index.html"><strong>Yue Pan</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/index.php/people/xingguang-zhong/index.html"><strong>Xingguang Zhong</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/index.php/people/louis-wiesmann/index.html"><strong>Louis Wiesmann</strong></a>
    .
    <a href=""><strong>Thorbjörn Posewsky</strong></a>
    .
    <a href="https://www.ipb.uni-bonn.de/index.php/people/jens-behley/index.html"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/index.html"><strong>Cyrill Stachniss</strong></a>
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de/index.html"><strong>University of Bonn</strong></a>
  <h3 align="center">Preprint</a> | Video</a></h3>
  <div align="center"></div>
</p>

**TL;DR: PIN-SLAM is a full-fledged implicit neural LiDAR SLAM system including odometry, loop closure detection, and globally consistent mapping**

----

![pin_slam_teaser](https://github.com/PRBonn/PIN_SLAM/assets/34207278/b5ab4c89-cdbe-464e-afbe-eb432b42fccc)

*Globally consistent point-based implicit neural (PIN) map built with PIN-SLAM in Bonn. The high-fidelity mesh can be reconstructed from the neural point map.*

----

![pin_slam_loop_compare](https://github.com/PRBonn/PIN_SLAM/assets/34207278/7dadd438-5a46-451a-9add-c9c08dcae277)

*Comparison of (a) the inconsistent mesh with duplicated structures reconstructed by PIN LiDAR odometry, and (b) the globally consistent mesh reconstructed by PIN-SLAM.*

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
