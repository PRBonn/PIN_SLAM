import csv
import numpy as np
from pyquaternion import Quaternion

def read_tum_format_poses_csv(filename: str):
    poses = []
    with open(filename, mode="r") as f:
        # reader = csv.reader(f, delimiter=' ')  # split with space
        reader = csv.reader(f, delimiter=',')  # split with comma
        # # get header and change timestamp label name
        header = next(reader)
        # header[0] = "ts"
        # Convert string odometry to numpy transfor matrices
        for row in reader:
            # print(row[2:5])
            trans = np.array(row[2:5])
            
            quat = Quaternion(np.array([row[8], row[5], row[6], row[7]])) # w, i, j, k

            rot = quat.rotation_matrix
            # Build numpy transform matrix
            odom_tf = np.eye(4)
            odom_tf[0:3, 3] = trans
            odom_tf[0:3, 0:3] = rot
            poses.append(odom_tf)

    return poses

# copyright: Nacho et al. KISS-ICP
def write_kitti_format_poses(filename: str, poses):
    def _to_kitti_format(poses: np.ndarray) -> np.ndarray:
        return np.array([np.concatenate((pose[0], pose[1], pose[2])) for pose in poses])

    np.savetxt(fname=f"{filename}.txt", X=_to_kitti_format(poses))


in_file = "./data/ncd_128/math_e/gt-state-easy.csv"
out_file = "./data/ncd_128/math_e/poses"

poses = read_tum_format_poses_csv(in_file)
write_kitti_format_poses(out_file, poses)