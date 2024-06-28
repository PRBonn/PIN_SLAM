import rosbag
import sensor_msgs.point_cloud2 as pc2

import os
import argparse
import numpy as np
import open3d as o3d

from module import ply

def rosbag2ply(args):

    os.makedirs(args.output_folder, 0o755, exist_ok=True)

    if args.output_pcd:
        output_folder_pcd = args.output_folder+"_pcd"
        os.makedirs(output_folder_pcd, 0o755, exist_ok=True)

    
    begin_flag = False

    print('Start extraction')
    in_bag = rosbag.Bag(args.input_bag)
    for topic, msg, t in in_bag.read_messages():

        if not begin_flag:
            print(topic)

        if topic == args.topic:
            gen = pc2.read_points(msg, skip_nans=True)
            data = list(gen)
            array = np.array(data)

            # NOTE: point cloud array: x,y,z,intensity,timestamp,ring,others...
            # could be different for some other rosbags
            # print(array[:, :6])
            
            timestamps = array[:, 4] # for hilti, vbr, and others
            # timestamps = array[:, 5] # for m2dgrmm
            # print(timestamps)

            if not begin_flag:
                shift_timestamp = timestamps[0]
                begin_flag = True

            timestamps_shifted = timestamps - shift_timestamp
            # print(timestamps_shifted)

            field_names = ['x','y','z','intensity','timestamp']
            ply_file_path = os.path.join(args.output_folder, str(t)+".ply")

            if ply.write_ply(ply_file_path, [array[:,:4], timestamps_shifted], field_names):
                print("Export : "+ply_file_path)
            else:
                print('ply.write_ply() failed')

            if args.output_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(array[:, :3])
                pcd_file_path = os.path.join(output_folder_pcd, str(t)+".pcd")
                o3d.io.write_point_cloud(pcd_file_path, pcd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_bag', help="path to the input rosbag")
    parser.add_argument('-o','--output_folder', help="path for output foler")
    parser.add_argument('-t','--topic', help="name of the point cloud topic used in the rosbag", default="/hesai/pandar_points")
    parser.add_argument('-p','--output_pcd', action='store_true', help='Also output the pcd file')
    args = parser.parse_args()
    print("usage: python3 rosbag2ply.py -i [path to input rosbag] -o [path to point cloud output folder] -t [name of point cloud rostopic]")
    
    rosbag2ply(args)
