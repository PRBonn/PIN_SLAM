import rosbag
import sensor_msgs.point_cloud2 as pc2

import os
import argparse
import numpy as np

from module import ply

def rosbag2ply(args):

    os.makedirs(args.output_folder, 0o755, exist_ok=True)
    
    begin_flag = False

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
            #print(array[:, :6])
            
            timestamps = array[:, 4] # for hilti and others
            # timestamps = array[:, 5] # for m2dgr
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_bag', help="path to the input rosbag")
    parser.add_argument('-o','--output_folder', help="path for output foler")
    parser.add_argument('-t','--topic', help="name of the point cloud topic used in the rosbag", default="/hesai/pandar_points")
    args = parser.parse_args()
    print("usage: python3 rosbag2ply.py -i [path to input rosbag] -o [path to point cloud output folder] -t [name of point cloud rostopic]")
    
    rosbag2ply(args)
