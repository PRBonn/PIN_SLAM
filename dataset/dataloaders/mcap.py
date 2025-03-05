# MIT License
#
# Copyright (c) 2023 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import importlib
import os
import sys

from natsort import natsorted

class McapDataloader:
    def __init__(self, data_dir: str, topic: str, *_, **__):
        """Standalone .mcap dataloader withouth any ROS distribution."""
        # Conditional imports to avoid injecting dependencies for non mcap users

        try:
            self.make_reader = importlib.import_module("mcap.reader").make_reader
            self.read_ros2_messages = importlib.import_module("mcap_ros2.reader").read_ros2_messages
        except ModuleNotFoundError:
            print("mcap plugins not installed: 'pip install mcap-ros2-support'")
            exit(1)

        from utils.point_cloud2 import read_point_cloud

        # Handle both single file and directory inputs
        if os.path.isfile(data_dir):
            self.mcap_files = [data_dir]
        elif os.path.isdir(data_dir):
            self.mcap_files = natsorted([
                os.path.join(data_dir, f) for f in os.listdir(data_dir)
                if f.endswith('.mcap')
            ])
            assert len(self.mcap_files) > 0, f"No .mcap files found in directory: {data_dir}"
            if len(self.mcap_files) > 1:
                print("Reading multiple .mcap files in directory:")
                print("\n".join([os.path.basename(path) for path in self.mcap_files]))
        else:
            raise ValueError(f"Input path {data_dir} is neither a file nor directory")

        # Initialize with first file
        self.current_file_idx = 0
        self.sequence_id = os.path.basename(self.mcap_files[0]).split(".")[0]
        self._initialize_current_file(self.mcap_files[0], topic)
        
        self.read_point_cloud = read_point_cloud
        
        # Calculate total number of scans across all files
        self.total_scans = self._calculate_total_scans(topic)

    def _initialize_current_file(self, mcap_file: str, topic: str):
        """Initialize readers for a new mcap file."""
        if hasattr(self, 'bag'):
            del self.bag
        self.bag = self.make_reader(open(mcap_file, "rb"))
        self.summary = self.bag.get_summary()
        self.topic = self.check_topic(topic)
        self.n_scans = self._get_n_scans()
        self.msgs = self.read_ros2_messages(mcap_file, topics=topic)
        self.current_scan = 0

    def __del__(self):
        if hasattr(self, "bag"):
            del self.bag

    def _calculate_total_scans(self, topic: str) -> int:
        """Calculate total number of scans across all mcap files."""
        total = 0
        for file in self.mcap_files:
            bag = self.make_reader(open(file, "rb"))
            summary = bag.get_summary()
            total += sum(
                count
                for (id, count) in summary.statistics.channel_message_counts.items()
                if summary.channels[id].topic == topic
            )
            del bag  # Clean up the reader
        return total

    def __getitem__(self, idx):
        # Check if we need to move to next file
        while self.current_scan >= self.n_scans:
            self.current_file_idx += 1
            if self.current_file_idx >= len(self.mcap_files):
                raise IndexError("Index out of range")
            self._initialize_current_file(self.mcap_files[self.current_file_idx], self.topic)

        msg = next(self.msgs).ros_msg
        self.current_scan += 1
        points, point_ts = self.read_point_cloud(msg)
        frame_data = {"points": points, "point_ts": point_ts}
        return frame_data

    def __len__(self):
        return self.total_scans

    def _get_n_scans(self) -> int:
        return sum(
            count
            for (id, count) in self.summary.statistics.channel_message_counts.items()
            if self.summary.channels[id].topic == self.topic
        )

    def check_topic(self, topic: str) -> str:
        # Extract schema id from the .mcap file that encodes the PointCloud2 msg
        schema_id = [
            schema.id
            for schema in self.summary.schemas.values()
            if schema.name == "sensor_msgs/msg/PointCloud2"
        ][0]

        point_cloud_topics = [
            channel.topic
            for channel in self.summary.channels.values()
            if channel.schema_id == schema_id
        ]

        def print_available_topics_and_exit():
            print("Select from the following topics:")
            print(50 * "-")
            for t in point_cloud_topics:
                print(f"{t}")
            print(50 * "-")
            sys.exit(1)

        if topic and topic in point_cloud_topics:
            return topic
        # when user specified the topic check that exists
        if topic and topic not in point_cloud_topics:
            print(
                f'[ERROR] Dataset does not containg any msg with the topic name "{topic}". '
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml mcap your/topic ... ..."
            )
            print_available_topics_and_exit()
        if len(point_cloud_topics) > 1:
            print(
                "Multiple sensor_msgs/msg/PointCloud2 topics available."
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml mcap your/topic ... ..."
            )
            print_available_topics_and_exit()

        if len(point_cloud_topics) == 0:
            print("[ERROR] Your dataset does not contain any sensor_msgs/msg/PointCloud2 topic")
        if len(point_cloud_topics) == 1:
            return point_cloud_topics[0]
