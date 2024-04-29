import numpy as np


class LabelDataConverter:
    """Convert .label binary data to instance id and rgb"""

    def __init__(self, labelscan):

        self.convertdata(labelscan)

    def convertdata(self, labelscan):

        self.semantic_id = []
        self.rgb_id = []
        self.instance_id = []
        self.rgb_arr_id = []

        for counting in range(len(labelscan)):

            sem_id = int(labelscan[counting]) & 0xFFFF  # lower 16 bit
            rgb, rgb_arr = self.get_sem_rgb(sem_id)
            instance_id = int(labelscan[counting]) >> 16  # higher 16 bit
            # rgb = self.get_random_rgb(instance_id)

            # print("Sem label:", sem_id, "Ins label:", instance_id, "Color:", hex(rgb))
            # print(hex(rgb))
            # instance label is given in each semantic label

            self.semantic_id.append(sem_id)
            self.rgb_id.append(rgb)
            self.rgb_arr_id.append(rgb_arr)
            self.instance_id.append(instance_id)


def get_random_rgb(n):
    n = ((n ^ n >> 15) * 2246822519) & 0xFFFFFFFF
    n = ((n ^ n >> 13) * 3266489917) & 0xFFFFFFFF
    n = (n ^ n >> 16) >> 8
    print(n)
    return tuple(n.to_bytes(3, "big"))


sem_kitti_learning_map = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 20,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 20,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car"
    253: 7,  # "moving-bicyclist"
    254: 6,  # "moving-person"
    255: 8,  # "moving-motorcyclist"
    256: 5,  # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
    257: 5,  # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
    258: 4,  # "moving-truck"
    259: 5,  # "moving-other-vehicle"
}


def sem_map_function(value):
    return sem_kitti_learning_map.get(value, value)


sem_kitti_labels = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
    20: "others",
}

sem_kitti_color_map = {  # rgb
    0: [255, 255, 255],
    1: [100, 150, 245],
    2: [100, 230, 245],
    3: [30, 60, 150],
    4: [80, 30, 180],
    5: [0, 0, 255],
    6: [255, 30, 30],
    7: [255, 40, 200],
    8: [150, 30, 90],
    9: [255, 0, 255],
    10: [255, 150, 255],
    11: [75, 0, 75],
    12: [175, 0, 75],
    13: [255, 200, 0],
    14: [255, 120, 50],
    15: [0, 175, 0],
    16: [135, 60, 0],
    17: [150, 240, 80],
    18: [255, 240, 150],
    19: [255, 0, 0],
    20: [30, 30, 30],
}
