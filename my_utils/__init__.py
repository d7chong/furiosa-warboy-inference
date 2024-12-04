import numpy as np

# Thresholds for YOLO models
CONF_THRES = 0.001
IOU_THRES = 0.7

# Directories for model weights
ONNX_DIR = "./models/onnx"
QUANT_ONNX_DIR = "./models/quant_onnx"
WEIGHTS_DIR = "./models/weights"

TASK_TO_ANNOTATION_PATH = {
    "object_detection":         "/home/furiosa/work_space/annotations/instances_val2017.json",
    "pose_estimation":          "/home/furiosa/work_space/annotations/person_keypoints_val2017.json",
    "instance_segmentation":    "/home/furiosa/work_space/annotations/instances_val2017.json"
}

# YOLO - Object Detection - Target Accuracies
TARGET_DET_ACCURACY = {
    "yolov5nu": 0.343,
    "yolov5su": 0.430,
    "yolov5mu": 0.490,
    "yolov5lu": 0.522,
    "yolov5xu": 0.532,
    "yolov5n": 0.280,
    "yolov5s": 0.374,
    "yolov5m": 0.454,
    "yolov5l": 0.490,
    "yolov5x": 0.507,
    "yolov7": 0.514,
    "yolov7x": 0.531,
    "yolov7-w6": 0.549,
    "yolov7-e6": 0.560,
    "yolov7-d6": 0.566,
    "yolov7-e6e": 0.568,
    "yolov8n": 0.373,
    "yolov8s": 0.449,
    "yolov8m": 0.502,
    "yolov8l": 0.529,
    "yolov8x": 0.539,
    "yolov9t": 0.383,
    "yolov9s": 0.468,
    "yolov9m": 0.514,
    "yolov9c": 0.530,
    "yolov9e": 0.556,
    "yolov5n6": 0.360,
    "yolov5n6u": 0.421,
    "yolov5s6": 0.448,
    "yolov5s6u": 0.486,
    "yolov5m6": 0.513,
    "yolov5m6u": 0.536,
    "yolov5l6": 0.537,
    "yolov5l6u": 0.557,
    "yolov5x6": 0.550,
    "yolov5x6u": 0.568,
}

# YOLO Segmentation - Target Accuracies
TARGET_MASK_ACCURACY = {
    "yolov8n-seg": 0.305,
    "yolov8s-seg": 0.368,
    "yolov8m-seg": 0.408,
    "yolov8l-seg": 0.426,
    "yolov8x-seg": 0.434,
    "yolov9c-seg": 0.422,
    "yolov9e-seg": 0.443, 
}

TARGET_BBOX_ACCURACY = {
    "yolov8n-seg": 0.367,
    "yolov8s-seg": 0.446,
    "yolov8m-seg": 0.499,
    "yolov8l-seg": 0.523,
    "yolov8x-seg": 0.534,
    "yolov9c-seg": 0.524,
    "yolov9e-seg": 0.551,
}

# YOLO - Pose Estimation - Target Accuracies
TARGET_POSE_ACCURACY = {
    "yolov8n-pose": 0.504,
    "yolov8s-pose": 0.600,
    "yolov8m-pose": 0.650,
    "yolov8l-pose": 0.676,
    "yolov8x-pose": 0.692,
}

# YOLO - List of all models
TEST_MODEL_LIST = {
    "object_detection": TARGET_DET_ACCURACY.keys(),
    "instance_segmentation": TARGET_MASK_ACCURACY.keys(),  # Renamed key from "segmentation"
    "pose_estimation": TARGET_POSE_ACCURACY.keys(),
}

# YOLO - List of all tasks
TASKS = [
    "object_detection",
    "pose_estimation",
    "instance_segmentation"
]

# YOLO - Category to COCO
YOLO_CATEGORY_TO_COCO_CATEGORY = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 13, 14, 15, 16,
    17, 18, 19, 20, 21,
    22, 23, 24, 25, 27,
    28, 31, 32, 33, 34,
    35, 36, 37, 38, 39,
    40, 41, 42, 43, 44,
    46, 47, 48, 49, 50,
    51, 52, 53, 54, 55,
    56, 57, 58, 59, 60,
    61, 62, 63, 64, 65,
    67, 70, 72, 73, 74,
    75, 76, 77, 78, 79,
    80, 81, 82, 84, 85,
    86, 87, 88, 89, 90,
]

# YOLO - Anchors
ANCHORS = {
    "yolov9": [None],
    "yolov8": [None],
    "yolov7": [
        [12, 16, 19, 36, 40, 28],
        [36, 75, 76, 55, 72, 146],
        [142, 110, 192, 243, 459, 401],
    ],
    "yolov7_6": [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ],
    "yolov5": [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ],
    "yolov5_6": [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ],
}

COLORS = [
    (144, 238, 144),
    (255, 0, 0),
    (178, 34, 34),
    (221, 160, 221),
    (0, 255, 0),
    (0, 128, 0),
    (210, 105, 30),
    (220, 20, 60),
    (192, 192, 192),
    (255, 228, 196),
    (50, 205, 50),
    (139, 0, 139),
    (100, 149, 237),
    (138, 43, 226),
    (238, 130, 238),
    (255, 0, 255),
    (0, 100, 0),
    (127, 255, 0),
    (255, 0, 255),
    (0, 0, 205),
    (255, 140, 0),
    (255, 239, 213),
    (199, 21, 133),
    (124, 252, 0),
    (147, 112, 219),
    (106, 90, 205),
    (176, 196, 222),
    (65, 105, 225),
    (173, 255, 47),
    (255, 20, 147),
    (219, 112, 147),
    (186, 85, 211),
    (199, 21, 133),
    (148, 0, 211),
    (255, 99, 71),
    (144, 238, 144),
    (255, 255, 0),
    (230, 230, 250),
    (0, 0, 255),
    (128, 128, 0),
    (189, 183, 107),
    (255, 255, 224),
    (128, 128, 128),
    (105, 105, 105),
    (64, 224, 208),
    (205, 133, 63),
    (0, 128, 128),
    (72, 209, 204),
    (139, 69, 19),
    (255, 245, 238),
    (250, 240, 230),
    (152, 251, 152),
    (0, 255, 255),
    (135, 206, 235),
    (0, 191, 255),
    (176, 224, 230),
    (0, 250, 154),
    (245, 255, 250),
    (240, 230, 140),
    (245, 222, 179),
    (0, 139, 139),
    (143, 188, 143),
    (255, 0, 0),
    (240, 128, 128),
    (102, 205, 170),
    (60, 179, 113),
    (46, 139, 87),
    (165, 42, 42),
    (178, 34, 34),
    (175, 238, 238),
    (255, 248, 220),
    (218, 165, 32),
    (255, 250, 240),
    (253, 245, 230),
    (244, 164, 96),
    (210, 105, 30),
]

PALETTE = np.array(
    [
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
        [58, 200, 253],
    ],
    np.int32,
)

SKELETONS = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

POSE_LIMB_COLOR = PALETTE[
    [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
].tolist()