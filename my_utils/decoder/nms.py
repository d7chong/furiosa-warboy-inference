from typing import List

import numpy as np
import torch
   
from torchvision.ops import nms

def non_max_suppression(
    prediction: List[np.ndarray], iou_thres: float = 0.45, class_agnostic=True
) -> List[np.ndarray]:
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    output = []
    for x in prediction:
        boxes = torch.tensor(x[:, :4], dtype=torch.float32)
        scores = torch.tensor(x[:, 4], dtype=torch.float32)
        keep = nms(boxes, scores, iou_thres)
        output.append(x[keep.numpy()])
    return output