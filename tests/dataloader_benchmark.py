import numpy as np
from typing import Callable, Dict, Iterator, Tuple
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import time
import cv2

from my_utils import (
    TASKS,
    ANCHORS,
    CONF_THRES,
    IOU_THRES,
    TASK_TO_ANNOTATION_PATH,
)
from my_utils.data_utils import get_dataloader
from my_utils.preprocess import YOLOPreProcessor
from my_utils.postprocess import getPostProcesser
from my_utils.quantize import get_yolo_task


class MSCOCODataLoader:
    """Data loader for MSCOCO dataset"""

    def __init__(
        self, image_dir: Path, annotations_file: Path, preprocess: Callable, input_shape
    ) -> None:
        self.coco = COCO(annotations_file)
        coco_images = self.coco.dataset["images"]
        self.image_paths = list(image_dir / image["file_name"] for image in coco_images)
        self.image_filename_to_annotation = {
            image["file_name"]: image for image in coco_images
        }
        self.preprocess = preprocess
        self.input_shape = input_shape[2:]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict]]:
        for path in self.image_paths:
            yield path, self.image_filename_to_annotation[path.name]

    def __len__(self) -> int:
        return len(self.image_paths)


def test(model_name, batch_size, num_workers, prefetch_factor, print_baseline=False):
    task = get_yolo_task(model_name)
    
    # Get preprocessor
    preprocessor = YOLOPreProcessor()
    
    # Get model anchors
    for model_keyword in ANCHORS.keys():
        if model_keyword in model_name:
            anchors = ANCHORS[model_keyword]
            break
    if "u" in model_name:
        anchors = [None]
    
    # Get postprocessor
    model_cfg = {"conf_thres": CONF_THRES, "iou_thres": IOU_THRES, "anchors": anchors}
    postprocessor = getPostProcesser(task, model_name, model_cfg, None, False).postprocess_func
    
    # Get data path (MSCOCO images)
    # Get annotation path, based on task
    data_path = Path("/home/furiosa/work_space/val2017")
    annotation_path = TASK_TO_ANNOTATION_PATH[task]
    
    # baseline dataloader, from furiosa-ai/warboy-vision-models/tree/refactor_jw
    baseline_dataloader = MSCOCODataLoader(
        data_path,
        annotation_path,
        preprocessor,
        input_shape=[1, 3, 640, 640],
    )
    
    # nullAI dataloader
    our_dataloader = get_dataloader(
        data_path,
        annotation_path,
        preprocessor,
        batch_size=batch_size,
        num_workers=num_workers,
        input_shape=[1, 3, 640, 640],
        prefetch_factor=prefetch_factor,
        # pin_memory=True,
    )
    
    if print_baseline:
        tik = time.time()
        for (img_path, annotation) in tqdm(baseline_dataloader):
            img = cv2.imread(str(img_path))
            input_, contexts = preprocessor(img, new_shape=baseline_dataloader.input_shape)
        print(f"BASELINE: {time.time() - tik:.3f}s")
    
    print(f"batch_size: {batch_size}, num_workers: {num_workers}, prefetch_factor: {prefetch_factor}")
    tik = time.time()
    for imgs, annotations, contexts in tqdm(our_dataloader):
        pass
    print(f"OURS:     {time.time() - tik:.3f}s")
    print()

if __name__ == "__main__":
    model_name = "yolov5s"
    task = get_yolo_task(model_name)
    
    print("="*20, f"TASK: {task}", "="*20)
    
    print_baseline = True
    for prefetch_factor in [1, 2, 4, 8]:
        for batch_size in [1, 2, 4, 8, 16]:
            for num_workers in [1, 2, 4, 8]:
                test(model_name, batch_size, num_workers, prefetch_factor, print_baseline)
                print_baseline = False