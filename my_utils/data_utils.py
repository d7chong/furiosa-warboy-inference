from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple


class MSCOCODataset(Dataset):
    def __init__(self, image_dir: Path, annotations_file: Path, preprocess: Callable, input_shape):
        self.coco = COCO(annotations_file)
        coco_images = self.coco.dataset["images"]
        self.image_paths = list(image_dir / image["file_name"] for image in coco_images)
        self.image_filename_to_annotation = {
            image["file_name"]: image for image in coco_images
        }
        self.preprocess = preprocess
        self.input_shape = input_shape[2:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        # Ensure input_shape has the correct number of dimensions
        input_, contexts = self.preprocess(img, new_shape=self.input_shape)
        annotation = self.image_filename_to_annotation[path.name]
        return input_, annotation, contexts


def collate_fn(batch):
    images, annotations, contexts = zip(*batch)
    return images, annotations, contexts


def get_dataloader(
    image_dir: Path,
    annotations_dir: Path,
    preprocessor: Callable,
    input_shape: Tuple[int, int, int],
    batch_size: int,
    num_workers: int = 16,
    shuffle: bool = False,
    pin_memory: bool = False,
    prefetch_factor: int = 8,
) -> DataLoader:
    dataset = MSCOCODataset(image_dir, annotations_dir, preprocessor, input_shape)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )