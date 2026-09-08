"""COCO-format loader for the official END train/validation/test splits."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


CLASS_NAMES: Dict[int, str] = {1: "Crack", 2: "LF", 3: "LP", 4: "SL", 5: "Pore"}


class EndCocoDataset(Dataset):
    def __init__(self, data_root, split: str, transforms=None, exclude_empty=False) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.root = self.data_root / "images" / split
        self.annotation_file = self.data_root / "annotations" / "instances_{}.json".format(split)
        if not self.root.is_dir():
            raise FileNotFoundError("Image directory not found: {}".format(self.root))
        if not self.annotation_file.is_file():
            raise FileNotFoundError("Annotation file not found: {}".format(self.annotation_file))
        self.coco = COCO(str(self.annotation_file))
        categories = {int(item["id"]): item["name"] for item in self.coco.dataset["categories"]}
        if categories != CLASS_NAMES:
            raise ValueError("END category mapping mismatch: {}".format(categories))
        self.ids = self._ordered_image_ids()
        if exclude_empty:
            self.ids = [image_id for image_id in self.ids if self.coco.imgToAnns.get(image_id)]
        self.transforms = transforms

    def _ordered_image_ids(self) -> List[int]:
        split_file = self.data_root / "splits" / "{}.txt".format(self.split)
        if not split_file.is_file():
            return [int(item["id"]) for item in self.coco.dataset["images"]]
        names = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        name_to_id = {item["file_name"]: int(item["id"]) for item in self.coco.dataset["images"]}
        missing = [name for name in names if name not in name_to_id]
        if missing:
            raise ValueError("Split file contains filenames absent from COCO JSON: {}".format(missing[:5]))
        if len(names) != len(name_to_id) or set(names) != set(name_to_id):
            raise ValueError("Split file does not exactly cover the COCO image records")
        return [name_to_id[name] for name in names]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        info = self.coco.imgs[image_id]
        image_path = self.root / info["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        annotations = self.coco.imgToAnns.get(image_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        crowds: List[int] = []
        for annotation in annotations:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(int(annotation["category_id"]))
            areas.append(float(annotation.get("area", width * height)))
            crowds.append(int(annotation.get("iscrowd", 0)))
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def get_height_and_width(self, index: int) -> Tuple[int, int]:
        info = self.coco.imgs[self.ids[index]]
        return int(info["height"]), int(info["width"])

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


def verify_dataset(data_root, require_images: bool = True) -> Dict[str, object]:
    root = Path(data_root).expanduser().resolve()
    expected_counts = {"train": (826, 2176), "val": (236, 544), "test": (118, 237)}
    expected_classes = {
        "train": {1: 94, 2: 100, 3: 181, 4: 654, 5: 1147},
        "val": {1: 29, 2: 18, 3: 45, 4: 147, 5: 305},
        "test": {1: 9, 2: 13, 3: 22, 4: 66, 5: 127},
    }
    names_by_split: Dict[str, set] = {}
    summary: Dict[str, object] = {"root": str(root), "splits": {}}
    for split, expected in expected_counts.items():
        annotation_path = root / "annotations" / "instances_{}.json".format(split)
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        categories = {int(item["id"]): item["name"] for item in payload["categories"]}
        if categories != CLASS_NAMES:
            raise RuntimeError("{} category mapping mismatch".format(split))
        images = payload["images"]
        annotations = payload["annotations"]
        if (len(images), len(annotations)) != expected:
            raise RuntimeError("{} count mismatch: {}".format(split, (len(images), len(annotations))))
        image_ids = {int(item["id"]) for item in images}
        names = [str(item["file_name"]) for item in images]
        if len(image_ids) != len(images) or len(set(names)) != len(names):
            raise RuntimeError("{} has duplicate image IDs or filenames".format(split))
        names_by_split[split] = set(names)
        info_by_id = {int(item["id"]): item for item in images}
        class_counts = {class_id: 0 for class_id in CLASS_NAMES}
        for annotation in annotations:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            if image_id not in image_ids or category_id not in CLASS_NAMES:
                raise RuntimeError("{} contains an invalid annotation reference".format(split))
            x, y, width, height = (float(value) for value in annotation["bbox"])
            info = info_by_id[image_id]
            coordinate_tolerance = 1e-3
            if width <= 0 or height <= 0 or x < -coordinate_tolerance or y < -coordinate_tolerance:
                raise RuntimeError("{} contains a broken bbox".format(split))
            if (
                x + width > float(info["width"]) + coordinate_tolerance
                or y + height > float(info["height"]) + coordinate_tolerance
            ):
                raise RuntimeError("{} contains an out-of-bounds bbox".format(split))
            class_counts[category_id] += 1
        if class_counts != expected_classes[split]:
            raise RuntimeError("{} class counts mismatch: {}".format(split, class_counts))
        if require_images:
            image_dir = root / "images" / split
            disk_names = {path.name for path in image_dir.iterdir() if path.is_file()}
            if disk_names != set(names):
                raise RuntimeError("{} image directory does not exactly match annotations".format(split))
            for item in images:
                with Image.open(image_dir / item["file_name"]) as image:
                    if image.size != (int(item["width"]), int(item["height"])):
                        raise RuntimeError("Image dimensions mismatch: {}".format(item["file_name"]))
        summary["splits"][split] = {
            "images": len(images),
            "instances": len(annotations),
            "class_counts": {CLASS_NAMES[key]: class_counts[key] for key in CLASS_NAMES},
        }
    overlaps = {
        "train_val": sorted(names_by_split["train"] & names_by_split["val"]),
        "train_test": sorted(names_by_split["train"] & names_by_split["test"]),
        "val_test": sorted(names_by_split["val"] & names_by_split["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError("Split filename overlap detected: {}".format(overlaps))
    summary["overlaps"] = overlaps
    summary["total_images"] = sum(item[0] for item in expected_counts.values())
    summary["total_instances"] = sum(item[1] for item in expected_counts.values())
    return summary
