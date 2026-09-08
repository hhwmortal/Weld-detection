"""COCO evaluation for validation and independent testing."""

import contextlib
import io
from typing import Any, Dict, List

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from .data import CLASS_NAMES


def _to_coco_predictions(image_id: int, output: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    boxes = output["boxes"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    rows: List[Dict[str, Any]] = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = (float(value) for value in box)
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return rows


@torch.no_grad()
def evaluate_model(model, data_loader, device, quiet: bool = False) -> Dict[str, Any]:
    model.eval()
    predictions: List[Dict[str, Any]] = []
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            predictions.extend(_to_coco_predictions(int(target["image_id"].item()), output))
    if not predictions:
        raise RuntimeError("The detector produced no predictions for COCO evaluation")

    stream = io.StringIO()
    output_context = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()
    with output_context:
        coco_gt = data_loader.dataset.coco
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = list(data_loader.dataset.ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = [float(value) for value in evaluator.stats.tolist()]
    precision = evaluator.eval["precision"]
    per_class = []
    for category_index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, category_index, 0, 2]
        valid = values[values > -1]
        ap = float(np.mean(valid)) if valid.size else -1.0
        values_50 = precision[0, :, category_index, 0, 2]
        valid_50 = values_50[values_50 > -1]
        ap50 = float(np.mean(valid_50)) if valid_50.size else -1.0
        per_class.append(
            {
                "class_id": int(category_id),
                "class_name": CLASS_NAMES[int(category_id)],
                "AP50_95": ap,
                "AP50": ap50,
            }
        )
    return {
        "AP50_95": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AR100": stats[8],
        "stats": stats,
        "per_class": per_class,
        "predictions": predictions,
    }

