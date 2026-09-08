"""Configuration loading and final-method invariant checks."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_final_config(config)
    return config


def validate_final_config(config: Dict[str, Any]) -> None:
    expected = {
        ("model", "backbone"): "resnet50",
        ("dataset", "num_classes"): 5,
        ("dataset", "classes"): {1: "Crack", 2: "LF", 3: "LP", 4: "SL", 5: "Pore"},
        ("dcp", "kernel_size"): 31,
        ("dcp", "levels"): ["P4", "P5"],
        ("dcp", "phys_gamma_init"): 0.0,
        ("dcp", "shared_gamma"): True,
        ("dcp", "input"): "normalized_padded",
        ("dcp_diou", "lambda"): 1.0,
        ("dcp_diou", "epsilon"): 1e-6,
        ("dcp_diou", "quality_source"): "matched_gt_dcp_mean",
        ("dcp_diou", "normalization"): "batch_global_positive_rois",
        ("dcp_diou", "detach_quality"): True,
        ("dcp_diou", "detach_weights"): True,
        ("training", "epochs"): 100,
        ("training", "batch_size"): 4,
        ("training", "num_workers"): 4,
        ("training", "pin_memory"): True,
        ("training", "persistent_workers"): False,
        ("training", "exclude_empty_train_images"): True,
        ("training", "aspect_ratio_group_factor"): 3,
        ("training", "optimizer"): "SGD",
        ("training", "lr"): 0.01,
        ("training", "momentum"): 0.9,
        ("training", "weight_decay"): 0.0001,
        ("scheduler", "name"): "StepLR",
        ("scheduler", "step_size"): 15,
        ("scheduler", "gamma"): 0.33,
        ("warmup", "enabled"): True,
        ("warmup", "epoch"): 0,
        ("warmup", "factor"): 0.001,
        ("warmup", "iterations"): 204,
        ("input", "min_size"): 800,
        ("input", "max_size"): 1333,
        ("inference", "box_score_thresh"): 0.05,
        ("inference", "box_nms_thresh"): 0.5,
        ("inference", "rpn_nms_thresh"): 0.7,
        ("inference", "detections_per_image"): 100,
        ("precision", "amp"): False,
        ("seed", "value"): 42,
        ("seed", "deterministic_algorithms"): False,
        ("selection", "metric"): "validation_AP50",
        ("initialization", "backbone"): "torchvision_ResNet50_IMAGENET1K_V1",
        ("initialization", "detector"): "torchvision_FasterRCNN_ResNet50_FPN_COCO_V1",
    }
    for keys, value in expected.items():
        current: Any = config
        for key in keys:
            current = current[key]
        if current != value:
            raise ValueError("Final DCP-Det requires {}={}".format(".".join(keys), value))
    if int(config["model"]["trainable_backbone_layers"]) != 3:
        raise ValueError("Final DCP-Det requires three trainable backbone stages")
