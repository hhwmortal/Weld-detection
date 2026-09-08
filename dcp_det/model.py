"""Final DCP-Det model construction and strict checkpoint loading."""

from pathlib import Path
from typing import Any, Dict

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from .backbone import build_resnet50_dcp_evc_backbone
from .detection import FasterRCNN, FastRCNNPredictor


def build_dcp_det(config: Dict[str, Any], pretrained: bool = True) -> FasterRCNN:
    """Build the single public DCP-Det architecture."""

    backbone = build_resnet50_dcp_evc_backbone(
        pretrained=pretrained,
        trainable_layers=int(config["model"]["trainable_backbone_layers"]),
        kernel_size=int(config["dcp"]["kernel_size"]),
        phys_gamma_init=float(config["dcp"]["phys_gamma_init"]),
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=91,
        min_size=int(config["input"]["min_size"]),
        max_size=int(config["input"]["max_size"]),
        rpn_nms_thresh=float(config["inference"]["rpn_nms_thresh"]),
        box_score_thresh=float(config["inference"]["box_score_thresh"]),
        box_nms_thresh=float(config["inference"]["box_nms_thresh"]),
        box_detections_per_img=int(config["inference"]["detections_per_image"]),
        dcp_lambda=float(config["dcp_diou"]["lambda"]),
        dcp_eps=float(config["dcp_diou"]["epsilon"]),
        prior_kernel_size=int(config["dcp"]["kernel_size"]),
    )

    if pretrained:
        detector_state = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(
            progress=True, check_hash=True
        )
        incompatible = model.load_state_dict(detector_state, strict=False)
        loaded_keys = set(model.state_dict()).intersection(detector_state)
        required_prefixes = (
            "backbone.body.",
            "rpn.head.",
            "roi_heads.box_head.",
            "roi_heads.box_predictor.",
        )
        for prefix in required_prefixes:
            if not any(key.startswith(prefix) for key in loaded_keys):
                raise RuntimeError("COCO initialization did not load {}".format(prefix))
        allowed_missing_prefixes = (
            "backbone.custom_fpn.",
            "roi_heads.dcp_prior_extractor.",
        )
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
            and not key.endswith("num_batches_tracked")
        ]
        allowed_unexpected_prefixes = ("backbone.fpn.",)
        invalid_unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith(allowed_unexpected_prefixes)
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "Unexpected COCO initialization mismatch: missing={} unexpected={}".format(
                    invalid_missing, invalid_unexpected
                )
            )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, int(config["dataset"]["num_classes"]) + 1
    )
    return model


def checkpoint_model_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint
    raise ValueError("Checkpoint does not contain a recognized model state_dict")


def load_checkpoint(model, path, map_location="cpu") -> Dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    model.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    return checkpoint


def architecture_audit(model) -> Dict[str, Any]:
    gamma_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("phys_gamma")
    ]
    return {
        "model": "DCP-Det",
        "backbone": "ResNet-50",
        "dcp_evc_levels": ["P4", "P5"],
        "dcp_kernel_size": model.backbone.custom_fpn.contrast_prior_extractor.kernel_size,
        "shared_phys_gamma_count": len(gamma_parameters),
        "phys_gamma_initial_value": float(gamma_parameters[0][1].detach().item()),
        "rpn": "standard",
        "nms": "standard",
        "roi_box_loss": "DCP-DIoU",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
