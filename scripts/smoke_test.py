"""Run isolated construction, data loading, training, backward, and inference checks."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dcp_det.config import load_config
from dcp_det.data import EndCocoDataset
from dcp_det.engine import LOSS_KEYS
from dcp_det.model import architecture_audit, build_dcp_det, load_checkpoint
from dcp_det.transforms import Compose, ToTensor


def _module_isolation(repo_root, forbidden_root):
    if forbidden_root is None:
        return []
    original_root = Path(forbidden_root).expanduser().resolve()
    violations = []
    for name, module in sorted(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        try:
            path.relative_to(original_root)
        except ValueError:
            continue
        try:
            path.relative_to(repo_root)
        except ValueError:
            violations.append({"module": name, "path": str(path)})
    return violations


def _valid_target(target):
    boxes = target["boxes"]
    return (
        boxes.ndim == 2
        and boxes.shape[1] == 4
        and bool(torch.all(boxes[:, 2] > boxes[:, 0]))
        and bool(torch.all(boxes[:, 3] > boxes[:, 1]))
        and bool(torch.all(target["labels"] >= 1))
        and bool(torch.all(target["labels"] <= 5))
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the clean DCP-Det repository")
    parser.add_argument("--config", default="configs/dcp_det.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--forbidden-import-root",
        default=None,
        help="Fail if an imported Python module comes from this directory",
    )
    args = parser.parse_args()

    repo_root = REPOSITORY_ROOT
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = {
        split: EndCocoDataset(
            args.data_root,
            split,
            Compose([ToTensor()]),
            exclude_empty=(split == "train"),
        )
        for split in ("train", "val", "test")
    }
    samples = {split: dataset[0] for split, dataset in datasets.items()}
    if not all(_valid_target(target) for _, target in samples.values()):
        raise RuntimeError("Data loader returned an invalid target")

    model = build_dcp_det(config, pretrained=False)
    checkpoint_status = "not_requested"
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        checkpoint_status = "strict_load_pass"
    audit = architecture_audit(model)
    evc_calls = {"P4": 0, "P5": 0}
    hooks = [
        model.backbone.custom_fpn.evc_c4.register_forward_hook(
            lambda module, inputs, output: evc_calls.__setitem__("P4", evc_calls["P4"] + 1)
        ),
        model.backbone.custom_fpn.evc_c5.register_forward_hook(
            lambda module, inputs, output: evc_calls.__setitem__("P5", evc_calls["P5"] + 1)
        ),
    ]
    model.to(device).train()
    train_image, train_target = samples["train"]
    losses = model(
        [train_image.to(device)],
        [{key: value.to(device) for key, value in train_target.items()}],
    )
    if set(losses) != set(LOSS_KEYS):
        raise RuntimeError("Unexpected loss keys: {}".format(sorted(losses)))
    total_loss = sum(losses.values())
    if not bool(torch.isfinite(total_loss)):
        raise RuntimeError("Training loss is not finite")
    total_loss.backward()
    if model.roi_heads.last_dcp_batch_statistics is None:
        raise RuntimeError("DCP-DIoU statistics were not produced")

    model.eval()
    inference_counts = {}
    with torch.no_grad():
        for split in ("val", "test"):
            image, _ = samples[split]
            output = model([image.to(device)])[0]
            if set(output) != {"boxes", "labels", "scores"}:
                raise RuntimeError("Unexpected inference output")
            inference_counts[split] = int(output["boxes"].shape[0])
    for hook in hooks:
        hook.remove()
    if evc_calls["P4"] == 0 or evc_calls["P5"] == 0:
        raise RuntimeError("EVC was not active on both P4 and P5")

    violations = _module_isolation(repo_root, args.forbidden_import_root)
    if violations:
        raise RuntimeError("Original-project imports found: {}".format(violations))
    result = {
        "model_build": "PASS",
        "train_forward": "PASS",
        "backward": "PASS",
        "inference": "PASS",
        "data_loader": "PASS",
        "checkpoint": checkpoint_status,
        "device": str(device),
        "loss_keys": sorted(losses),
        "losses": {key: float(value.detach().cpu().item()) for key, value in losses.items()},
        "loss_box_reg": "DCP-DIoU",
        "dcp_diou_statistics": model.roi_heads.last_dcp_batch_statistics,
        "evc_calls": evc_calls,
        "inference_detections": inference_counts,
        "architecture": audit,
        "original_project_dependency": "NONE",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
