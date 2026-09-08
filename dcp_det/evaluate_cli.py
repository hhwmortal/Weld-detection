"""Shared validation/test command implementation."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import EndCocoDataset
from .evaluation import evaluate_model
from .model import build_dcp_det, load_checkpoint
from .transforms import Compose, ToTensor


def run(default_split: str) -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCP-Det with COCO metrics")
    parser.add_argument("--config", default="configs/dcp_det.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = EndCocoDataset(args.data_root, default_split, Compose([ToTensor()]))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=EndCocoDataset.collate_fn,
    )
    model = build_dcp_det(config, pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    metrics = evaluate_model(model, loader, device)
    serializable = {key: value for key, value in metrics.items() if key != "predictions"}
    print(json.dumps(serializable, indent=2))
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "{}_metrics.json".format(default_split)).write_text(
            json.dumps(serializable, indent=2), encoding="utf-8"
        )
        (output_dir / "{}_predictions.json".format(default_split)).write_text(
            json.dumps(metrics["predictions"]), encoding="utf-8"
        )

