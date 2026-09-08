"""Train the final DCP-Det model."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from dcp_det.config import load_config
from dcp_det.data import EndCocoDataset
from dcp_det.engine import train_one_epoch
from dcp_det.evaluation import evaluate_model
from dcp_det.model import architecture_audit, build_dcp_det
from dcp_det.transforms import Compose, RandomHorizontalFlip, ToTensor
from dcp_det.utils import (
    GroupedBatchSampler,
    capture_rng_state,
    create_aspect_ratio_groups,
    restore_rng_state,
    seed_everything,
    worker_init_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCP-Det on the END dataset")
    parser.add_argument("--config", default="configs/dcp_det.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs/dcp_det")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None, help="Path to a full last.pth checkpoint")
    return parser.parse_args()


def append_csv(path: Path, row):
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_loaders(config, data_root, generator):
    train_dataset = EndCocoDataset(
        data_root,
        "train",
        Compose([ToTensor(), RandomHorizontalFlip(0.5)]),
        exclude_empty=bool(config["training"]["exclude_empty_train_images"]),
    )
    val_dataset = EndCocoDataset(data_root, "val", Compose([ToTensor()]))
    sampler = RandomSampler(train_dataset, generator=generator)
    group_ids = create_aspect_ratio_groups(
        train_dataset, k=int(config["training"]["aspect_ratio_group_factor"])
    )
    batch_sampler = GroupedBatchSampler(
        sampler, group_ids, int(config["training"]["batch_size"])
    )
    loader_kwargs = {
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": bool(config["training"]["pin_memory"]),
        "persistent_workers": bool(config["training"]["persistent_workers"]),
        "collate_fn": EndCocoDataset.collate_fn,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def save_checkpoint(path, model, optimizer, scheduler, generator, epoch, best_state, config):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "data_generator_state": generator.get_state(),
            "rng_state": capture_rng_state(),
            "epoch": int(epoch),
            "best_state": dict(best_state),
            "config": config,
        },
        str(path),
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = int(config["seed"]["value"])
    seed_everything(seed)

    model = build_dcp_det(config, pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["lr"]),
        momentum=float(config["training"]["momentum"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["scheduler"]["step_size"]),
        gamma=float(config["scheduler"]["gamma"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader, val_loader = build_loaders(config, args.data_root, generator)
    if len(train_loader) - 1 != int(config["warmup"]["iterations"]):
        raise RuntimeError(
            "Official warmup assumes 204 iterations; got train loader length {}".format(
                len(train_loader)
            )
        )

    start_epoch = 0
    best_state = {"AP50": -1.0, "epoch": -1}
    if args.resume:
        checkpoint = torch.load(str(Path(args.resume).expanduser().resolve()), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        generator.set_state(checkpoint["data_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_state = dict(checkpoint["best_state"])

    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "architecture.json").write_text(
        json.dumps(architecture_audit(model), indent=2), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.csv"
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            int(config["warmup"]["iterations"]) if config["warmup"]["enabled"] else 0,
            float(config["warmup"]["factor"]),
        )
        scheduler.step()
        validation = evaluate_model(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "loss_classifier": train_metrics["loss_classifier"],
            "loss_box_reg": train_metrics["loss_box_reg"],
            "loss_objectness": train_metrics["loss_objectness"],
            "loss_rpn_box_reg": train_metrics["loss_rpn_box_reg"],
            "lr": train_metrics["learning_rate"],
            "AP50": validation["AP50"],
            "AP75": validation["AP75"],
            "AP50_95": validation["AP50_95"],
            "AR100": validation["AR100"],
            "AP_small": validation["AP_small"],
            "phys_gamma": train_metrics["phys_gamma"],
            "epoch_seconds": train_metrics["epoch_seconds"],
        }
        append_csv(metrics_path, row)
        (output_dir / "validation_metrics_latest.json").write_text(
            json.dumps({key: value for key, value in validation.items() if key != "predictions"}, indent=2),
            encoding="utf-8",
        )
        if validation["AP50"] > best_state["AP50"]:
            best_state = {"AP50": validation["AP50"], "epoch": epoch}
            save_checkpoint(
                output_dir / "best_by_ap50.pth",
                model,
                optimizer,
                scheduler,
                generator,
                epoch,
                best_state,
                config,
            )
        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            generator,
            epoch,
            best_state,
            config,
        )
        print(
            "epoch={} AP50={:.6f} AP75={:.6f} AP50-95={:.6f} AR100={:.6f}".format(
                epoch,
                validation["AP50"],
                validation["AP75"],
                validation["AP50_95"],
                validation["AR100"],
            )
        )


if __name__ == "__main__":
    main()
