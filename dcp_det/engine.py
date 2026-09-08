"""Single-GPU FP32 training loop for the final DCP-Det objective."""

import math
import time
from collections import defaultdict
from typing import Dict

import torch


LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


def _warmup_scheduler(optimizer, iterations: int, factor: float):
    def schedule(step):
        if step >= iterations:
            return 1.0
        alpha = float(step) / float(iterations)
        return factor * (1.0 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch: int,
    warmup_iterations: int,
    warmup_factor: float,
    print_frequency: int = 20,
) -> Dict[str, float]:
    model.train()
    if hasattr(model.roi_heads, "reset_dcp_statistics"):
        model.roi_heads.reset_dcp_statistics()
    warmup = None
    if epoch == 0 and warmup_iterations > 0:
        warmup = _warmup_scheduler(optimizer, warmup_iterations, warmup_factor)
    totals = defaultdict(float)
    start = time.perf_counter()
    for step, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        if set(loss_dict) != set(LOSS_KEYS):
            raise RuntimeError("Unexpected loss dictionary: {}".format(sorted(loss_dict)))
        loss = sum(loss_dict.values())
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite training loss: {}".format(value))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if warmup is not None:
            warmup.step()
        totals["loss"] += value
        for key, tensor in loss_dict.items():
            totals[key] += float(tensor.detach().item())
        if print_frequency and (step % print_frequency == 0 or step + 1 == len(data_loader)):
            print(
                "epoch={} step={}/{} loss={:.6f} lr={:.8f}".format(
                    epoch, step + 1, len(data_loader), value, optimizer.param_groups[0]["lr"]
                )
            )
    count = max(len(data_loader), 1)
    result = {key: value / count for key, value in totals.items()}
    result["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    result["epoch_seconds"] = time.perf_counter() - start
    result["phys_gamma"] = float(model.backbone.custom_fpn.phys_gamma.detach().item())
    return result

