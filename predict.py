"""Run DCP-Det inference on an image or directory."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from dcp_det.config import load_config
from dcp_det.data import CLASS_NAMES
from dcp_det.model import build_dcp_det, load_checkpoint


COLORS = {1: "#dc372f", 2: "#266fbf", 3: "#2e9656", 4: "#ef8a21", 5: "#9f4eb1"}


def parse_args():
    parser = argparse.ArgumentParser(description="DCP-Det inference")
    parser.add_argument("--config", default="configs/dcp_det.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def image_paths(input_path: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.iterdir() if path.suffix.lower() in extensions)
    raise FileNotFoundError("Input not found: {}".format(input_path))


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_dcp_det(config, pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in image_paths(Path(args.input).expanduser().resolve()):
        with Image.open(path) as source:
            image = source.convert("RGB")
        with torch.no_grad():
            prediction = model([to_tensor(image).to(device)])[0]
        boxes = prediction["boxes"].detach().cpu().tolist()
        labels = prediction["labels"].detach().cpu().tolist()
        scores = prediction["scores"].detach().cpu().tolist()
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for box, label, score in zip(boxes, labels, scores):
            if score < args.score_threshold:
                continue
            draw.rectangle(tuple(box), outline=COLORS[int(label)], width=3)
            draw.text(
                (box[0] + 2, max(0, box[1] - 12)),
                "{} {:.2f}".format(CLASS_NAMES[int(label)], score),
                fill=COLORS[int(label)],
                font=font,
            )
        canvas.save(output_dir / "{}_prediction.png".format(path.stem))
        records.append(
            {"file_name": path.name, "boxes": boxes, "labels": labels, "scores": scores}
        )
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Processed {} image(s); outputs: {}".format(len(records), output_dir))


if __name__ == "__main__":
    main()

