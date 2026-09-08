"""Export deterministic filename lists from the official COCO annotations."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create END train/val/test split files")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        annotation_path = root / "annotations" / "instances_{}.json".format(split)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        names = [str(item["file_name"]) for item in payload["images"]]
        (split_dir / "{}.txt".format(split)).write_text(
            "\n".join(names) + "\n", encoding="utf-8"
        )
        print("{}: {} filenames".format(split, len(names)))


if __name__ == "__main__":
    main()
