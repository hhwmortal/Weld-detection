"""Generate a release manifest and report public-repository hygiene checks."""

import argparse
import json
import re
from pathlib import Path


EXCLUDED_SUFFIXES = {".pth", ".pt", ".ckpt", ".weights", ".pyc", ".pyo"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".gitignore"}


def purpose(path):
    value = path.as_posix()
    if value == "README.md":
        return "Project overview and user guide"
    if value == "LICENSE":
        return "Code-license placeholder requiring author completion"
    if value == "requirements.txt":
        return "Minimal Python dependencies"
    if value == ".gitignore":
        return "Generated-output and development-artifact exclusions"
    if value.startswith("configs/"):
        return "Final verified model and training configuration"
    if value.startswith("dcp_det/backbone/"):
        return "ResNet-50 and feature-pyramid assembly"
    if value.startswith("dcp_det/modules/"):
        return "DCP extraction and EVC feature modules"
    if value.startswith("dcp_det/detection/"):
        return "Required Faster R-CNN, RPN, RoI, box, and transform utilities"
    if value.startswith("dcp_det/utils/"):
        return "Reproducibility and sampling utilities"
    if value.startswith("dcp_det/"):
        return "DCP-Det package runtime"
    if value.startswith("datasets/END/annotations/"):
        return "Official END COCO annotations"
    if value.startswith("datasets/END/splits/"):
        return "Official END split filename list"
    if value.endswith(".gitkeep"):
        return "Preserves the expected dataset image directory"
    if value == "datasets/END/README.md":
        return "External END dataset layout and download instructions"
    if value.startswith("datasets/"):
        return "Dataset layout documentation"
    if value.startswith("docs/"):
        return "Dataset or reproducibility documentation"
    if value.startswith("scripts/"):
        return "Repository verification utility"
    if value == "train.py":
        return "Final DCP-Det training entry point"
    if value == "evaluate.py":
        return "Validation entry point"
    if value == "test.py":
        return "Independent test entry point"
    if value == "predict.py":
        return "Image and folder inference entry point"
    if value == "RELEASE_MANIFEST.txt":
        return "Complete release file manifest"
    return "Release support file"


def main():
    parser = argparse.ArgumentParser(description="Audit a prepared DCP-Det repository")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    forbidden = [path for path in files if path.suffix.lower() in EXCLUDED_SUFFIXES]
    oversized = [path for path in files if path.stat().st_size > 20 * 1024 * 1024]
    local_paths = []
    credentials = []
    drive_pattern = re.compile(r"[A-Za-z]:" + re.escape("\\"))
    credential_pattern = re.compile(
        r"(?i)(api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[^\s]+"
    )
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            user_path_marker = "C:/" + "Users/"
            if drive_pattern.search(line) or user_path_marker in line:
                local_paths.append("{}:{}".format(path.relative_to(root), line_number))
            if credential_pattern.search(line) and "credential_pattern" not in line:
                credentials.append("{}:{}".format(path.relative_to(root), line_number))

    manifest_path = root / "RELEASE_MANIFEST.txt"
    manifest_files = [path for path in files if path != manifest_path]
    lines = ["DCP-Det release file manifest", ""]
    for path in manifest_files:
        relative = path.relative_to(root)
        lines.append("{} | {}".format(relative.as_posix(), purpose(relative)))
    lines.append("RELEASE_MANIFEST.txt | {}".format(purpose(Path("RELEASE_MANIFEST.txt"))))
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    total_size = sum(path.stat().st_size for path in files)
    dataset_size = sum(
        path.stat().st_size
        for path in files
        if path.relative_to(root).as_posix().startswith("datasets/")
    )
    report = {
        "root": str(root),
        "total_bytes": total_size,
        "code_bytes": total_size - dataset_size,
        "dataset_bytes": dataset_size,
        "files": len(files),
        "python_files": sum(path.suffix.lower() == ".py" for path in files),
        "images": sum(path.suffix.lower() in IMAGE_SUFFIXES for path in files),
        "forbidden_artifacts": [str(path.relative_to(root)) for path in forbidden],
        "files_over_20_mib": [str(path.relative_to(root)) for path in oversized],
        "hard_coded_local_paths": local_paths,
        "possible_credentials": credentials,
    }
    print(json.dumps(report, indent=2))
    if forbidden or local_paths or credentials:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
