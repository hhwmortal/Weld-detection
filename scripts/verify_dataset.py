"""Validate END metadata, split independence, annotations, and optional images."""

import argparse
import json
import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from dcp_det.data import verify_dataset


def main():
    parser = argparse.ArgumentParser(description="Verify the official END release")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Verify annotations and split independence without requiring image files",
    )
    args = parser.parse_args()
    summary = verify_dataset(args.data_root, require_images=not args.metadata_only)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
