"""Evaluate DCP-Det on the official validation split."""

from dcp_det.evaluate_cli import run


if __name__ == "__main__":
    run("val")

