#!/usr/bin/env python3
"""Validate and merge annotated masked-dataset chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from annotation_app.backend.workflow import merge_chunks, validate_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks-dir",
        default="artifacts/masked_dataset/chunks",
        type=Path,
        help="Directory containing chunk CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/masked_dataset/merged",
        type=Path,
        help="Directory for merged outputs.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    merge_chunks(args.chunks_dir, args.output_dir)


if __name__ == "__main__":
    main()
