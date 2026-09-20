#!/usr/bin/env python3
"""Create and resumably upload the prepared dataset folder to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_FILES = (
    "README.md",
    "PARQUET_SHA256SUMS",
    "PROVENANCE.md",
    "split_manifest.json",
    "stage1_train.parquet",
    "stage1_val.parquet",
    "stage2_train.parquet",
    "stage2_val.parquet",
)

OPTIONAL_FILES = (
    "excluded_samples.jsonl",
    "row_manifest.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Hugging Face dataset id, e.g. user/training-dataset")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "training_data"
        / "Counterfactual_RLVR_ECD_7B_clean_v1",
    )
    parser.add_argument("--public", action="store_true", help="Create a public dataset (default: private)")
    args = parser.parse_args()
    folder = args.folder.resolve()
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset folder is incomplete: {missing}")
    files = list(REQUIRED_FILES) + [name for name in OPTIONAL_FILES if (folder / name).is_file()]

    # This server exports HF_ENDPOINT=https://hf-mirror.com in ~/.bashrc.
    # Authentication headers are dropped when that mirror redirects requests
    # across hosts, so authenticated create/upload calls must use the official
    # endpoint explicitly.
    api = HfApi(endpoint="https://huggingface.co")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=folder,
        allow_patterns=files,
        commit_message=f"Publish {folder.name} training data",
    )
    print(result)


if __name__ == "__main__":
    main()
