#!/usr/bin/env python3
"""Fast, read-only checks for the publishable training package."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
CF_DATA = WORKSPACE_ROOT / "training_data" / "Counterfactual_RLVR_ECD_7B_clean_v1"
BIPS_DATA = WORKSPACE_ROOT / "training_data" / "BiPS_ECD_3B_filtered_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        checksums[name.lstrip("* ")] = digest
    return checksums


def verify_dataset(data: Path, expected_counts: dict, expected_hashes: dict[str, str]) -> None:
    manifest = json.loads((data / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest["stage_counts"] != expected_counts:
        raise AssertionError(f"unexpected stage counts in {data}: {manifest['stage_counts']}")
    expected_names = {
        "stage1_train.parquet",
        "stage1_val.parquet",
        "stage2_train.parquet",
        "stage2_val.parquet",
    }
    if set(expected_hashes) != expected_names:
        raise AssertionError(f"unexpected checksum entries in {data}: {sorted(expected_hashes)}")
    for name, expected_hash in expected_hashes.items():
        actual_hash = sha256(data / name)
        if actual_hash != expected_hash:
            raise AssertionError(f"hash mismatch for {data / name}: {actual_hash} != {expected_hash}")


def main() -> None:
    required = [
        ROOT / "patches" / "bips_counterfactual_rlvr.patch",
        ROOT / "counterfactual_rlvr" / "reward_accuracy.py",
        ROOT / "bips_reproduction" / "reward_paper.py",
        ROOT / "run_counterfactual_rlvr_7b.sh",
        ROOT / "run_bips_reproduction_7b.sh",
        ROOT / "docs" / "BIPS_7B_REPRODUCTION.md",
        ROOT / "docs" / "PUBLISHING.md",
        ROOT / "AGENTS.md",
        CF_DATA / "README.md",
        CF_DATA / "PROVENANCE.md",
        CF_DATA / "PARQUET_SHA256SUMS",
        CF_DATA / "split_manifest.json",
        BIPS_DATA / "README.md",
        BIPS_DATA / "PROVENANCE.md",
        BIPS_DATA / "PARQUET_SHA256SUMS",
        BIPS_DATA / "split_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    cf_manifest = json.loads((CF_DATA / "split_manifest.json").read_text(encoding="utf-8"))
    cf_counts = {
        "stage1": {"total": 12219, "train": 11622, "val": 597},
        "stage2": {"total": 12630, "train": 12007, "val": 623},
    }
    verify_dataset(CF_DATA, cf_counts, cf_manifest["output_sha256"])

    bips_counts = {
        "stage1": {"total": 10781, "train": 10257, "val": 524},
        "stage2": {"total": 14860, "train": 14136, "val": 724},
    }
    verify_dataset(BIPS_DATA, bips_counts, checksum_file(BIPS_DATA / "PARQUET_SHA256SUMS"))

    for script in (
        "run_counterfactual_rlvr_7b.sh",
        "run_counterfactual_rlvr_7b_stage1.sh",
        "run_counterfactual_rlvr_7b_stage2.sh",
        "run_bips_reproduction_7b.sh",
        "run_bips_reproduction_7b_stage1.sh",
        "run_bips_reproduction_7b_stage2.sh",
        "tools/setup_bips_counterfactual_rlvr.sh",
        "tools/merge_final_checkpoint.sh",
        "tools/run_counterfactual_rlvr_7b_pipeline.sh",
        "tools/run_bips_reproduction_7b_pipeline.sh",
        "tools/publish_qwen25vl_7b_github_release.sh",
        "tools/download_qwen25vl_7b_github_release.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    print("Both training tracks, scripts, counts, and all eight Parquet hashes are valid.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"VERIFY FAILED: {error}", file=sys.stderr)
        raise
