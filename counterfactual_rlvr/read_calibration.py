#!/usr/bin/env python3
"""Read a required scalar from a calibration JSON artifact."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("key", choices=("tau1", "alpha2"))
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    value = float(payload[args.key])
    if value <= 0:
        raise ValueError(f"{args.key} must be positive, got {value}")
    print(f"{value:.17g}")


if __name__ == "__main__":
    main()
