#!/usr/bin/env python3
"""Render local training history as PNG/SVG and a concise JSON summary."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smarthome_efficiency.tracking import plot_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing metrics.jsonl")
    args = parser.parse_args()
    try:
        outputs = plot_metrics(args.run_dir)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(1, f"Could not plot training history: {error}\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
