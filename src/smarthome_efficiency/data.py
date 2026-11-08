"""Small, explicit dataset adapters for the Qwen2.5-Omni pilots.

MIntRec taxonomy: https://github.com/thuiar/MIntRec/blob/main/data/__init__.py
MER development examples come exclusively from the official Train&Val set.
SEMI labels are read only when the caller explicitly requests split="test".
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from sklearn.model_selection import train_test_split


class Record(TypedDict):
    id: str
    video: str
    label: str
    transcript: str | None


LABELS = {
    "mintrec2": ("emotions_attitudes", "goals"),
    "mintrec20": (
        "Complain", "Praise", "Apologise", "Thank", "Criticize", "Agree",
        "Taunt", "Flaunt", "Joke", "Oppose", "Comfort", "Care", "Inform",
        "Advise", "Arrange", "Introduce", "Leave", "Prevent", "Greet",
        "Ask for help",
    ),
    "mer6": ("neutral", "angry", "happy", "sad", "worried", "surprise"),
}

# Match the official binary_maps exactly: Comfort is Goal, Care is Emotion.
MINTREC_COARSE = {
    label: "emotions_attitudes" if label in {
        "Complain", "Praise", "Apologise", "Thank", "Criticize", "Care",
        "Agree", "Taunt", "Flaunt", "Joke", "Oppose",
    } else "goals"
    for label in LABELS["mintrec20"]
}
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _validate(records: list[Record], task: str) -> list[Record]:
    seen: set[str] = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError(f"Duplicate example id: {record['id']}")
        seen.add(record["id"])
        if record["label"] not in LABELS[task]:
            raise ValueError(f"Unexpected {task} label: {record['label']!r}")
        if not Path(record["video"]).is_file():
            raise FileNotFoundError(record["video"])
    if not records:
        raise ValueError("Dataset split is empty")
    return sorted(records, key=lambda record: record["id"])


def _load_mintrec(root: Path, task: str, split: str) -> list[Record]:
    records = []
    with (root / f"{split}.tsv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"season", "episode", "clip", "text", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"MIntRec TSV lacks columns: {sorted(required)}")
        for row in reader:
            label = row["label"]
            if label not in MINTREC_COARSE:
                raise ValueError(f"Unknown MIntRec intent: {label!r}")
            records.append({
                "id": "_".join((row["season"], row["episode"], row["clip"])),
                "video": str(root / "raw_data" / row["season"] / row["episode"] / f"{row['clip']}.mp4"),
                "label": MINTREC_COARSE[label] if task == "mintrec2" else label,
                "transcript": row["text"],
            })
    return _validate(records, task)


def _load_mer_manifest(root: Path, name: str) -> list[Record]:
    from .mer_transcripts import validate_input_manifest
    protocol = validate_input_manifest(root, name)
    records = []
    with (root / "manifests" / f"{name}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if protocol and not isinstance(row.get("transcript"), str):
                raise ValueError("AVT manifest requires explicit text, including author-empty strings")
            records.append({
                "id": row["id"], "video": str((root / row["video"]).resolve()),
                "label": row["label"], "transcript": row.get("transcript"),
            })
    return _validate(records, "mer6")


def load_records(
    task: str,
    split: str,
    seed: int = 42,
    mer_dev_fraction: float = 0.2,
    *,
    data_root: str | Path | None = None,
    mer_partition: str | None = None,
) -> list[Record]:
    """Read one task split, with absolute media paths and stable label names.

    MIntRec uses official train/dev/test TSVs. MER train/dev is a deterministic
    stratified partition of sorted Train&Val IDs, with ceil(N * fraction) dev
    examples. This pilot holdout differs from official five-fold evaluation.
    Requesting MER test explicitly reads the 834 held-out SEMI annotations;
    train/dev never opens that manifest. nested_90_10 retains a stratified
    10% of the full pool from the historical 20% dev set; all old train IDs
    remain in train, so historical models can safely initialize new runs.
    """
    if task not in LABELS:
        raise ValueError(f"Unknown task {task!r}; expected one of {tuple(LABELS)}")
    if split not in ("train", "dev", "test"):
        raise ValueError(f"Unknown split {split!r}; expected train, dev, or test")
    if mer_partition is not None and (task != 'mer6' or mer_partition != 'nested_90_10'):
        raise ValueError('mer_partition must be nested_90_10 and is only valid for MER')
    if mer_partition is not None and mer_dev_fraction != 0.2:
        raise ValueError('nested_90_10 must retain the original 80/20 parent partition')
    root = (Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT).resolve()
    dataset_root = (root / ("MER2023" if task == "mer6" else "MIntRec")).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    if task.startswith("mintrec"):
        return _load_mintrec(dataset_root, task, split)
    if split == "test":
        return _load_mer_manifest(dataset_root, "semi_test")
    if not 0 < mer_dev_fraction < 1:
        raise ValueError("mer_dev_fraction must be between 0 and 1")
    records = _load_mer_manifest(dataset_root, "trainval")
    train, dev = train_test_split(
        records, test_size=mer_dev_fraction, random_state=seed,
        stratify=[record["label"] for record in records],
    )
    if mer_partition == 'nested_90_10':
        # Reused teachers/checkpoints have never seen ANY old development ID.
        # Select retained dev IDs by labels and seed, without model predictions.
        old_dev = sorted(dev, key=lambda record: record['id'])
        transferred, retained = train_test_split(
            old_dev, test_size=math.ceil(len(records) * 0.1), random_state=seed,
            stratify=[r['label'] for r in old_dev],
        )
        train, dev = train + transferred, retained
    if {row["id"] for row in train} & {row["id"] for row in dev}:
        raise AssertionError("MER train/dev partition overlaps")
    return sorted(train if split == "train" else dev, key=lambda record: record["id"])


def select_records(
    records: list[Record], limit: int | None, seed: int = 42,
    *, balanced: bool = False,
) -> list[Record]:
    """Choose a reproducible stratified pilot subset without replacement.

    By default, integer quotas approximate the input class proportions while
    retaining every class when the limit permits. If the limit is smaller than
    the number of classes, include one example from each of the largest classes
    (seeded tie breaking). balanced=True instead allocates equal class quotas
    where possible, useful for small overfitting diagnostics. A None limit, or
    one >= N, keeps all examples.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    if limit is None or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item["id"]):
        groups[record["label"]].append(record)
    classes = sorted(groups)
    rng.shuffle(classes)
    for group in groups.values():
        rng.shuffle(group)
    if balanced:
        targets = {label: limit / len(classes) for label in classes}
    else:
        targets = {label: limit * len(groups[label]) / len(records) for label in classes}
    quotas = {label: 0 for label in classes}
    # At most one example per class until all classes are represented. This
    # protects rare-class coverage even when rounding would otherwise erase it.
    for label in sorted(classes, key=lambda item: targets[item], reverse=True)[:limit]:
        quotas[label] = 1
    for _ in range(limit - sum(quotas.values())):
        eligible = [label for label in classes if quotas[label] < len(groups[label])]
        label = max(eligible, key=lambda item: targets[item] - quotas[item])
        quotas[label] += 1
    selected = [record for label in classes for record in groups[label][:quotas[label]]]
    rng.shuffle(selected)
    return selected
