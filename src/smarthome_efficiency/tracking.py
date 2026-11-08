"""Local training records and plots; JSONL is the authoritative history."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path


_CSV_FIELDS = ("event", "global_step", "epoch", "elapsed_seconds", "metric", "value")
_CONTEXT_FIELDS = {"event", "global_step", "epoch", "elapsed_seconds", "recorded_at"}


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def read_metrics(run_dir: str | Path) -> list[dict]:
    """Read records in file order, failing explicitly on a damaged history."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_nonfinite)
                if not isinstance(record, dict) or not isinstance(record.get("event"), str):
                    raise ValueError("expected an object with an event name")
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid metrics history at {path}:{line_number}: {error}") from error
            records.append(record)
    return records


def _scalars(record: Mapping, prefix: str = "") -> dict[str, float]:
    result = {}
    for key, value in record.items():
        if not prefix and key in _CONTEXT_FIELDS:
            continue
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_scalars(value, name))
        elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = float(value)
    return result


class Tracker:
    """Append strict JSONL and a long-form scalar CSV; all tracking stays local.

    Resume callers must first restore/truncate metrics.jsonl to their checkpoint.
    CSV is rebuilt from that history when the tracker opens. The purge_step
    argument is retained for caller compatibility; JSONL trimming is caller-owned.
    JSONL is flushed on every event and fsynced on validation and close.
    """

    def __init__(self, output_dir: str | Path, enabled: bool = True, purge_step: int | None = None):
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self._closed = False
        self._json = self._csv_file = None
        if not enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history = read_metrics(self.output_dir)
        # An intact final record may lack its newline after manual recovery.
        path = self.output_dir / "metrics.jsonl"
        if path.exists() and path.stat().st_size:
            with path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) != b"\n"
            if needs_newline:
                with path.open("ab") as handle:
                    handle.write(b"\n")
        self._json = path.open("a", encoding="utf-8")
        self._csv_file = (self.output_dir / "scalars.csv").open("w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDS)
        self._csv.writeheader()
        for record in history:
            self._write_scalars(record)
        self._csv_file.flush()

    def _write_scalars(self, record: dict) -> None:
        context = {key: record.get(key, "") for key in _CSV_FIELDS[:4]}
        for key, value in _scalars(record).items():
            self._csv.writerow({**context, "metric": key, "value": value})

    def log(self, record: dict) -> None:
        if self._closed:
            raise RuntimeError("Tracker is closed")
        if not self.enabled:
            return
        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
            raise ValueError("Metric record must be a dictionary with an event name")
        saved = {"recorded_at": time.time(), **record}
        # Validate serialization before changing any on-disk files.
        line = json.dumps(saved, ensure_ascii=False, allow_nan=False)
        self._json.write(line + "\n")
        self._json.flush()
        self._write_scalars(saved)
        self._csv_file.flush()
        if record["event"] in {"validation", "checkpoint", "complete"}:
            os.fsync(self._json.fileno())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in (self._json, self._csv_file):
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _series(records: list[dict], event: str, key: str) -> tuple[list[float], list[float]]:
    """Keep each metric on its own event steps; never bridge a step rollback."""
    xs, ys = [], []
    previous = None
    for record in records:
        if record.get("event") != event:
            continue
        step = record.get("global_step")
        if not isinstance(step, Real) or isinstance(step, bool) or not math.isfinite(step):
            continue
        value = record
        for part in key.split("/"):
            value = value.get(part) if isinstance(value, Mapping) else None
        if previous is not None and step < previous:
            xs.append(float("nan"))
            ys.append(float("nan"))
        xs.append(float(step))
        ys.append(float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) else float("nan"))
        previous = step
    return xs, ys


def plot_metrics(run_dir: str | Path) -> dict[str, str]:
    """Generate PNG/SVG figures and a JSON summary without requiring a GPU."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    records = read_metrics(run_dir)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def draw(ax, event, key, label, **kwargs):
        xs, ys = _series(records, event, key)
        if any(math.isfinite(y) for y in ys):
            ax.plot(xs, ys, label=label, linewidth=1.3, **kwargs)
            return True
        return False

    panels = [
        ("Training objective and supervised loss", "Loss", [
            ("train", "loss", "Train objective"), ("train", "sft_loss", "Train SFT"),
            ("train", "distillation_loss", "Train KD (T² KL)"),
            ("validation", "sft_loss", "Validation SFT")]),
        ("Target-token loss", "Cross entropy", [
            ("train", "class_loss", "Train class"), ("train", "eos_loss", "Train EOS"),
            ("validation", "class_loss", "Validation class"), ("validation", "eos_loss", "Validation EOS")]),
        ("Validation classification", "Score (0–1)", [
            ("validation", f"{mode}/{metric}", f"{mode}: {metric}")
            for mode in ("logit", "greedy") for metric in ("accuracy", "weighted_f1", "macro_f1")]),
        ("Prediction agreement", "Fraction", [
            ("validation", "agreement/agreement", "All predictions"),
            ("validation", "agreement/agreement_on_valid", "Valid greedy predictions only")]),
        ("Invalid predictions", "Fraction", [
            ("validation", f"{mode}/invalid_rate", mode) for mode in ("logit", "greedy")]),
        ("Learning rate", "Learning rate", [("train", "learning_rate", "LM / default"),
                                               ("train", "encoder_learning_rate", "Encoders")]),
        ("Gradient norm", "Norm", [("train", "grad_norm", "Gradient norm")]),
        ("Training throughput", "Examples / second", [("train", "examples_per_second", "Examples / second")]),
        ("GPU peak memory", "GiB", [
            ("train", "peak_allocated_gib", "Allocated"), ("train", "peak_reserved_gib", "Reserved")]),
        ("Wall time", "Seconds", [("train", "step_seconds", "Training step"), ("validation", "seconds", "Validation pass")]),
    ]
    if any(record.get('event') == 'train' and 'audio_feature_loss' in record for record in records):
        panels.append(("Encoder feature distillation", "Mean cosine distance", [
            ("train", "audio_feature_loss", "Audio"), ("train", "vision_feature_loss", "Vision")]))
    rows = (len(panels) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(14, 3.6 * rows), constrained_layout=True)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax, (title, ylabel, curves) in zip(axes.flat, panels):
        plotted = False
        for event, key, label in curves:
            plotted = draw(ax, event, key, label, marker="o" if event == "validation" else None,
                           markersize=3) or plotted
        ax.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Not recorded", ha="center", va="center", transform=ax.transAxes, color="gray")
        if title in {"Validation classification", "Prediction agreement", "Invalid predictions"}:
            ax.set_ylim(-0.02, 1.02)
    figure.suptitle(run_dir.name, fontsize=15)
    outputs = {}
    try:
        for extension in ("png", "svg"):
            target = run_dir / f"curves.{extension}"
            temporary = run_dir / f".curves.{extension}.tmp"
            figure.savefig(temporary, format=extension, dpi=150)
            temporary.replace(target)
            outputs[extension] = str(target.resolve())
    finally:
        plt.close(figure)
    summary = {
        "record_count": len(records),
        "train_records": sum(record["event"] == "train" for record in records),
        "validation_records": sum(record["event"] == "validation" for record in records),
        "latest_train": next((record for record in reversed(records) if record["event"] == "train"), None),
        "latest_validation": next((record for record in reversed(records) if record["event"] == "validation"), None),
        "last_record": records[-1] if records else None,
        "note": "Metrics use their recorded optimizer steps. Missing metrics are not interpolated; step rollbacks break curves.",
    }
    target = run_dir / "training_summary.json"
    temporary = run_dir / ".training_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    outputs["summary"] = str(target.resolve())
    return outputs
