"""Protect metric provenance through sparse evaluations and checkpoint rollback."""
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smarthome_efficiency.tracking import Tracker, _series, plot_metrics, read_metrics


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_resume_rebuilds_csv_from_restored_authoritative_history(self):
        with Tracker(self.root) as tracker:
            tracker.log({"event": "train", "global_step": 1, "loss": 1.2})
            tracker.log({"event": "validation", "global_step": 2, "logit": {"accuracy": 0.75}, "greedy": {"accuracy": 0.5}})
            tracker.log({"event": "train", "global_step": 3, "loss": 0.4})
        # Simulate restoring an earlier checkpoint after a preemption.
        history = (self.root / "metrics.jsonl").read_text().splitlines()
        (self.root / "metrics.jsonl").write_text("\n".join(history[:2]))
        with Tracker(self.root, purge_step=3) as tracker:
            tracker.log({"event": "train", "global_step": 3, "loss": 0.6})
        records = read_metrics(self.root)
        self.assertEqual(len(records), 3)
        with (self.root / "scalars.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(r["metric"], r["value"]) for r in rows], [
            ("loss", "1.2"), ("logit/accuracy", "0.75"), ("greedy/accuracy", "0.5"), ("loss", "0.6")])
    def test_nonfinite_or_malformed_event_does_not_damage_history(self):
        with Tracker(self.root) as tracker:
            tracker.log({"event": "train", "global_step": 1, "loss": 1.0})
            with self.assertRaises(ValueError):
                tracker.log({"event": "train", "loss": float("nan")})
            with self.assertRaises(ValueError):
                tracker.log({"loss": 1.0})
        self.assertEqual(len(read_metrics(self.root)), 1)
        with (self.root / "metrics.jsonl").open("a") as handle:
            handle.write('{"event":')
        with self.assertRaisesRegex(ValueError, "metrics.jsonl:2"):
            Tracker(self.root)

    def test_tracking_is_local_and_disabled_mode_creates_nothing(self):
        # Logging must remain usable without any ML or hosted tracking package.
        with patch.dict("sys.modules", {"torch": None, "tensorboard": None, "wandb": None, "trackio": None}):
            with Tracker(self.root) as tracker:
                tracker.log({"event": "validation", "global_step": 0, "logit": {"accuracy": 0.5}})
        self.assertEqual({path.name for path in self.root.iterdir()}, {"metrics.jsonl", "scalars.csv"})
        absent = self.root / "disabled"
        with Tracker(absent, enabled=False) as tracker:
            tracker.log({"event": "train", "loss": 1.0})
        self.assertFalse(absent.exists())
        self.assertEqual(read_metrics(self.root)[0]["logit"]["accuracy"], 0.5)

    def test_sparse_metrics_keep_actual_steps_and_break_rollback(self):
        records = [
            {"event": "validation", "global_step": 0, "greedy": {"accuracy": 0.2}},
            {"event": "train", "global_step": 1, "loss": 1.0},
            {"event": "validation", "global_step": 20, "logit": {"accuracy": 0.4}},
            {"event": "validation", "global_step": 40, "greedy": {"accuracy": 0.6}},
            {"event": "validation", "global_step": 30, "greedy": {"accuracy": 0.5}},
        ]
        xs, ys = _series(records, "validation", "greedy/accuracy")
        self.assertEqual(xs[:3], [0, 20, 40])
        self.assertTrue(math.isnan(ys[1]))
        self.assertTrue(math.isnan(xs[3]))
        self.assertEqual((xs[-1], ys[-1]), (30, 0.5))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plots_handle_empty_and_single_mode_validation(self):
        outputs = plot_metrics(self.root)
        self.assertEqual(json.loads(Path(outputs["summary"]).read_text())["record_count"], 0)
        with Tracker(self.root) as tracker:
            tracker.log({"event": "train", "global_step": 1, "loss": 0.8, "grad_norm": 0.2})
            tracker.log({"event": "validation", "global_step": 1, "logit": {"accuracy": 0.5}, "sft_loss": 1.2})
        outputs = plot_metrics(self.root)
        self.assertEqual(Path(outputs["png"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("<svg", Path(outputs["svg"]).read_text())
        summary = json.loads(Path(outputs["summary"]).read_text())
        self.assertEqual(summary["validation_records"], 1)
        self.assertNotIn("greedy", summary["latest_validation"])


    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_feature_losses_are_retained_in_saved_curves(self):
        with Tracker(self.root) as tracker:
            tracker.log({"event":"train", "global_step":1, "loss":.8,
                         "audio_feature_loss":.4, "vision_feature_loss":.2})
        outputs=plot_metrics(self.root)
        self.assertIn("Encoder feature distillation",Path(outputs["svg"]).read_text())
        self.assertEqual(json.loads(Path(outputs["summary"]).read_text())["latest_train"]["audio_feature_loss"],.4)


if __name__ == "__main__":
    unittest.main()
