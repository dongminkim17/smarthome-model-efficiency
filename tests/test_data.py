"""Protocol tests: taxonomy and keeping pilot development separate from SEMI."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from smarthome_efficiency.data import LABELS, MINTREC_COARSE, load_records, select_records


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_mer(self):
        root = self.root / "MER2023"
        (root / "manifests").mkdir(parents=True)
        (root / "raw" / "train").mkdir(parents=True)
        rows = []
        for label in LABELS["mer6"]:
            for index in range(10):
                name = f"{label}_{index:02}"
                relative = f"raw/train/{name}.avi"
                (root / relative).touch()
                rows.append({"id": name, "label": label, "video": relative})
        # Deliberately no SEMI manifest: requesting train/dev must not need it.
        (root / "manifests" / "trainval.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in reversed(rows))
        )
        return root

    def test_official_binary_mapping(self):
        self.assertEqual(set(MINTREC_COARSE), set(LABELS["mintrec20"]))
        emotion = {label for label, coarse in MINTREC_COARSE.items() if coarse == "emotions_attitudes"}
        self.assertEqual(emotion, {
            "Complain", "Praise", "Apologise", "Thank", "Criticize", "Care",
            "Agree", "Taunt", "Flaunt", "Joke", "Oppose",
        })
        self.assertEqual(MINTREC_COARSE["Comfort"], "goals")

    def test_mintrec_keeps_official_splits_and_transcripts(self):
        root = self.root / "MIntRec"
        (root / "raw_data" / "S01" / "E01").mkdir(parents=True)
        for split, clip, label in [("train", "1", "Care"), ("dev", "2", "Comfort"), ("test", "3", "Apologise")]:
            (root / "raw_data" / "S01" / "E01" / f"{clip}.mp4").touch()
            with (root / f"{split}.tsv").open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["season", "episode", "clip", "text", "label"])
                writer.writerow(["S01", "E01", clip, "Some transcript.", label])
        splits = [load_records("mintrec2", split, data_root=self.root) for split in ("train", "dev", "test")]
        self.assertEqual(len({rows[0]["id"] for rows in splits}), 3)
        self.assertEqual(splits[0][0]["label"], "emotions_attitudes")
        self.assertEqual(splits[1][0]["label"], "goals")
        self.assertEqual(splits[0][0]["transcript"], "Some transcript.")
        self.assertTrue(Path(splits[0][0]["video"]).is_absolute())
        self.assertEqual(load_records("mintrec20", "test", data_root=self.root)[0]["label"], "Apologise")

    def test_mer_disjoint_stratified_deterministic_and_no_semi_read(self):
        self.make_mer()
        train = load_records("mer6", "train", data_root=self.root)
        dev = load_records("mer6", "dev", data_root=self.root)
        self.assertEqual((len(train), len(dev)), (48, 12))
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in dev})
        self.assertEqual({row["label"] for row in dev}, set(LABELS["mer6"]))
        self.assertTrue(all(row["transcript"] is None for row in train + dev))
        self.assertEqual(dev, load_records("mer6", "dev", data_root=self.root))
        self.assertNotEqual(dev, load_records("mer6", "dev", seed=43, data_root=self.root))
        with self.assertRaises(FileNotFoundError):
            load_records("mer6", "test", data_root=self.root)

    def test_subset_preserves_classes_without_replacement(self):
        self.make_mer()
        train = load_records("mer6", "train", data_root=self.root)
        subset = select_records(train, 12)
        self.assertEqual(len({row["id"] for row in subset}), 12)
        self.assertEqual({label: sum(row["label"] == label for row in subset) for label in LABELS["mer6"]}, {label: 2 for label in LABELS["mer6"]})
        self.assertEqual(subset, select_records(list(reversed(train)), 12))
        self.assertEqual(train, select_records(train, None))
        with self.assertRaises(ValueError):
            select_records(train, 0)

    def test_subset_preserves_imbalanced_proportions_by_default(self):
        records = [
            {"id": str(index), "label": "major" if index < 90 else "minor", "video": "unused", "transcript": None}
            for index in range(100)
        ]
        subset = select_records(records, 20)
        self.assertEqual(sum(row["label"] == "major" for row in subset), 18)
        self.assertEqual(sum(row["label"] == "minor" for row in subset), 2)
        balanced = select_records(records, 20, balanced=True)
        self.assertEqual(sum(row["label"] == "minor" for row in balanced), 10)
        self.assertEqual(len({row["id"] for row in balanced}), 20)
        self.assertEqual({row["label"] for row in select_records(records, 2)}, {"major", "minor"})

    def test_subset_smaller_than_class_count_maximizes_coverage(self):
        records = [{"id": str(index), "label": str(index // 3), "video": "unused", "transcript": None} for index in range(60)]
        subset = select_records(records, 12)
        self.assertEqual(len(subset), 12)
        self.assertEqual(len({row["label"] for row in subset}), 12)

    def test_missing_media_and_duplicate_ids_fail(self):
        root = self.make_mer()
        manifest = root / "manifests" / "trainval.jsonl"
        lines = manifest.read_text().splitlines()
        manifest.write_text("\n".join(lines + [lines[0]]) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate example"):
            load_records("mer6", "train", data_root=self.root)
        manifest.write_text("\n".join(lines) + "\n")
        (root / json.loads(lines[0])["video"]).unlink()
        with self.assertRaises(FileNotFoundError):
            load_records("mer6", "train", data_root=self.root)


if __name__ == "__main__":
    unittest.main()
