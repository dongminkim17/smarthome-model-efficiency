#!/usr/bin/env python3
"""Download the official MIntRec 2022 TSVs and raw MP4 clips, without features.

Uses only Python's standard library. Downloads stay in --output; the repository
contains no dataset media. Google Drive's embedded folder view avoids the normal
folder page's listing limit. Existing verified files and partial transfers resume.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SOURCE = "https://github.com/thuiar/MIntRec"
RAW_ID = "1Q9JKhWPvQoInX00pIoFSaY1n_3MopedY"
SPLITS = {
    "train": ("1TDYifA9ZEw5pORDEx5NuIhL2bzhNkyvU", 1334, 75488),
    "dev": ("1sZFq9wbeTXiR0qAhJyGertKZdnNpc69o", 445, 25594),
    "test": ("1wjlr86Oav9G5oFU2XBuSba6x727OvKbM", 445, 26287),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def is_mp4(path):
    if not path.is_file() or path.stat().st_size < 32:
        return False
    with path.open("rb") as stream:
        return stream.read(12)[4:8] == b"ftyp"


def media_path(root, relative):
    if not re.fullmatch(r"raw_data/S\d{2}/E\d{2}/\d+\.mp4", relative):
        raise ValueError(f"Unexpected media path: {relative!r}")
    return root / relative


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.href = None
        self.parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.parts = []

    def handle_data(self, data):
        if self.href:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, "".join(self.parts).strip()))
            self.href = None


def folder_listing(folder_id):
    url = "https://drive.google.com/embeddedfolderview?" + urlencode({"id": folder_id})
    for attempt in range(4):
        try:
            with urlopen(url, timeout=60) as response:
                parser = Links()
                parser.feed(response.read().decode("utf-8"))
            entries = []
            for href, name in parser.links:
                match = re.match(r"https://drive\.google\.com/(file/d|drive/folders)/([\w-]+)", href)
                if match:
                    entries.append((match[2], name, match[1] == "drive/folders"))
            if not entries:
                raise RuntimeError(f"Empty public folder listing: {folder_id}")
            return entries
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def discover(workers):
    episodes = []
    for season_id, season, is_folder in folder_listing(RAW_ID):
        if not is_folder or not re.fullmatch(r"S\d{2}", season):
            raise ValueError(f"Unexpected season entry: {season!r}")
        for episode_id, episode, is_folder in folder_listing(season_id):
            if not is_folder or not re.fullmatch(r"E\d{2}", episode):
                raise ValueError(f"Unexpected episode entry: {episode!r}")
            episodes.append((season, episode, episode_id))
    paths = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for episode, entries in zip(episodes, pool.map(lambda x: folder_listing(x[2]), episodes)):
            for file_id, name, is_folder in entries:
                if is_folder or not re.fullmatch(r"\d+\.mp4", name):
                    raise ValueError(f"Unexpected clip entry: {name!r}")
                relative = str(PurePosixPath("raw_data", episode[0], episode[1], name))
                paths.setdefault(relative, []).append(file_id)
    return [
        {"path": path, "google_drive_id": ids[0], "alias_google_drive_ids": ids[1:]}
        for path, ids in sorted(paths.items())
    ]


def download(file_id, path, expected_bytes=None, expected_sha256=None):
    if path.exists() and expected_bytes is not None and path.stat().st_size == expected_bytes:
        if expected_sha256 is None or digest(path) == expected_sha256:
            return {"bytes": path.stat().st_size, "sha256": digest(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    url = "https://drive.usercontent.google.com/download?" + urlencode({
        "id": file_id, "export": "download", "confirm": "t"
    })
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Drive returned an HTML page instead of data")
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset:
                        raise RuntimeError("Invalid resumed response range")
                    total = int(match[3])
                    mode = "ab"
                else:
                    offset = 0
                    total = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
                    mode = "wb"
                with partial.open(mode) as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
            size = partial.stat().st_size
            if total is not None and size != total:
                raise RuntimeError(f"Incomplete download: {size} != {total}")
            if expected_bytes is not None and size != expected_bytes:
                raise RuntimeError(f"Unexpected file size: {size} != {expected_bytes}")
            sha256 = digest(partial)
            if expected_sha256 is not None and sha256 != expected_sha256:
                partial.unlink()
                raise RuntimeError("Checksum mismatch")
            if path.suffix == ".mp4" and not is_mp4(partial):
                partial.unlink()
                raise RuntimeError("Missing MP4 ftyp header")
            partial.replace(path)
            return {"bytes": size, "sha256": sha256}
        except Exception as exc:
            if isinstance(exc, HTTPError) and exc.code == 416 and partial.exists():
                partial.unlink()
            if attempt == 3:
                raise RuntimeError(f"Failed {path.name}: {exc}") from exc
            time.sleep(2 ** attempt)


def expected_paths(root):
    expected = set()
    counts = {}
    for split, (_, rows_expected, _) in SPLITS.items():
        with (root / f"{split}.tsv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        if len(rows) != rows_expected or len({row["label"] for row in rows}) != 20:
            raise ValueError(f"Unexpected {split} counts or labels")
        counts[split] = len(rows)
        for row in rows:
            relative = str(PurePosixPath("raw_data", row["season"], row["episode"], row["clip"] + ".mp4"))
            media_path(root, relative)
            if relative in expected:
                raise ValueError(f"Duplicate split assignment: {relative}")
            expected.add(relative)
    return expected, counts


def verify(root, manifest):
    expected, counts = expected_paths(root)
    listed = {entry["path"] for entry in manifest["files"]}
    if listed != expected or len(listed) != 2224:
        raise ValueError(f"Manifest mismatch: missing={len(expected-listed)}, extra={len(listed-expected)}")
    actual = {str(path.relative_to(root)) for path in (root / "raw_data").rglob("*.mp4")}
    if actual != expected:
        raise ValueError(f"Local media mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    for entry in manifest["files"]:
        path = media_path(root, entry["path"])
        if not is_mp4(path) or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Verification failed: {entry['path']}")
    return {"verified_at": now(), "split_counts": counts, "media_files": len(expected),
            "media_bytes": sum(entry["bytes"] for entry in manifest["files"]),
            "all_mp4_headers_valid": True, "all_sha256_valid": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/MIntRec")
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=6)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    manifest_path = root / "download_manifest.json"
    if args.verify_only:
        print(json.dumps(verify(root, json.loads(manifest_path.read_text())), indent=2), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "dataset": "MIntRec (ACM MM 2022)", "source": SOURCE,
        "raw_folder": "https://drive.google.com/drive/folders/" + RAW_ID,
        "created_at": now(), "splits": {}, "files": [],
    }
    for split, (file_id, _, size) in SPLITS.items():
        manifest["splits"][split] = {"google_drive_id": file_id, **download(file_id, root / f"{split}.tsv", size)}
    expected, counts = expected_paths(root)
    if not manifest["files"]:
        print("Discovering complete public raw-media listing...", flush=True)
        manifest["files"] = discover(args.workers)
    listed = {entry["path"] for entry in manifest["files"]}
    if listed != expected or len(manifest["files"]) != 2224:
        raise ValueError(f"Source coverage mismatch: missing={len(expected-listed)}, extra={len(listed-expected)}")
    manifest["split_counts"] = counts
    manifest["duplicate_alias_count"] = sum(len(entry["alias_google_drive_ids"]) for entry in manifest["files"])
    atomic_json(manifest_path, manifest)
    print(f"Downloading {len(listed)} clips; {manifest['duplicate_alias_count']} alias IDs recorded", flush=True)
    started = time.monotonic()
    failed = []
    completed = 0
    def fetch(entry):
        return download(entry["google_drive_id"], media_path(root, entry["path"]), entry.get("bytes"), entry.get("sha256"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, entry): entry for entry in manifest["files"]}
        for future in concurrent.futures.as_completed(pending):
            entry = pending[future]
            try:
                entry.update(future.result())
                completed += 1
            except Exception as exc:
                failed.append({"path": entry["path"], "error": str(exc)})
                print(f"Failed: {entry['path']}: {exc}", flush=True)
            if (completed + len(failed)) % 100 == 0:
                atomic_json(manifest_path, manifest)
                size = sum(entry.get("bytes", 0) for entry in manifest["files"])
                print(f"{completed}/2224 clips; {size/1e9:.3f} GB; {time.monotonic()-started:.0f}s; failures={len(failed)}", flush=True)
    manifest["failed_downloads"] = failed
    atomic_json(manifest_path, manifest)
    if failed:
        raise RuntimeError(f"{len(failed)} downloads failed; rerun to resume")
    result = verify(root, manifest)
    manifest["verification"] = result
    atomic_json(manifest_path, manifest)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
