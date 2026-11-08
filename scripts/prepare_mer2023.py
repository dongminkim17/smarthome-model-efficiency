#!/usr/bin/env python3
"""Prepare official MER2023 Train&Val and the labeled MER-SEMI evaluation subset.

The protected README and downloaded archives remain outside the repository.
Use --remote for selective downloading (requests, pyzipper, and libarchive-c required).
The full-archive fallback needs 7zip. The archive
password is read from the approved local README and is never logged by this tool.
No development split is imposed here; the 834 SEMI examples are evaluation data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

REVISION = "9d7a285654c722abd30d3c67b6ade477f1a75e16"
LABEL_SHA256 = "a3ad0518053b31e7b1f53bc659652fdacaea1ef34224516d384833888effc8fe"
PART_SIZES = {**{f"mer2023train.z{i:02d}": 21474836480 for i in range(1, 7)},
              "mer2023train.zip": 10785113893}
LABELS = {"neutral", "angry", "happy", "sad", "worried", "surprise"}


def passwords(archive_root: Path) -> dict[str, str]:
    content = (archive_root / "README_AFTER_APPROVAL.md").read_text()
    result = dict(re.findall(r"\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", content))
    if not {"test-labels.zip", "mer2023train.zip"} <= result.keys():
        raise ValueError("Cannot locate archive credentials in the approved README")
    return result


def validate_labels(content: bytes, expected: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} labeled examples, found {len(rows)}")
    ids = [row["name"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample identifiers in labels")
    invalid = {row["discrete"] for row in rows} - LABELS
    if invalid:
        raise ValueError(f"Unexpected emotion labels: {sorted(invalid)}")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        raise ValueError("Unsafe sample identifier")
    return rows


def safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or "\n" in name:
        raise ValueError("Unsafe archive member path")


def run_7z(binary: str, arguments: list[str]) -> str:
    # Do not use check=True: CalledProcessError would include the password argv.
    result = subprocess.run([binary, *arguments], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"7-Zip operation failed (exit {result.returncode}); details suppressed")
    return result.stdout


def archive_entries(binary: str, archive: Path, password: str) -> list[dict[str, str]]:
    output = run_7z(binary, ["l", "-slt", "-ba", f"-p{password}", str(archive)])
    entries = []
    for block in output.strip().split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" in fields:
            safe_member(fields["Path"])
            entries.append(fields)
    return entries


def extract(binary: str, archive: Path, password: str, names: list[str],
            raw: Path, list_path: Path) -> None:
    for name in names:
        safe_member(name)
    list_path.write_text("\n".join(names) + "\n")
    raw.mkdir(parents=True, exist_ok=True)
    run_7z(binary, ["x", "-y", "-bd", "-scsUTF-8", f"-p{password}",
                    f"-o{raw}", str(archive), f"@{list_path}"])


def write_manifest(path: Path, rows: list[dict[str, str]], media: dict[str, dict[str, str]],
                   raw: Path, data_root: Path, split: str) -> None:
    records = []
    for row in rows:
        name = row["name"]
        entry = media[name]
        video = raw / entry["Path"]
        if not video.is_file() or video.stat().st_size != int(entry["Size"]):
            raise ValueError(f"Missing or truncated extracted video: {name}")
        records.append({"id": name, "label": row["discrete"], "split": split,
                        "video": video.relative_to(data_root).as_posix()})
    path.write_text("".join(json.dumps(record) + "\n" for record in records))



def central_entries(content: bytes) -> list[dict]:
    """Read this split ZIP's public central directory, including ZIP64 offsets."""
    entries, cursor = [], 0
    while cursor < len(content):
        fields = struct.unpack("<4s6H3L5H2L", content[cursor:cursor + 46])
        if fields[0] != b"PK\x01\x02":
            raise ValueError("Invalid cached ZIP central directory")
        name_length, extra_length, comment_length = fields[10:13]
        end = cursor + 46 + name_length + extra_length + comment_length
        record = content[cursor:end]
        name = record[46:46 + name_length].decode("utf-8")
        safe_member(name)
        extra = record[46 + name_length:46 + name_length + extra_length]
        offset, disk, size, packed = fields[16], fields[13], fields[9], fields[8]
        at = 0
        while at < len(extra):
            tag, length = struct.unpack("<HH", extra[at:at + 4])
            payload = extra[at + 4:at + 4 + length]
            if tag == 1:
                q = 0
                if size == 0xFFFFFFFF:
                    size = struct.unpack_from("<Q", payload, q)[0]; q += 8
                if packed == 0xFFFFFFFF:
                    packed = struct.unpack_from("<Q", payload, q)[0]; q += 8
                if offset == 0xFFFFFFFF:
                    offset = struct.unpack_from("<Q", payload, q)[0]; q += 8
                if disk == 0xFFFF:
                    disk = struct.unpack_from("<L", payload, q)[0]
            at += 4 + length
        entries.append({"Path": name, "Size": size, "packed": packed,
                        "offset": offset, "disk": disk, "record": record, "crc32": fields[7]})
        cursor = end
    return entries


class RemoteArchive:
    """Fetch selected encrypted ZIP members; refuses whole-archive HTTP 200 responses."""
    def __init__(self, password: str):
        self.password = password.encode()
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError("Remote extraction requires the approved HF_TOKEN environment variable")
        self.thread = threading.local()
        self.part_names = list(PART_SIZES)

    def fetch(self, filename: str, start: int, end: int) -> bytes:
        import requests
        if not hasattr(self.thread, "session"):
            self.thread.session = requests.Session()
        url = f"https://huggingface.co/datasets/MERChallenge/MER2023/resolve/{REVISION}/{filename}"
        for attempt in range(3):
            try:
                with self.thread.session.get(url, headers={
                        "Authorization": f"Bearer {self.token}",
                        "Range": f"bytes={start}-{end}"}, timeout=(15, 90), stream=True) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"Expected HTTP 206 for range request; got {response.status_code}")
                    expected = f"bytes {start}-{end}/{PART_SIZES[filename]}"
                    if response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server returned an unexpected byte range")
                    payload = response.content
                    if len(payload) != end - start + 1:
                        raise RuntimeError("Truncated range response")
                    return payload
            except (requests.RequestException, RuntimeError):
                if attempt == 2:
                    raise RuntimeError(f"Range download failed for {filename}; private URL details suppressed") from None
                time.sleep(attempt + 1)
        raise RuntimeError("Unreachable retry state")

    def member_bytes(self, entry: dict) -> bytes:
        # Media members are small; leave padding for differing local-header extras.
        disk, offset = entry["disk"], entry["offset"]
        remaining = entry["packed"] + 46 + len(entry["Path"].encode()) + 1024
        chunks = []
        while remaining:
            filename = self.part_names[disk]
            count = min(remaining, PART_SIZES[filename] - offset)
            chunks.append(self.fetch(filename, offset, offset + count - 1))
            remaining -= count
            disk += 1
            offset = 0
        body = b"".join(chunks)
        return self.decode_member(entry, body)

    def decode_member(self, entry: dict, body: bytes) -> bytes:
        native_library = Path(sys.prefix) / "lib" / "libarchive.so"
        if native_library.exists():
            os.environ.setdefault("LIBARCHIVE", str(native_library))
        try:
            import libarchive
        except ImportError:
            raise RuntimeError("Remote mode requires libarchive-c for native ZIP decryption") from None
        if body[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        name_len, extra_len = struct.unpack_from("<HH", body, 26)
        end = 30 + name_len + extra_len + entry["packed"]
        if end > len(body):
            raise ValueError("Unexpectedly large local ZIP header; use full archive extraction")
        body = body[:end]
        center = bytearray(entry["record"])
        struct.pack_into("<H", center, 34, 0)  # disk number in a synthetic single-file ZIP
        struct.pack_into("<L", center, 42, 0)  # local header offset
        footer = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, len(center), len(body), 0)
        with libarchive.memory_reader(body + center + footer, format_name="zip", passphrase=self.password) as archive:
            members = iter(archive)
            member = next(members)
            if member.pathname != entry["Path"]:
                raise ValueError("Unexpected decrypted member name")
            content = b"".join(member.get_blocks())
            if next(members, None) is not None:
                raise ValueError("Unexpected additional archive member")
        if zlib.crc32(content) != entry["crc32"]:
            raise ValueError("Decrypted member CRC32 differs from the official archive")
        if len(content) != entry["Size"]:
            raise ValueError("Decrypted archive member has unexpected size")
        return content


def prepare_remote(args, root: Path, pwd: dict[str, str], semi: list[dict[str, str]]) -> None:
    archives, raw, labels_dir, manifests = (root / n for n in ("archives", "raw", "labels", "manifests"))
    remote = RemoteArchive(pwd["mer2023train.zip"])
    cache = archives / "mer2023train.central-directory.bin"
    if not cache.exists():
        cache.write_bytes(remote.fetch("mer2023train.zip", 10775909764, 10785113794))
    content = cache.read_bytes()
    if len(content) != 9204031:
        raise ValueError("Invalid cached central-directory size")
    entries = central_entries(content)
    label_entries = [entry for entry in entries if entry["Path"] == "train-label.csv"]
    if len(label_entries) != 1:
        raise ValueError("Expected one train-label.csv")
    train_content = remote.member_bytes(label_entries[0])
    train = validate_labels(train_content, 3373)
    (labels_dir / "train-label.csv").write_bytes(train_content)
    train_ids, semi_ids = ({row["name"] for row in rows} for rows in (train, semi))
    if train_ids & semi_ids:
        raise ValueError("Training/evaluation sample overlap")
    wanted = train_ids | semi_ids
    media = {}
    for entry in entries:
        path = PurePosixPath(entry["Path"])
        if path.stem in wanted and path.suffix.lower() in {".avi", ".mp4"}:
            if path.stem in media:
                raise ValueError("Duplicate media member")
            media[path.stem] = entry
    if set(media) != wanted:
        raise ValueError("Archive does not contain every labeled sample")
    print(json.dumps({"remote_selected_clips": len(media),
                      "compressed_bytes": sum(e["packed"] for e in media.values()),
                      "uncompressed_bytes": sum(e["Size"] for e in media.values())}), flush=True)
    selected = list(media.values())
    if args.max_clips:
        selected = selected[:args.max_clips]
    def present(entry):
        destination = raw / entry["Path"]
        return destination.exists() and destination.stat().st_size == entry["Size"]
    def store(entry, data):
        destination = raw / entry["Path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(data)
        temporary.replace(destination)
    pending = [entry for entry in selected if not present(entry)]
    batches = []
    for entry in pending:
        if batches:
            previous = batches[-1][-1]
            first = batches[-1][0]
            gap = entry["offset"] - (previous["offset"] + previous["packed"])
            total = entry["offset"] + entry["packed"] + 2048 - first["offset"]
            part_size = PART_SIZES[remote.part_names[entry["disk"]]]
            if (entry["disk"] == first["disk"] and 0 <= gap <= 4096
                    and total < 16 * 1024 * 1024 and first["offset"] + total <= part_size):
                batches[-1].append(entry)
                continue
        batches.append([entry])
    def process_batch(batch):
        if len(batch) == 1:
            store(batch[0], remote.member_bytes(batch[0]))
        else:
            start = batch[0]["offset"]
            last = batch[-1]
            end = last["offset"] + last["packed"] + len(last["Path"].encode()) + 46 + 1024 - 1
            payload = remote.fetch(remote.part_names[last["disk"]], start, end)
            for entry in batch:
                store(entry, remote.decode_member(entry, payload[entry["offset"] - start:]))
        return len(batch)
    completed = len(selected) - len(pending)
    print(json.dumps({"already_present": completed, "http_batches": len(batches)}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        last_report = completed
        for future in as_completed(futures):
            completed += future.result()
            if completed - last_report >= 100 or completed == len(selected):
                print(json.dumps({"completed": completed, "selected": len(selected)}), flush=True)
                last_report = completed
    if args.max_clips and len(selected) < len(media):
        print("Limited smoke extraction completed; full dataset manifests were not written.")
        return
    write_manifest(manifests / "trainval.jsonl", train, media, raw, root, "trainval")
    write_manifest(manifests / "semi_test.jsonl", semi, media, raw, root, "semi_test")
    (manifests / "dataset_info.json").write_text(json.dumps({
        "dataset": "MER2023", "revision": REVISION, "source": "MERChallenge/MER2023",
        "trainval_count": len(train), "semi_test_count": len(semi),
        "metric": "weighted_f1", "labels": sorted(LABELS),
        "video_paths_relative_to": str(root), "acquisition": "authenticated HTTP ranges; encrypted ZIP extraction with CRC32 verification",
        "note": "Select development data only from trainval; semi_test is held-out evaluation."
    }, indent=2) + "\n")
    print(json.dumps({"prepared": str(root), "trainval": len(train), "semi_test": len(semi)}))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/MER2023")
    parser.add_argument("--extract-media", action="store_true",
                        help="Extract only official training clips and 834 SEMI evaluation clips")
    parser.add_argument("--sevenzip", help="7z/7zz executable; otherwise discovered from PATH")
    parser.add_argument("--remote", action="store_true", help="Fetch only the labeled clips with authenticated HTTP ranges")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-clips", type=int, help="Smoke-test remote extraction on at most this many clips")
    args = parser.parse_args()
    if args.max_clips is not None and not args.remote:
        parser.error("max-clips requires remote mode")
    if args.workers < 1 or (args.max_clips is not None and args.max_clips < 1):
        parser.error("workers and max-clips must be positive")
    root = args.data_root.resolve()
    archives, labels_dir, manifests, raw = (root / name for name in ("archives", "labels", "manifests", "raw"))
    pwd = passwords(archives)
    label_archive = archives / "test-labels.zip"
    if hashlib.sha256(label_archive.read_bytes()).hexdigest() != LABEL_SHA256:
        raise ValueError("test-labels.zip checksum does not match the official approved README")
    try:
        import pyzipper
    except ImportError:
        raise RuntimeError("Install pyzipper in the project environment before running this script") from None
    with pyzipper.AESZipFile(label_archive) as archive:
        semi_content = archive.read("test3-label.csv", pwd=pwd["test-labels.zip"].encode())
    semi = validate_labels(semi_content, 834)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    (labels_dir / "test3-label.csv").write_bytes(semi_content)
    print(json.dumps({"split": "semi_test", "count": len(semi),
                      "labels": dict(sorted(Counter(row["discrete"] for row in semi).items()))}))
    if args.remote:
        prepare_remote(args, root, pwd, semi)
        return
    if not args.extract_media:
        print("Labels verified. Add --extract-media after all seven media archive parts are downloaded.")
        return
    for name, expected in PART_SIZES.items():
        part = archives / name
        if not part.is_file() or part.stat().st_size != expected:
            raise ValueError(f"Archive part is incomplete or missing: {name}")
    binary = args.sevenzip or shutil.which("7zz") or shutil.which("7z")
    if not binary:
        raise RuntimeError("Install 7zip in the project environment or supply --sevenzip")
    archive = archives / "mer2023train.zip"
    entries = archive_entries(binary, archive, pwd[archive.name])
    train_labels = [entry["Path"] for entry in entries if PurePosixPath(entry["Path"]).name == "train-label.csv"]
    if len(train_labels) != 1:
        raise ValueError("Expected exactly one official train-label.csv in media archive")
    extract(binary, archive, pwd[archive.name], train_labels, raw, manifests / "train_label_members.txt")
    train_content = (raw / train_labels[0]).read_bytes()
    train = validate_labels(train_content, 3373)
    (labels_dir / "train-label.csv").write_bytes(train_content)
    train_ids, semi_ids = ({row["name"] for row in rows} for rows in (train, semi))
    if train_ids & semi_ids:
        raise ValueError("Training/evaluation sample overlap")
    selected_ids = train_ids | semi_ids
    media: dict[str, dict[str, str]] = {}
    for entry in entries:
        path = PurePosixPath(entry["Path"])
        if path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"} and path.stem in selected_ids:
            if path.stem in media:
                raise ValueError(f"Ambiguous media for sample: {path.stem}")
            media[path.stem] = entry
    missing = selected_ids - media.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} labeled video members in archive")
    unpacked_bytes = sum(int(entry["Size"]) for entry in media.values())
    print(json.dumps({"extracting_clips": len(media), "uncompressed_bytes": unpacked_bytes}))
    extract(binary, archive, pwd[archive.name], [entry["Path"] for entry in media.values()],
            raw, manifests / "selected_media_members.txt")
    write_manifest(manifests / "trainval.jsonl", train, media, raw, root, "trainval")
    write_manifest(manifests / "semi_test.jsonl", semi, media, raw, root, "semi_test")
    info = {"dataset": "MER2023", "source": "MERChallenge/MER2023", "revision": REVISION,
            "trainval_count": len(train), "semi_test_count": len(semi),
            "metric": "weighted_f1", "labels": sorted(LABELS),
            "video_paths_relative_to": str(root), "selected_media_bytes": unpacked_bytes,
            "note": "Select development data only from trainval; semi_test is held-out evaluation."}
    (manifests / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"prepared": str(root), "trainval": len(train), "semi_test": len(semi)}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
