#!/usr/bin/env python3
"""Resume verified BF16 shards from the existing immutable source manifest.

No pruning: original source is retained. Intended to run on the NAS itself,
because Z4's canonical NAS mount is read-only.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def partial_bytes(root):
    total = 0
    for path in root.glob("*.safetensors.part"):
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            # A worker may atomically publish this file during the inventory.
            continue
    return total


def fetch(root, item, logdir):
    name = item["path"]
    if Path(name).name != name:
        raise ValueError("Unexpected source filename")
    destination = root / name
    if destination.exists():
        if destination.stat().st_size == item["size"] and digest(destination) == item["sha256"]:
            return name, "present"
        raise ValueError(f"Existing source shard failed identity check: {name}")
    partial = root / (name + ".part")
    url = f"https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{REVISION}/{name}"
    with (logdir / (name + ".fetch.log")).open("ab") as log:
        subprocess.run(["wget", "--continue", "--tries=20", "--timeout=30", "--read-timeout=60", "--no-verbose", "--output-document", str(partial), url], stdout=log, stderr=log, check=True)
    if partial.stat().st_size != item["size"] or digest(partial) != item["sha256"]:
        raise ValueError(f"Downloaded source shard failed identity check: {name}")
    os.replace(partial, destination)
    return name, "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("jobs must be 1..8")
    manifest = json.loads((args.source / "q38-fetch-manifest.json").read_text())
    if manifest["repo"] != "Qwen/Qwen3.8-Flash-Next" or manifest["commit"] != REVISION:
        raise ValueError("Source manifest differs from pinned original model")
    files = [f for f in manifest["files"] if f["path"].endswith(".safetensors")]
    if len(files) != 131 or any(not f.get("sha256") for f in files):
        raise ValueError("Expected 131 LFS-hashed original source shards")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    # Front-load a few ordinary/expert shards for genuine single-layer probes.
    priority = {1: 0, 2: 1, 3: 2, 57: 3, 58: 4, 59: 5}
    files.sort(key=lambda f: (priority.get(int(f["path"].split("-")[1]), 1000), f["path"]))
    started = time.time()
    state = {"status": "downloading", "pid": os.getpid(), "source": str(args.source), "revision": REVISION, "required_shards": 131, "verified_shards": 0, "verified_bytes": 0, "total_bytes": sum(f["size"] for f in files), "jobs": args.jobs, "started_unix": started, "errors": []}
    sizes = {f["path"]: f["size"] for f in files}
    atomic_json(args.state_dir / "fetch-status.json", state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(fetch, args.source, f, args.state_dir): f for f in files}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                try:
                    name, action = future.result()
                    state["verified_shards"] += 1
                    state["verified_bytes"] += sizes[name]
                    print(json.dumps({"file": name, "action": action, "verified_shards": state["verified_shards"]}), flush=True)
                except Exception as exc:
                    state["errors"].append({"file": item["path"], "error": type(exc).__name__ + ": " + str(exc)})
            state["updated_unix"] = time.time()
            state["partial_bytes"] = partial_bytes(args.source)
            atomic_json(args.state_dir / "fetch-status.json", state)
    state["status"] = "complete" if state["verified_shards"] == 131 and not state["errors"] else "failed"
    state["elapsed_seconds"] = time.time() - started
    atomic_json(args.state_dir / "fetch-status.json", state)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
