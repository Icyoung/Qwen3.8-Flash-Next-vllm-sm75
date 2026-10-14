#!/usr/bin/env python3
"""Render a pinned build sequence. Builds run only with --run."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(engine):
    lock = json.loads((ROOT / "config/source-lock.json").read_text())
    engine = Path(engine).resolve()
    if not (engine / "docker/Dockerfile").is_file():
        raise ValueError("Clone the pinned VLLM-SM75 source before building")
    commit = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["engine"]["commit"]:
        raise ValueError(f"Engine commit differs from source lock: {commit}")
    if subprocess.check_output(["git", "-C", str(engine), "status", "--porcelain"], text=True).strip():
        raise ValueError("Engine checkout must be clean")
    return [
        ["docker", "build", "--file", str(engine / "docker/Dockerfile"), "--build-arg", "MAX_JOBS=1", "--build-arg", f"SOURCE_REVISION={commit}", "--build-arg", f"BASE_IMAGE={lock['cuda_base_image']}", "--tag", "q38-sm75-pinned:beta2", str(engine)],
        ["docker", "build", "--file", str(ROOT / "docker/Dockerfile.4xt10-ssd"), "--build-arg", "SM75_BASE=q38-sm75-pinned:beta2", "--tag", "q38-sm75-4xt10-ssd:dev", str(ROOT)],
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / ".upstream/VLLM-SM75")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    for cmd in commands(args.engine):
        print(shlex.join(cmd), flush=True)
        if args.run:
            subprocess.run(cmd, check=True)
