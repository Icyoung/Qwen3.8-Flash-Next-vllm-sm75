#!/usr/bin/env python3
"""Render the experimental bounded-memory TP4 command. Default: dry run."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(model, manifest, image="q38-sm75-4xt10-ssd:dev"):
    model, manifest = Path(model).resolve(), Path(manifest).resolve()
    profile = json.loads((ROOT / "run/profiles/flash-next-tp4-32k-ssd.json").read_text())
    cmd = ["docker", "run", "--rm", "--name", "q38-4xt10-ssd", "--gpus", '"device=0,1,2,3"', "--ipc", "private", "--shm-size", "512m", "--memory", str(profile["container_memory_bytes"]), "--memory-swap", str(profile["container_memory_bytes"]), "--ulimit", "memlock=-1", "--publish", "127.0.0.1:18440:8000", "--mount", f"type=bind,src={model},dst={model},readonly", "--mount", f"type=bind,src={manifest.parent},dst={manifest.parent},readonly"]
    for key, value in {**profile["env"], "Q38_PLE_MANIFEST": str(manifest)}.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += ["--entrypoint", "python3", image, "-m", "vllm.entrypoints.openai.api_server"]
    for arg in profile["args"]:
        cmd.append(json.dumps(arg, separators=(",", ":")) if isinstance(arg, dict) else arg.replace("${MODEL}", str(model)))
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image", default="q38-sm75-4xt10-ssd:dev")
    parser.add_argument("--run", action="store_true", help="Launch experimental profile; requires prepared model and built image")
    args = parser.parse_args()
    cmd = command(args.model, args.manifest, args.image)
    if args.run:
        if not (args.model / "config.json").is_file() or not args.manifest.is_file():
            parser.error("Prepared model config and PLE manifest must exist")
        print("Experimental profile: SM75 W3, full-model capacity and 8GB host RAM are not yet qualified", flush=True)
        raise SystemExit(subprocess.call(cmd))
    print(shlex.join(cmd))
