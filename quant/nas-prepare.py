#!/usr/bin/env python3
"""Wait for verified original shards, then extract BF16 PLE on the NAS."""
import argparse
import importlib.util
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-state", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ple-output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    def status(phase, **extra):
        data = {"phase": phase, "updated_unix": time.time(), **extra}
        temp = args.state.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(args.state)
    try:
        status("waiting_for_original")
        while True:
            data = json.loads(args.fetch_state.read_text())
            if data["status"] == "failed":
                raise RuntimeError("Original source download failed; inspect fetch-status.json")
            if data["status"] == "complete" and data["verified_shards"] == 131:
                break
            time.sleep(15)
        status("extracting_bf16_ple")
        repo = Path(__file__).resolve().parents[1]
        path = repo / "tools/prepare-ssd-ple.py"
        spec = importlib.util.spec_from_file_location("prepare_ple", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = module.prepare(args.source, args.ple_output)
        status("ready", ple_manifest=str(args.ple_output / "ple-manifest.json"), ple_data_bytes=manifest["data_bytes"])
    except BaseException as exc:
        status("failed", error=type(exc).__name__ + ": " + str(exc))
        raise


if __name__ == "__main__":
    main()
