#!/usr/bin/env python3
"""Queue dual-170HX calibration after source, SSD PLE and dataset are ready.

Does not stop other containers or launch serving. An export remains unvalidated.
Run on the Z4 host, outside Docker, to observe container exit and RAM guards.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--nas-preparation-state", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--ple-manifest", required=True, type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    job = args.job.resolve()
    state_file = job / "pipeline-status.json"
    if args.attempt < 1:
        parser.error("attempt must be positive")
    work = job / f"full-run-{args.attempt:02d}"
    container = "q38-mixed-autoround-20261012" if args.attempt == 1 else f"q38-mixed-autoround-20261012-attempt{args.attempt}"
    calibration_name = "q38-prepare-calibration-20261012"
    def status(phase, **extra):
        data = {"phase": phase, "updated_unix": time.time(), "source": str(args.source), "full_quantization_complete": False, **extra}
        temp = state_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(state_file)
    def container_state(name):
        result = subprocess.run(["docker", "inspect", name, "--format", "{{json .State}}"], text=True, capture_output=True, timeout=15)
        return json.loads(result.stdout) if result.returncode == 0 else None
    try:
        if container_state(container) is not None:
            raise RuntimeError("Full quantization container already exists; inspect before starting another")
        status("waiting_for_inputs")
        while True:
            preparation = json.loads(args.nas_preparation_state.read_text()) if args.nas_preparation_state.exists() else {"phase": "waiting"}
            if preparation["phase"] == "failed":
                raise RuntimeError("NAS source/PLE preparation failed")
            calibration = container_state(calibration_name)
            ready_file = job / "calibration-ready.json"
            validated = ready_file.exists() and json.loads(ready_file.read_text()).get("status") == "passed"
            if calibration and calibration["Status"] == "exited" and calibration["ExitCode"] != 0 and not validated:
                raise RuntimeError("Calibration dataset preparation failed")
            if preparation["phase"] == "ready" and validated and (job / "calibration-tokens.pt").is_file() and (job / "calibration-tokens.json").is_file():
                break
            time.sleep(15)
        while True:
            text = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=15)
            values = [[int(x.strip()) for x in row.split(",")] for row in text.strip().splitlines()]
            if len(values) == 2 and all(free >= 40 * 1024 and temperature < 75 for free, temperature in values):
                break
            status("waiting_for_gpu_capacity", gpu_free_mib_and_temperature=values)
            time.sleep(15)
        # Docker parses this argument as CSV, even when subprocess receives an
        # argv list. Preserve the device list as one quoted CSV field.
        cmd = ["docker", "run", "--detach", "--name", container, "--gpus", '"device=0,1"', "--ipc", "private", "--shm-size", "4g", "--memory", "210g", "--memory-swap", "210g", "--env", "HF_HOME=" + str(job / "dataset-cache"), "--env", "OMP_NUM_THREADS=8", "--mount", f"type=bind,src={job},dst={job}", "--mount", "type=bind,src=/mnt/nas,dst=/mnt/nas,readonly", "q38-quant-sm80:20261012", str(job / "repo/quant/run-autoround.py"), "--source", str(args.source), "--ple-manifest", str(args.ple_manifest), "--work-dir", str(work), "--calibration", str(job / "calibration-tokens.pt")]
        identity = subprocess.check_output(cmd, text=True, timeout=30).strip()
        (job / "full-run-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        status("full_quantization_running", container=container, container_id=identity)
        while True:
            current = container_state(container)
            if not current:
                raise RuntimeError("Full quantization container disappeared")
            quant_state = json.loads((work / "quant-status.json").read_text()) if (work / "quant-status.json").exists() else {}
            status("full_quantization_running", container=container, quantization=quant_state)
            if current["Status"] == "exited":
                if current["ExitCode"] != 0 or quant_state.get("status") != "exported_unvalidated":
                    raise RuntimeError("Full quantization failed; inspect container logs and quant-status.json")
                status("exported_unvalidated", container=container, export=str(work / "export"), quality_and_sm75_validation_pending=True)
                break
            time.sleep(15)
    except BaseException as exc:
        status("failed", error=type(exc).__name__ + ": " + str(exc))
        raise


if __name__ == "__main__":
    main()
