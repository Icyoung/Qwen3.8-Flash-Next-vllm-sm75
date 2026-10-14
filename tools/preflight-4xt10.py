#!/usr/bin/env python3
"""Read-only inventory. Does not qualify P2P, model capacity or host RAM."""
import csv
import io
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path


def inventory():
    report = {"hostname": socket.gethostname(), "logical_cpus": os.cpu_count(), "docker_installed": bool(shutil.which("docker")), "nvcc_installed": bool(shutil.which("nvcc"))}
    if Path("/proc/meminfo").exists():
        mem = {line.split(":", 1)[0]: int(line.split(":", 1)[1].strip().split()[0]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()}
        report.update({"ram_total_bytes": mem["MemTotal"], "ram_available_bytes": mem["MemAvailable"], "swap_total_bytes": mem["SwapTotal"]})
    report["root_free_bytes"] = shutil.disk_usage("/").free
    query = "index,name,uuid,memory.total,memory.used,compute_cap,temperature.gpu,power.limit,driver_version"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True)
    keys = ["index", "name", "uuid", "memory_total_mib", "memory_used_mib", "compute_cap", "temperature_c", "power_limit_w", "driver"]
    report["gpus"] = [dict(zip(keys, [v.strip() for v in row])) for row in csv.reader(io.StringIO(result.stdout))]
    report["four_t10_sm75_present"] = len(report["gpus"]) == 4 and all(g["name"] == "Tesla T10" and g["compute_cap"] == "7.5" and int(g["memory_total_mib"]) == 16384 for g in report["gpus"])
    report.update({"p2p_qualified": False, "model_capacity_qualified": False, "host_ram_qualified": False})
    return report


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
