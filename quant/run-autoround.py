#!/usr/bin/env python3
"""Run calibrated W3/W4 on the pinned original, using both SM80 GPUs.

PLE must already be extracted and excluded from the source-view index. Output
is marked exported-but-unvalidated until independent quality/runtime checks.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "quant"))


def prepare_view(source, output):
    from q38ssd.store import PLE_NAME, read_header
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weights = {n: f for n, f in index["weight_map"].items() if not PLE_NAME.fullmatch(n)}
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".jinja", ".txt") and path.name not in ("model.safetensors.index.json", "q38-fetch-manifest.json"):
            shutil.copyfile(path, output / path.name)
    for filename in set(weights.values()):
        if Path(filename).name != filename or not (source / filename).is_file():
            raise ValueError("Source view contains missing/invalid shard")
        header, data_start, _ = read_header(source / filename)
        if any(PLE_NAME.fullmatch(name) for name in header if name != "__metadata__"):
            # HF may inspect actual file headers, not just the index. Remove
            # PLE bytes from the two mixed boundary shards as well, so checkpoint
            # conversion cannot accidentally reconstruct a partial/full table.
            selected = {name: record for name, record in header.items() if name != "__metadata__" and not PLE_NAME.fullmatch(name)}
            new_header, offset = {}, 0
            if "__metadata__" in header:
                new_header["__metadata__"] = header["__metadata__"]
            for name, record in selected.items():
                start, end = record["data_offsets"]
                new_header[name] = {"dtype": record["dtype"], "shape": record["shape"], "data_offsets": [offset, offset + end - start]}
                offset += end - start
            encoded = json.dumps(new_header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            target = output / filename
            partial = target.with_suffix(target.suffix + ".partial")
            with (source / filename).open("rb") as original, partial.open("xb") as destination:
                destination.write(struct.pack("<Q", len(encoded)) + encoded)
                for record in selected.values():
                    start, end = record["data_offsets"]
                    original.seek(data_start + start)
                    remaining = end - start
                    while remaining:
                        chunk = original.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("Mixed source shard truncated")
                        destination.write(chunk)
                        remaining -= len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            partial.replace(target)
        else:
            (output / filename).symlink_to(source / filename)
    index["weight_map"] = weights
    index.get("metadata", {}).pop("total_size", None)
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def patch_expert_views():
    # Unbind first so already-contiguous expert slices remain mmap-backed.
    # The dedicated quantization container's writable layer is the only target.
    spec = importlib.util.find_spec("auto_round")
    path = Path(spec.origin).parent / "modeling/fused_moe/moe_experts_interface.py"
    original = path.read_text()
    old = """        if not weights_cpu.is_contiguous():
            weights_cpu = weights_cpu.contiguous()
        # Unbind into a tuple of 2D tensors (zero-copy views since contiguous)
        weight_slices = weights_cpu.unbind(0)"""
    new = """        # Q38: preserve contiguous 2D expert views instead of cloning 3D banks.
        weight_slices = weights_cpu.unbind(0)
        if any(not weight.is_contiguous() for weight in weight_slices):
            weight_slices = tuple(weight.contiguous() for weight in weight_slices)"""
    if original.count(old) != 1:
        raise ValueError("AutoRound expert-view code differs from reviewed 0.16.0")
    path.write_text(original.replace(old, new, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--ple-manifest", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.work_dir / "quant-status.json"
    state = {"status": "preflight", "started_unix": time.time(), "output_validated": False}
    def status(value, **extra):
        state.update(status=value, updated_unix=time.time(), **extra)
        tmp = state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_file)
    try:
        versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "auto-round", "accelerate"]}
        if versions["auto-round"] != "0.16.0" or versions["transformers"] != "5.16.1":
            raise ValueError("Quantization environment differs from pinned versions")
        import torch
        if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (8, 0) for i in range(2)):
            raise ValueError("Expected exactly two visible SM80 GPUs")
        source_manifest = json.loads((args.source / "q38-fetch-manifest.json").read_text())
        if source_manifest["commit"] != "de4b8e4d43b917e7706784d8bb445c9af86a3540":
            raise ValueError("Original model revision differs")
        for item in source_manifest["files"]:
            if item["path"].endswith(".safetensors") and (not (args.source / item["path"]).is_file() or (args.source / item["path"]).stat().st_size != item["size"]):
                raise ValueError("Original source is incomplete")
        if shutil.disk_usage(args.work_dir).free < 70 * 1024**3:
            raise ValueError("Less than 70 GiB free for export and temporary files")
        source_view = args.work_dir / "source-view"
        prepare_view(args.source.resolve(), source_view)
        patch_expert_views()
        from calibration_ple import install
        reader = install(args.ple_manifest)
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
        from transformers import AutoTokenizer, AutoProcessor
        status("loading_original", versions=versions, gpu_names=[torch.cuda.get_device_name(i) for i in range(2)])
        # Loading before importing AutoRound's expert interface retains the
        # architecture's original packed expert layout and avoids auto dispatch.
        model = Qwen4ExpForConditionalGeneration.from_pretrained(str(source_view), dtype=torch.bfloat16, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(source_view), local_files_only=True)
        processor = AutoProcessor.from_pretrained(str(source_view), local_files_only=True)
        from runtime_recipe import build_recipe
        recipe = build_recipe(model)
        (args.work_dir / "resolved-layer-config.json").write_text(json.dumps(recipe, indent=2) + "\n")
        norm_groups = {n: m.group_size for n, m in model.named_modules() if type(m).__name__ == "Qwen4ExpTextRMSNorm" and hasattr(m, "group_size")}
        calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
        if len(calibration) != 128 or any(tuple(x["input_ids"].shape) != (512,) for x in calibration):
            raise ValueError("Expected recorded 128 x 512-token calibration dataset")
        from torch.utils.data import DataLoader
        from auto_round import AutoRound
        from auto_round.compressors import base as ar_base
        original_ignore_layers = ar_base.get_predefined_ignore_layers
        def explicit_shared_expert_policy(current_model):
            ignored = original_ignore_layers(current_model)
            if getattr(current_model.config, "model_type", None) == "qwen4_exp":
                # Shared expert matrices have explicit W4/G32 pins here;
                # the gate stays source precision through the exact-name recipe.
                return [name for name in ignored if name != "shared_expert"]
            return ignored
        ar_base.get_predefined_ignore_layers = explicit_shared_expert_policy
        original_build = ar_base.BaseCompressor._build_layer_config
        def build_and_audit(compressor):
            original_build(compressor)
            actual = compressor.layer_config
            mismatches = []
            for name, expected in recipe.items():
                resolved = actual.get(name)
                if expected["bits"] in (3, 4):
                    if resolved is None or resolved.get("bits") != expected["bits"] or resolved.get("group_size") != expected["group_size"]:
                        mismatches.append({"name": name, "expected": expected, "actual": resolved})
                elif resolved is not None and resolved.get("bits", 16) != 16:
                    mismatches.append({"name": name, "expected": expected, "actual": resolved})
            (args.work_dir / "effective-recipe-audit.json").write_text(json.dumps({"passed": not mismatches, "requested_rules": len(recipe), "effective_rules": len(actual), "mismatches": mismatches}, indent=2, default=str) + "\n")
            if mismatches:
                raise ValueError(f"AutoRound changed {len(mismatches)} recipe entries; inspect effective-recipe-audit.json")
        ar_base.BaseCompressor._build_layer_config = build_and_audit
        status("initializing_calibration", target_layers=len(recipe), ple_cache_bytes=reader.reader.capacity * reader.reader.page_bytes)
        # Torch 2.13 AOT autograd fails on this MoE backward with a re-entered
        # contextvars Context. Use eager execution without changing the recipe,
        # calibration samples, optimization steps, or export format.
        ar = AutoRound(model, tokenizer=tokenizer, scheme="W4A16", group_size=32, layer_config=recipe, dataset=DataLoader(calibration, batch_size=1), iters=50, nsamples=128, seqlen=512, batch_size=1, gradient_accumulate_steps=8, device_map="0,1", low_gpu_mem_usage=True, low_cpu_mem_usage=True, enable_torch_compile=False, seed=42)
        # A preloaded multimodal model does not load its processor through
        # AutoRound's path-based loader. Supply it before MLLM calibration starts.
        ar.model_context.processor = processor
        ar.model_context.image_processor = getattr(processor, "image_processor", None)
        for name, group in norm_groups.items():
            if getattr(model.get_submodule(name), "group_size", None) != group:
                raise ValueError(f"AutoRound changed architecture RMSNorm grouping: {name}")
        from stream_export import install as install_streamed_export
        install_streamed_export(ar, status)
        status("caching_inputs", device_map="0,1", iterations=50, nsamples=128, sequence_length=512, torch_compile=False)
        ar.quantize_and_save(str(args.work_dir / "export"), format="auto_round:auto_gptq")
        status("exported_unvalidated", elapsed_seconds=time.time() - state["started_unix"], ple_disk_bytes_read=reader.reader.disk_bytes)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        status("failed", error=type(exc).__name__ + ": " + str(exc))
        raise


if __name__ == "__main__":
    main()
