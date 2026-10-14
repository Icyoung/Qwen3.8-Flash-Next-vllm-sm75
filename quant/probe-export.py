#!/usr/bin/env python3
"""Small real-GPU AutoRound packing probe; not model calibration or quality eval.

Use one process per GPU to exercise W3 G128 and W4 G32 concurrently. The
architecture has this model's 2560/640 dimensions, but random toy weights.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=[3, 4], required=True)
    parser.add_argument("--group-size", type=int, choices=[32, 128], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.utils.data import DataLoader
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=2560, intermediate_size=640, num_hidden_layers=1, num_attention_heads=20, num_key_value_heads=4, max_position_embeddings=64, tie_word_embeddings=False)).to(dtype=torch.bfloat16).eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")), unk_token="t0", pad_token="t0", eos_token="t1")
    batches = [{"input_ids": torch.randint(0, 128, (16,)), "attention_mask": torch.ones(16, dtype=torch.long)} for _ in range(4)]
    dataset = DataLoader(batches, batch_size=1)
    from auto_round import AutoRound
    started = time.time()
    ar = AutoRound(model, tokenizer=tokenizer, scheme=f"W{args.bits}A16", group_size=args.group_size, dataset=dataset, iters=2, nsamples=4, seqlen=16, batch_size=1, gradient_accumulate_steps=1, low_gpu_mem_usage=True, low_cpu_mem_usage=True, device_map="cuda", enable_torch_compile=False, seed=42)
    ar.quantize_and_save(str(args.output / "checkpoint"), format="auto_round:auto_gptq")
    from safetensors import safe_open
    qweights = []
    for path in (args.output / "checkpoint").rglob("*.safetensors"):
        with safe_open(path, framework="pt") as tensors:
            for name in tensors.keys():
                if name.endswith("qweight"):
                    view = tensors.get_slice(name)
                    qweights.append({"name": name, "dtype": view.get_dtype(), "shape": list(view.get_shape())})
    if not qweights:
        raise RuntimeError("Export completed without packed weight tensors")
    result = {"status": "passed", "scope": "toy-model GPU calibration and export only", "gpu": torch.cuda.get_device_name(0), "bits": args.bits, "group_size": args.group_size, "elapsed_seconds": time.time() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "qweights": qweights, "sm75_inference_qualified": False, "full_model_qualified": False}
    (args.output / "probe-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
