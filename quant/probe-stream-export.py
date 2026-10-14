#!/usr/bin/env python3
"""Compare ordinary and streaming export bytes in two isolated GPU processes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.utils.data import DataLoader
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=2560, intermediate_size=640, num_hidden_layers=2, num_attention_heads=20, num_key_value_heads=4, max_position_embeddings=64, tie_word_embeddings=False)).to(dtype=torch.bfloat16).eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")), unk_token="t0", pad_token="t0", eos_token="t1")
    data = DataLoader([{"input_ids": torch.randint(0, 128, (16,)), "attention_mask": torch.ones(16, dtype=torch.long)} for _ in range(4)], batch_size=1)
    from auto_round import AutoRound
    ar = AutoRound(model, tokenizer=tokenizer, scheme="W3A16", group_size=128, layer_config={"lm_head": {"bits": 4, "group_size": 32, "sym": True}}, dataset=data, iters=2, nsamples=4, seqlen=16, batch_size=1, gradient_accumulate_steps=1, device_map="cuda", low_gpu_mem_usage=True, low_cpu_mem_usage=True, enable_torch_compile=False, seed=42)
    events = []
    if args.stream:
        from stream_export import install
        install(ar, lambda state, **extra: events.append({"state": state, **extra}))
    ar.quantize_and_save(str(args.output / "checkpoint"), format="auto_round:auto_gptq")
    (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    print(json.dumps({"exported": True, "streaming": args.stream, "events": events}), flush=True)


if __name__ == "__main__":
    main()
