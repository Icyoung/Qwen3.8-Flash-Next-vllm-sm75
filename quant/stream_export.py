"""Bounded per-block export for AutoRound 0.16's uncached output-head path.

AutoRound already uses RTN for a head with no cached activations. Its generic
outside-block guard nevertheless disables packing and retains every block in
BF16. Only enable streaming when the actual cache proves that exact case;
reject calibrated outside layers rather than silently changing their algorithm.
"""
import time


def install(ar, report, outside_layers=("lm_head",)):
    from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

    original_cache = ar.cache_data
    activated = False

    def cache_and_enable(block_names, nsamples, layer_names=None, **kwargs):
        nonlocal activated
        result = original_cache(block_names, nsamples, layer_names, **kwargs)
        if block_names and not activated:
            if set(layer_names or ()) != set(outside_layers):
                raise ValueError("Streaming export requires exactly the reviewed outside layers")
            if any(name in result for name in outside_layers):
                raise ValueError("Outside layer activations were cached; use a calibrated streaming-head path")
            if len(ar.formats) != 1 or not ar.formats[0].is_supported_immediate_packing() or not ar.formats[0].is_supported_immediate_saving():
                raise ValueError("Export format does not support per-block packing and saving")
            # This preserves the existing no-input RTN head path. Reference and
            # quantized block outputs are computed before AutoRound packs them.
            ar.inplace = True
            ar.compress_context.is_immediate_packing = True
            ar.compress_context.is_immediate_saving = True
            ar.compress_context.low_cpu_mem_usage = False
            ar._ensure_shard_writer()
            activated = True
            report("training_prepared", streaming_export=True, outside_layer_policy={name: "RTN W4/G32, no activation cache" for name in outside_layers})
        return result

    ar.cache_data = cache_and_enable
    original_quantize = SignRoundQuantizer.quantize_block
    original_step = SignRoundQuantizer._step
    active = {}

    def quantize_with_progress(quantizer, block, *args, **kwargs):
        import inspect
        bound = inspect.signature(original_quantize).bind(quantizer, block, *args, **kwargs)
        ctx = bound.arguments["block_ctx"]
        active.update(block=ctx.block_name, block_index=ctx.block_index, iterations_completed=0, block_started_unix=time.time())
        report("training", **active, total_blocks=ctx.block_cnt)
        result = original_quantize(quantizer, block, *args, **kwargs)
        report("block_calibrated", **active, completed_training_blocks=ctx.block_index + 1)
        return result

    def step_with_progress(quantizer, *args, **kwargs):
        result = original_step(quantizer, *args, **kwargs)
        if active:
            active["iterations_completed"] += 1
            report("training", **active)
        return result

    SignRoundQuantizer.quantize_block = quantize_with_progress
    SignRoundQuantizer._step = step_with_progress
