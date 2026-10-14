"""Opt-in eager diagnostic scopes; CUDA elapsed time includes stream gaps/waits.

Nested categories overlap. Never sum them as exclusive GPU kernel time. A control
file enables collection between requests; disabled runs create no CUDA events.
"""
import functools
import json
import os
import time
from pathlib import Path


def install():
    import torch
    import importlib.metadata
    if importlib.metadata.version("vllm").partition("+")[0] != "0.30.0":
        raise RuntimeError("Stage scopes require the pinned vLLM 0.30.0 engine")
    from vllm.models.qwen4_exp.nvidia import model, mtp
    from vllm.distributed.parallel_state import GroupCoordinator
    from .embedding import SSDPLEEmbedding
    if getattr(model.Qwen4ExpModel, '_q38_stage_installed', False):
        return
    model.Qwen4ExpModel._q38_stage_installed = True
    control = Path(os.environ['Q38_STAGE_PROFILE_CONTROL'])
    output = Path(os.environ['Q38_STAGE_PROFILE_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    state = {'current': None}

    def scope(fn, category):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            ctx = state['current']
            if ctx is None:
                return fn(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            wall = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - wall
                end.record()
                ctx['scopes'].append((category, start, end, elapsed))
        return run

    def root_scope(fn, kind):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            if state['current'] is not None or not control.exists():
                return fn(*args, **kwargs)
            config = json.loads(control.read_text())
            if not config.get('enabled'):
                return fn(*args, **kwargs)
            tokens = None
            for value in list(kwargs.values()) + list(args[1:]):
                if isinstance(value, torch.Tensor) and value.ndim:
                    tokens = value.shape[0]
                    break
            ctx = {'scopes': []}
            state['current'] = ctx
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record()
            wall = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                last.record()
                last.synchronize()
                state['current'] = None
                grouped = {}
                for category, begin, end, cpu in ctx['scopes']:
                    record = grouped.setdefault(category, {'calls': 0, 'stream_ms': 0, 'host_dispatch_ms': 0})
                    record['calls'] += 1
                    record['stream_ms'] += begin.elapsed_time(end)
                    record['host_dispatch_ms'] += cpu * 1000
                record = {'tag': config.get('tag'), 'kind': kind, 'tokens': tokens,
                          'pid': os.getpid(), 'rank': os.environ.get('RANK'),
                          'device': torch.cuda.current_device(), 'time': time.time(),
                          'root_stream_ms': first.elapsed_time(last),
                          'root_wall_ms': (time.perf_counter() - wall) * 1000,
                          'inclusive_scopes': grouped}
                with (output / f'stages-{os.getpid()}.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
        return run

    def patch_init(cls):
        original = cls.__init__
        @functools.wraps(original)
        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            for name, module in self.named_modules():
                leaf = name.rsplit('.', 1)[-1]
                category = {'self_attn': 'qsa_attention', 'linear_attn': 'gdn_attention',
                            'mlp': 'mlp_including_moe_and_communication', 'ple': 'ple_including_ssd'}.get(leaf)
                if category and not getattr(module, '_q38_stage_wrapped', False):
                    module.forward = scope(module.forward, category)
                    module._q38_stage_wrapped = True
        cls.__init__ = init
    for cls, kind in [(model.Qwen4ExpModel, 'target'), (mtp.Qwen4ExpMultiTokenPredictor, 'draft')]:
        patch_init(cls)
        cls.forward = root_scope(cls.forward, kind)
    for name in ['mix', 'combine', 'combine_and_mix']:
        if hasattr(model.GatedResidual, name):
            setattr(model.GatedResidual, name, scope(getattr(model.GatedResidual, name), 'hyper_connection'))
    SSDPLEEmbedding.forward = scope(SSDPLEEmbedding.forward, 'ssd_embedding_including_sync_h2d')
    for name in ['all_reduce', 'all_gather', 'reduce_scatter']:
        if hasattr(GroupCoordinator, name):
            setattr(GroupCoordinator, name, scope(getattr(GroupCoordinator, name), 'collective_' + name))
