"""Opt-in eager diagnostic scopes; CUDA elapsed time includes stream gaps/waits.

Nested categories overlap. Never sum them as exclusive GPU kernel time. A control
file enables collection between requests; disabled runs create no CUDA events.
"""
import functools
import json
import os
import time
from pathlib import Path


def scheduler_phase(computed, prompt_lengths):
    """Classify actual request progress, including one-token prefill tails."""
    flags = [int(done) < int(prompt) for done, prompt in zip(computed, prompt_lengths)]
    if not flags:
        return 'empty'
    return 'prefill' if all(flags) else 'decode' if not any(flags) else 'mixed'


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
    runner_enabled = os.environ.get('Q38_STAGE_PROFILE_RUNNER') == '1'

    def scope(fn, category, payload=False):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            ctx = state['current']
            if ctx is None:
                return fn(*args, **kwargs)
            nbytes = None
            if payload:
                tensor = next((v for v in list(args[1:]) + list(kwargs.values())
                               if isinstance(v, torch.Tensor)), None)
                if tensor is not None:
                    nbytes = tensor.numel() * tensor.element_size()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            wall = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - wall
                end.record()
                ctx['scopes'].append((category, start, end, elapsed, nbytes))
        return run

    def root_scope(fn, kind):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            if state['current'] is not None:
                return scope(fn, kind + '_forward')(*args, **kwargs)
            if not control.exists():
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
            owner = args[0]
            if kind.startswith('runner_'):
                if kind == 'runner_execute':
                    owner._q38_profile_step = getattr(owner, '_q38_profile_step', 0) + 1
                    owner._q38_profile_phase = 'unknown'
                    scheduled = kwargs.get('scheduler_output') or args[1]
                    owner._q38_profile_tokens = scheduled.total_num_scheduled_tokens
                tokens = getattr(owner, '_q38_profile_tokens', None)
            state['current'] = ctx
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record()
            wall = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dispatch_ms = (time.perf_counter() - wall) * 1000
                last.record()
                last.synchronize()
                state['current'] = None
                grouped = {}
                for category, begin, end, cpu, nbytes in ctx['scopes']:
                    record = grouped.setdefault(category, {'calls': 0, 'stream_ms': 0, 'host_dispatch_ms': 0})
                    record['calls'] += 1
                    record['stream_ms'] += begin.elapsed_time(end)
                    record['host_dispatch_ms'] += cpu * 1000
                    if nbytes is not None:
                        histogram = record.setdefault('input_bytes_histogram', {})
                        histogram[str(nbytes)] = histogram.get(str(nbytes), 0) + 1
                record = {'tag': config.get('tag'), 'kind': kind, 'tokens': tokens,
                          'pid': os.getpid(), 'rank': os.environ.get('RANK'),
                          'device': torch.cuda.current_device(), 'time': time.time(),
                          'root_stream_ms': first.elapsed_time(last),
                          'root_wall_ms': (time.perf_counter() - wall) * 1000,
                          'root_host_dispatch_ms': dispatch_ms,
                          'inclusive_scopes': grouped}
                if kind.startswith('runner_'):
                    record.update(schema=2, step=getattr(owner, '_q38_profile_step', None),
                                  phase=getattr(owner, '_q38_profile_phase', 'unknown'))
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
            setattr(GroupCoordinator, name, scope(getattr(GroupCoordinator, name), 'collective_' + name, payload=True))
    if runner_enabled:
        # The pinned deployment selects V2. V1 has different state and method names.
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
        from vllm.v1.worker.gpu.sample.sampler import Sampler
        from vllm.v1.worker.gpu.async_utils import AsyncOutput
        original_gather = GPUModelRunner.gather_batch_req_state

        @functools.wraps(original_gather)
        def gather(self, *args, **kwargs):
            result = original_gather(self, *args, **kwargs)
            batch, _ = result
            if state['current'] is not None and batch is not None:
                self._q38_profile_phase = scheduler_phase(
                    batch.num_computed_prefill_tokens_np, batch.prefill_len_np)
            return result

        GPUModelRunner.gather_batch_req_state = gather
        for method, category in {
            'update_requests': 'runner_update_requests',
            'gather_batch_req_state': 'runner_batch_state',
            'prepare_inputs': 'runner_prepare_inputs',
            'prepare_attn': 'runner_prepare_attention',
            'sample': 'sampling_including_logits_and_rejection',
            'postprocess_sampled': 'runner_postprocess_sampled',
        }.items():
            setattr(GPUModelRunner, method, scope(getattr(GPUModelRunner, method), category))
        MTPSpeculator.propose = scope(MTPSpeculator.propose, 'draft_proposal_including_models')
        RejectionSampler.__call__ = scope(RejectionSampler.__call__, 'rejection_sampler')
        Sampler.__call__ = scope(Sampler.__call__, 'normal_sampler')
        AsyncOutput.__init__ = scope(AsyncOutput.__init__, 'output_copy_enqueue')
        for cls, category in [(model.Qwen4ExpForCausalLM, 'target_logits'),
                              (mtp.Qwen4ExpMTP, 'draft_logits')]:
            cls.compute_logits = scope(cls.compute_logits, category)
        GPUModelRunner.execute_model = root_scope(GPUModelRunner.execute_model, 'runner_execute')
        GPUModelRunner.sample_tokens = root_scope(GPUModelRunner.sample_tokens, 'runner_sample')
