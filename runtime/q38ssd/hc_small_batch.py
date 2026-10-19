"""Opt-in eager SM75 HC up-projection + gate fusion for one to four rows.

Only the pinned, unquantized Qwen4Exp HC path is changed. Norm, down/injection,
SiLU and residual combine retain the engine implementation. The fp16 projection
rounding boundary is preserved before sigmoid; accumulation stays fp32.
"""
import functools
import hashlib
from pathlib import Path

# Trial controllers may toggle this between requests; production uses the
# process-start Q38_HC_SMALL_BATCH opt-in. No file polling in individual layers.
_enabled = True


def supported_layout(rows, hidden_size, rank, hc_count, dtype, contiguous, capability):
    return (1 <= rows <= 4 and hidden_size == 2560 and rank == 320
            and hc_count == 4 and dtype == 'torch.float16'
            and contiguous and capability == (7, 5))


def install():
    import importlib.metadata
    if importlib.metadata.version('vllm').partition('+')[0] != '0.30.0':
        raise RuntimeError('HC small-batch fusion requires pinned vLLM 0.30.0')
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.qwen4_exp.nvidia import hyperconnection as hc
    from .hc_small_batch_kernel import up_gate

    cls = hc.GatedResidual
    if getattr(cls, '_q38_small_batch_installed', False):
        return
    # Installed after the pinned engine overlay; fail closed on an unknown
    # implementation instead of silently copying different HC math.
    expected = {
        'hyperconnection.py': '42b7a7afa2d3120966feda3fe171a45d8412b907fcd2d2807ab00ebfdecf0aca',
        'ops/hc.py': 'f4af1d4bf4564c6ce18b59f5aa3bc6e4c07a8e785a671de26ff4a92f62927813',
    }
    directory = Path(hc.__file__).parent
    for name, digest in expected.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('HC fusion source mismatch: ' + name)

    original_init = cls.__init__
    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        model_config = get_current_vllm_config().model_config
        if model_config is None or not model_config.enforce_eager:
            raise RuntimeError('Q38_HC_SMALL_BATCH currently requires --enforce-eager')
        up = self.input_mix_weight_up
        # Cache device capability once per module, avoiding a Python CUDA
        # device query for each HC call. Check tensor layout at dispatch time.
        self._q38_hc_sm75 = (up.weight.is_cuda
                            and torch.cuda.get_device_capability(up.weight.device) == (7, 5)
                            and type(up.quant_method) is UnquantizedLinearMethod)

    def eligible(self, hidden):
        up = self.input_mix_weight_up.weight
        return (_enabled and self._q38_hc_sm75 and hidden.ndim == 2
                and hidden.shape[1] == 10240 and hidden.device == up.device
                and up.shape == (10240, 320) and up.dtype == hidden.dtype
                and supported_layout(hidden.shape[0], self.hidden_size,
                                     self.lora_rank, self.hc_count, str(hidden.dtype),
                                     hidden.is_contiguous() and up.is_contiguous(), (7, 5)))

    def finish(self, hidden, xn):
        if self.use_combine:
            projected = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = projected.split(
                [self.lora_rank, self.hc_count, self.pad_size], dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None
        lora = hc.hc_silu(lora, self.hc_count)
        block = up_gate(lora, self.input_mix_weight_up.weight, xn)
        return hidden, block, injection

    original_mix = cls.mix
    @functools.wraps(original_mix)
    def mix(self, hidden_states):
        if not eligible(self, hidden_states):
            return original_mix(self, hidden_states)
        xn = hc.grouped_gemma_rmsnorm(hidden_states, self.hc_norm.weight,
                                    self.config.rms_norm_eps, self.hc_count)
        return finish(self, hidden_states, xn)

    original_combine_mix = cls.combine_and_mix
    @functools.wraps(original_combine_mix)
    def combine_mix(self, hidden_states, prev_block_output, prev_injection):
        if not eligible(self, hidden_states):
            return original_combine_mix(self, hidden_states, prev_block_output, prev_injection)
        hidden, xn = hc.hc_combine_norm(hidden_states, prev_block_output, prev_injection,
                                      self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
        return finish(self, hidden, xn)

    cls.__init__ = init
    cls.mix = mix
    cls.combine_and_mix = combine_mix
    cls._q38_small_batch_installed = True
