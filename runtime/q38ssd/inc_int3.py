"""Experimental exact-recipe INC INT3 -> Humming MoE bridge.

Opt-in trial adapter: single-matrix / two-expert SM75 probes passed; full
TP4/EP4 deployment remains unqualified. No weight requantization is performed.
Keep W4 on the original backend and reject incomplete/mixed expert metadata.
"""
import re


def expert_scheme(extra, prefix, default_bits, default_group, default_sym=True):
    pattern = re.compile(re.escape(prefix) + r"\.(\d+)\.(gate|up|down)_proj$")
    schemes = {}
    for name, rule in extra.items():
        if match := pattern.fullmatch(name):
            schemes[(int(match[1]), match[2])] = (
                rule.get('bits', default_bits), rule.get('group_size', default_group),
                rule.get('sym', default_sym))
    expected = {(i, p) for i in range(512) for p in ('gate', 'up', 'down')}
    if set(schemes) != expected:
        raise ValueError(f'Incomplete 512-expert recipe for {prefix}')
    values = set(schemes.values())
    if len(values) != 1:
        raise ValueError(f'Mixed expert schemes within {prefix}')
    result = values.pop()
    allowed = ((3, 128, True), (4, 128, True))
    # The separately exported draft block is W4/G32, never W3/G128.
    if re.fullmatch(r'mtp\.layers\.48\.mlp\.experts', prefix):
        allowed = ((4, 32, True),)
    if result not in allowed:
        raise ValueError(f'Unreviewed expert scheme: {result}')
    return result


def install():
    import importlib.metadata
    if importlib.metadata.version('vllm').partition('+')[0] != '0.30.0':
        raise RuntimeError('INT3 bridge requires pinned vLLM 0.30.0')
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.layers.quantization.inc.config_parser import INCConfigParser
    from vllm.model_executor.layers.quantization.inc.schemes import inc_wna16_scheme
    from vllm.model_executor.layers.quantization.humming import HummingLayerQuantizationConfig, HummingMoEMethod
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.quantization.utils.humming_utils import weight_schema_to_quant_key
    from vllm.utils.humming import GPTQWeightSchema

    if getattr(INCConfigParser, '_q38_int3_installed', False):
        return
    original = INCConfigParser._resolve_raw

    def resolve(parser, layer, name):
        if isinstance(layer, ParallelLMHead) and name == 'lm_head':
            # The conditional target mapper prefixes the shared head rule;
            # the standalone MTP head retains the short prefix.
            c = parser._config
            rules = [v for k, v in (c.extra_config or {}).items()
                     if k == 'lm_head' or k.endswith('.lm_head')]
            values = {(v.get('bits', c.weight_bits),
                       v.get('group_size', c.group_size), v.get('sym', c.sym))
                      for v in rules}
            if values == {(4, 32, True)}:
                return 4, 32, True
        if isinstance(layer, RoutedExperts):
            c = parser._config
            return expert_scheme(c.extra_config or {}, name, c.weight_bits, c.group_size, c.sym)
        return original(parser, layer, name)

    class NoActOrderGPTQ3(GPTQWeightSchema):
        def get_tensors_attrs(self, *args, **kwargs):
            attrs = super().get_tensors_attrs(*args, **kwargs)
            # AutoRound emits no g_idx for desc_act=False; the conversion
            # ignores it. Do not create an uninitialized, missing parameter.
            attrs.pop('g_idx', None)
            return attrs

    expected_key = weight_schema_to_quant_key(GPTQWeightSchema(bits=3, group_size=128, sym=True))

    class Q38INT3IndexedExperts(HummingIndexedExperts):
        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key):
            # vLLM's GPTQ schema maps the integer family to an INT4 QuantKey,
            # while the actual Humming config retains uint3. Accept only this
            # exact G128 weight-only bridge; retain all other device checks.
            return weight_key == expected_key and activation_key is None

    class Q38INT3MoEMethod(HummingMoEMethod):
        def __init__(self, config, moe):
            super().__init__(config, moe)
            supported, reason = Q38INT3IndexedExperts.is_supported_config(
                Q38INT3IndexedExperts, moe, expected_key, None,
                mk.FusedMoEActivationFormat.Standard)
            if not supported:
                raise RuntimeError(f'INT3 Humming deployment unsupported: {reason}')
            self.experts_cls = Q38INT3IndexedExperts

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **attrs):
            if (hidden_size, intermediate_size_per_partition) != (2560, 640):
                raise ValueError('INT3 bridge requires EP with full 2560/640 expert matrices')
            return super().create_weights(layer, num_experts, hidden_size,
                                          intermediate_size_per_partition, params_dtype, **attrs)

    original_moe = inc_wna16_scheme._resolve_gptq_moe

    def resolve_moe(layer, config):
        if config.bits == 3:
            if config.group_size != 128 or not config.sym:
                raise ValueError('INT3 bridge requires symmetric G128')
            quant = HummingLayerQuantizationConfig(NoActOrderGPTQ3(bits=3, group_size=128, sym=True))
            return Q38INT3MoEMethod(quant, layer.moe_config)
        return original_moe(layer, config)

    INCConfigParser._resolve_raw = resolve
    inc_wna16_scheme._resolve_gptq_moe = resolve_moe
    INCConfigParser._q38_int3_installed = True
