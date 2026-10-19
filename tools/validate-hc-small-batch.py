"""Real checkpoint HC wrappers, all four T10s; no full model resident."""
import argparse, json, os, statistics
from pathlib import Path
import torch
from safetensors import safe_open
from vllm.config import VllmConfig, ModelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_model_parallel, destroy_distributed_environment
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
from q38ssd import hc_small_batch as fusion

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--devices',default='0,1,2,3')
args=parser.parse_args()
if args.output.exists():parser.error('Use a new output path')
ROOT=args.output.resolve().parent
ROOT.mkdir(parents=True,exist_ok=True)
MODEL=args.model.resolve()
devices=[int(x) for x in args.devices.split(',')]
if not devices:parser.error('At least one GPU is required')
torch.set_num_threads(1)
index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
raw=json.loads((MODEL/'config.json').read_text()); eps=raw.get('text_config',raw).get('rms_norm_eps',1e-6)
model_config=ModelConfig(model=str(MODEL),dtype='float16',enforce_eager=True,max_model_len=2048)
context=set_current_vllm_config(VllmConfig(model_config=model_config));context.__enter__()
init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method=f'file://{ROOT}/hc-probe-{os.getpid()}',backend='gloo')
initialize_model_parallel(backend='gloo')
fusion.install()

def tensors(prefix):
    result={}
    for name in ('hc_norm','input_mix_weight_down','input_mix_weight_up','block_inject_weight'):
        key=prefix+name+'.weight'
        if key not in index:continue
        with safe_open(MODEL/index[key],framework='pt',device='cpu') as f:result[name]=f.get_tensor(key).half()
    return result

prefixes=['model.language_model.layers.0.attn_hyper_connection.',
          'model.language_model.layers.24.mlp_hyper_connection.',
          'model.language_model.layers.47.attn_hyper_connection.',
          'model.language_model.hyper_connection_mixer.',
          'mtp.layers.0.mlp_hyper_connection.']
weights={prefix:tensors(prefix) for prefix in prefixes}
results=[]

def compare(a,b):
    if a is None or b is None:
        assert a is b
        return {'relative_l2':0.,'max_abs':0.,'exact':True}
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta=(a.float()-b.float()); denom=a.float().norm().clamp(min=1e-12)
    return {'relative_l2':(delta.norm()/denom).item(),'max_abs':delta.abs().max().item(),'exact':torch.equal(a,b)}

def bench(fn):
    for _ in range(10):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(40):fn()
        b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/40)
    return statistics.median(samples)

with torch.inference_mode():
 for device in devices:
    torch.cuda.set_device(device)
    for prefix in prefixes:
        w=weights[prefix]; combine='block_inject_weight' in w
        with torch.device(f'cuda:{device}'):
            hc=GatedResidual(HyperConnectionConfig(hc_count=4,hidden_size=2560,params_dtype=torch.float16,hc_lowrank=320,rms_norm_eps=eps,hc_per_branch_norm=True),use_combine=combine)
        hc.hc_norm.weight.copy_(w['hc_norm']); hc.input_mix_weight_up.weight.copy_(w['input_mix_weight_up'])
        if combine:
            target=hc.input_mix_weight_down_block_inject.weight
            target.zero_();target[:320].copy_(w['input_mix_weight_down']);target[320:324].copy_(w['block_inject_weight'])
        else:hc.input_mix_weight_down.weight.copy_(w['input_mix_weight_down'])
        for m in (1,2,3,4,5,128):
            for scale in (0.1,1.0,3.0):
                torch.manual_seed(42+m)
                x=torch.randn(m,10240,device=f'cuda:{device}',dtype=torch.float16)*scale
                block=torch.randn(m,2560,device=x.device,dtype=x.dtype)*.1
                injection=torch.randn(m,4,device=x.device,dtype=x.dtype)
                for kind in ('mix','combine_mix'):
                    call=(lambda:hc.mix(x)) if kind=='mix' else (lambda:hc.combine_and_mix(x,block,injection))
                    fusion._enabled=False;ref=call()
                    fusion._enabled=True;out=call()
                    errors=[compare(a,b) for a,b in zip(ref,out)]
                    assert errors[0]['exact'] and errors[2]['exact'],errors
                    if m>4:assert all(e['exact'] for e in errors),errors
                    assert errors[1]['relative_l2']<.001 and errors[1]['max_abs']<.01,errors
                    results.append({'device':device,'prefix':prefix,'m':m,'scale':scale,'kind':kind,'errors':errors})
        if device==devices[0] and prefix==prefixes[1]:
            timings=[]
            for m in (1,4,128):
                x=torch.randn(m,10240,device=f'cuda:{device}',dtype=torch.float16)
                for enabled in (False,True,True,False):
                    fusion._enabled=enabled
                    timings.append({'m':m,'enabled':enabled,'us':bench(lambda:hc.mix(x))})
            args.output.with_suffix('.timings.json').write_text(json.dumps(timings,indent=2)+'\n')
        del hc
    print(json.dumps({'device':device,'validated_cases':len(results),'max_rel_l2':max(r['errors'][1]['relative_l2'] for r in results)}),flush=True)
args.output.write_text(json.dumps({'passed':True,'eps':eps,'cases':results},indent=2)+'\n')
destroy_model_parallel();destroy_distributed_environment();context.__exit__(None,None,None)
