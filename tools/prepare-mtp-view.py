"""Attach untouched exported MTP shard to a separate, recipe-audited view."""
from pathlib import Path
import json,struct,hashlib,os
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-root',type=Path,required=True,
                    help='Contains checkpoint-nomtp/, runtime-ssd/, ple/ and mtp/model_extra_tensors.safetensors[.partial]')
args=parser.parse_args()
MODEL=args.model_root.resolve()
f=MODEL/'mtp/model_extra_tensors.safetensors.partial'
if not f.exists(): f=MODEL/'mtp/model_extra_tensors.safetensors'
assert hashlib.file_digest(f.open('rb'),'sha256').hexdigest()=='1aa640d89ba9764791a5d14cc5fa32f739069f06c776c07b0158af753d642ff8'
if f.suffix=='.partial': f=f.rename(f.with_suffix(''))
with f.open('rb') as stream:
 n=struct.unpack('<Q',stream.read(8))[0];h=json.loads(stream.read(n))
assert len(h)==4653 and all(k.startswith('mtp.') for k in h)
q=json.loads((MODEL/'checkpoint-nomtp/quantization_config.json').read_text())
rules={}
for name,v in h.items():
 if name.endswith('.qweight'):
  prefix=name.removesuffix('.qweight');scales=h[prefix+'.scales'];z=h[prefix+'.qzeros'];assert v['dtype']=='I32' and scales['dtype']=='F16'
  k=v['shape'][0]*8;out=v['shape'][1];group=k//scales['shape'][0]
  assert group==32 and scales['shape'][1]==out and z['shape']==[k//32,out//8]
  rules[prefix]={'bits':4,'group_size':32,'sym':True,'data_type':'int'}
 elif name.endswith('.weight'):
  assert v['dtype']=='BF16';rules[name.removesuffix('.weight')]={'bits':16,'group_size':-1,'sym':True,'data_type':'float'}
assert sum('.experts.' in k for k in rules)==1536
for key,value in rules.items():
 q['extra_config'][key.replace('mtp.layers.0.','mtp.layers.48.')]=value
view=MODEL/'runtime-mtp';view.mkdir(exist_ok=True)
for p in (MODEL/'runtime-ssd').iterdir():
 if p.name not in ['config.json','quantization_config.json','model.safetensors.index.json'] and not (view/p.name).exists():(view/p.name).symlink_to(os.path.relpath(p.resolve(),view))
if not (view/f.name).exists(): (view/f.name).symlink_to('../mtp/'+f.name)
c=json.loads((MODEL/'checkpoint-nomtp/config.json').read_text());c['quantization_config']=q
(view/'config.json').write_text(json.dumps(c,indent=2)+'\n');(view/'quantization_config.json').write_text(json.dumps(q,indent=2)+'\n')
i=json.loads((MODEL/'runtime-ssd/model.safetensors.index.json').read_text());i['weight_map'].update({k:f.name for k in h});i.setdefault('metadata',{});total_size=0
for filename in set(i['weight_map'].values()):
 with (view/filename).open('rb') as stream:
  size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
 total_size+=sum(v['data_offsets'][1]-v['data_offsets'][0] for k,v in header.items() if k!='__metadata__')
i['metadata']['total_size']=total_size;(view/'model.safetensors.index.json').write_text(json.dumps(i,indent=2)+'\n')
assert all((view/n).is_file() for n in set(i['weight_map'].values()))
(MODEL/'mtp-preparation.json').write_text(json.dumps({'mtp_sha256':'1aa640d89ba9764791a5d14cc5fa32f739069f06c776c07b0158af753d642ff8','tensor_count':len(h),'mtp_quantized_matrix_count':sum(v['bits']==4 for v in rules.values()),'runtime_view':str(view),'remapped_recipe_layer':48},indent=2)+'\n')
print('MTP view prepared, untouched W4/G32 draft, original controls retained',flush=True)
