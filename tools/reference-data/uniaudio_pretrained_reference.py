"""Full released UniAudio multi-scale Transformer, using unchanged author methods.

Arguments: pinned UniAudio checkout, original weight directory, output JSON.
The archive's config and trained tensors, rather than the README announcement,
determine dimensions, vocabulary, context length and causal/prefix attention.
"""
import argparse
import collections
from contextlib import nullcontext
from datetime import timedelta
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import typing
import numpy as np
from numpy._core.multiarray import scalar
import torch
import yaml
from torch.utils._python_dispatch import TorchDispatchMode
from omegaconf import DictConfig,ListConfig
from omegaconf.base import ContainerMetadata,Metadata
from omegaconf.nodes import AnyNode

checkout=Path(sys.argv[1]).resolve()
weights=Path(sys.argv[2]).resolve()
destination=Path(sys.argv[3]).resolve()
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('original_uniaudio',checkout/'UniAudio/model.py')
original=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original
spec.loader.exec_module(original)
configuration=yaml.safe_load((weights/'config.yaml').read_text(encoding='utf-8'))
vocabulary=configuration['token_list']
model=original.MegaByteModel(n_vocab=len(vocabulary),n_ctx=configuration['block_size'],
    n_state=configuration['n_embd'],n_head=configuration['n_head'],n_layer=configuration['n_layer'],
    prefix_lm=configuration['prefix_lm'],na_repeat=configuration['non_acoustic_repeat'],
    local_sos=vocabulary.index('<mega_local>'),global_sos=vocabulary.index('<mega_global>')).eval()
metadata=[argparse.Namespace,DictConfig,ListConfig,ContainerMetadata,Metadata,AnyNode,
          collections.defaultdict,typing.Any,int,dict,list,(scalar,'numpy.core.multiarray.scalar'),
          np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType,timedelta]
with torch.serialization.safe_globals(metadata):
    package=torch.load(weights/'ep5-iter340000.checkpoint',map_location='cpu',weights_only=True,mmap=True)
model.load_state_dict(package['model'],strict=True)
del package
gc.collect()
flat=lambda value:value.detach().flatten().tolist()
data=dict(revision='835f07dcf1bdf586b0f427be59e4d48d0320f00e',
          code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
          config={key:configuration[key] for key in ['n_layer','n_head','n_embd','block_size',
                  'prefix_lm','non_acoustic_repeat']},vocabulary=vocabulary,cases=[])
repeat=lambda token:[token]*3
for sequence,prefix in [
    (repeat(2)+repeat(18)+repeat(4)+repeat(3205)+repeat(5)+repeat(8)+
     [131,1159,2187]+repeat(9)+repeat(3),18),
    (repeat(2)+repeat(18)+repeat(4)+repeat(3217)+repeat(3241)+repeat(5)+repeat(12)+
     [137,1165,2193]+repeat(13)+repeat(8)+[141,1171,2201]+[145,1177,2207]+repeat(9)+repeat(3),30)]:
    data['cases'].append(dict(sequence=sequence,prefix_tokens=prefix))
class Float64OperatorPrecision(TorchDispatchMode):
    """Isolate arithmetic precision; no original neural forward method is replaced."""
    def __init__(self):super().__init__();self.casts=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        cast_dtype=kwargs.get('dtype') if func is torch.ops.aten._to_copy.default else \
            args[1] if func is torch.ops.aten.to.dtype else None
        if args and isinstance(args[0],torch.Tensor) and args[0].dtype==torch.float64 and cast_dtype==torch.float32:
            self.casts+=1
            return args[0]
        return func(*args,**kwargs)

for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64_mixed_norm_softmax'),
                    (torch.float64,'float64_operator_precision')]:
    model.to(dtype)
    if label=='float64_mixed_norm_softmax':
        # The original LayerNorm explicitly casts its input to float32. Keep its
        # trained affine tensors in float32 too, as required by the unchanged
        # author method. Attention likewise retains its original float32 softmax.
        for module in model.modules():
            if isinstance(module,original.LayerNorm):module.float()
    for item in data['cases']:
        sequence=torch.tensor([item['sequence']])
        mask=torch.zeros_like(sequence,dtype=torch.bool)
        mask[:,item['prefix_tokens']:]=True
        precision=Float64OperatorPrecision() if label=='float64_operator_precision' else None
        with torch.inference_mode(),precision if precision is not None else nullcontext():
            logits=model(sequence,mask)
            _,loss,_=original.CrossEntropyAndAccuracy(logits,sequence,mask,
                        prefix_lm=configuration['prefix_lm'],ignore_id=0)
        item[label]=dict(logits=flat(logits),argmax=flat(logits.argmax(-1)),loss=loss.item())
        if precision is not None:
            if precision.casts==0:raise RuntimeError('Expected original forced-float32 operations were not reached')
            item[label]['isolated_float32_casts']=precision.casts
        print('Original complete pretrained UniAudio:',label,len(item['sequence']),flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original full UniAudio checkpoint references:',destination,flush=True)
