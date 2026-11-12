"""Measure the unchanged author's numeric sensitivity with identical excitation draws.

This diagnostic does not alter phases, change a neural forward, or define an
acceptance tolerance. It records whether legal Torch CPU thread settings alone
change a raw phase branch in the author's released model.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--source',required=True,type=Path)
p.add_argument('--release',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
sys.argv=['kokoro_pretrained_reference.py','--source',str(a.source),'--release',str(a.release),
          '--output',str(a.output.with_suffix('.inference.json'))]
scope=runpy.run_path(str(Path(__file__).with_name('kokoro_pretrained_reference.py')))
model,tokens,style=scope['model'],scope['tokens'],scope['style']
results=[]
for promoted,key in [(False,'original'),(True,'precision_matched')]:
    reference=scope[key]
    baseline=np.array(reference['waveform'])
    target=np.array(reference['intermediates']['harmonic_spectrum']).reshape(22,-1)
    for module in model.modules():
        if module.__class__.__name__=='TorchSTFT':
            module.window=torch.hann_window(module.win_length,periodic=True,dtype=torch.float64 if promoted else torch.float32)
    for threads in (1,2,4,8):
        torch.set_num_threads(threads)
        replay=iter(reference['excitation_draws'])
        def draw(kind):
            item=next(replay)
            assert item['kind']==kind
            return torch.tensor(item['values'],dtype=torch.float64).reshape(item['shape'])
        captured={}
        def spectrum(_,inputs): captured['spectrum']=inputs[0].detach().numpy().reshape(22,-1).copy()
        handle=model.decoder.generator.noise_convs[0].register_forward_pre_hook(spectrum)
        with scope['DoubleBuffers']() if promoted else nullcontext(), \
                patch('torch.rand',side_effect=lambda *args,**kwargs: draw('uniform')), \
                patch('torch.randn_like',side_effect=lambda value,**kwargs: draw('normal')):
            waveform,_=model.forward_with_tokens(tokens,style,speed=20.)
        handle.remove()
        error=float(np.max(np.abs(waveform.detach().flatten().numpy()-baseline)))
        phase_difference=np.abs(captured['spectrum'][11:]-target[11:])
        result=dict(promoted_double_buffers=promoted,cpu_threads=threads,
                    waveform_maximum_error=error,raw_phase_branch_changes=int(np.count_nonzero(phase_difference>np.pi)))
        print(json.dumps(result),flush=True)
        if threads==1: assert error==0
        results.append(result)
output=dict(source_revision=scope['result']['source_revision'],checkpoint_sha256=scope['result']['checkpoint_sha256'],
            torch=torch.__version__,unchanged_author_neural_forwards=True,identical_source_random_draws=True,
            acceptance_tolerance_changed=False,results=results)
a.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
