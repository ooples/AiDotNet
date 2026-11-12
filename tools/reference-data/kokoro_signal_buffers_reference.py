"""Original Kokoro NSF and TorchSTFT, retaining explicit float32 buffers."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import torch

p=argparse.ArgumentParser()
p.add_argument('--source',required=True,type=Path)
a=p.parse_args()
package=types.ModuleType('kokoro'); package.__path__=[str(a.source/'kokoro')]; sys.modules['kokoro']=package
module=importlib.import_module('kokoro.istftnet')
torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(229487)
source=module.SourceModuleHnNSF(sampling_rate=24000,upsample_scale=300,harmonic_num=8,voiced_threshod=10).double()
with torch.no_grad():
    source.l_linear.weight.uniform_(-.3,.3); source.l_linear.bias.fill_(.17)
pitch=torch.tensor([0.,30.,0.,400.],requires_grad=True)
f0=pitch.reshape(1,-1,1).repeat_interleave(300,dim=1)
draws=[]; uniform=torch.rand; normal=torch.randn_like
def rand(*shape,**kwargs):
    result=uniform(*shape,**kwargs)
    draws.append(dict(kind='uniform',shape=list(result.shape),values=result.detach().flatten().tolist()))
    return result
def randn(value,**kwargs):
    result=normal(value,**kwargs)
    draws.append(dict(kind='normal',shape=list(result.shape),values=result.detach().flatten().tolist()))
    return result
with patch('torch.rand',side_effect=rand),patch('torch.randn_like',side_effect=randn):
    harmonic=source(f0)[0]
probe=torch.linspace(-.7,.9,harmonic.numel()).reshape_as(harmonic)
loss=(harmonic*probe).sum(); loss.backward(); assert pitch.grad is None
stft=module.TorchSTFT(20,5,20)
signal=harmonic.detach().flatten()[:80].reshape(1,-1)
magnitude,phase=stft.transform(signal)
spectrum=torch.cat((magnitude,phase),dim=1)
result=dict(torch=torch.__version__,source='unchanged kokoro.istftnet SourceModuleHnNSF and TorchSTFT',
            pitch=pitch.detach().tolist(),draws=draws,harmonic=harmonic.detach().flatten().tolist(),
            probe=probe.flatten().tolist(),loss=loss.item(),
            weight=source.l_linear.weight.detach().flatten().tolist(),bias=source.l_linear.bias.detach().flatten().tolist(),
            weight_gradient=source.l_linear.weight.grad.flatten().tolist(),bias_gradient=source.l_linear.bias.grad.flatten().tolist(),
            window=stft.window.double().tolist(),signal=signal.flatten().tolist(),spectrum=spectrum.flatten().tolist(),spectrum_shape=list(spectrum.shape))
path=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kokoro_signal_buffers_reference.json'
path.write_text(json.dumps(result)+'\n',encoding='utf-8'); print(path)
